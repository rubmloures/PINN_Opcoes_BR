# /src/model.py

import torch
import torch.nn as nn
import numpy as np

#  Função de ativação mapeamento       
def get_activation_function(activation_name: str):
    """
    Mapeia nome da ativação (string) para função PyTorch.
    """
    activation_map = {
        'relu': nn.ReLU(),
        'silu': nn.SiLU(),
        'swish': nn.SiLU(),  
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.1),
    }
    
    if activation_name.lower() not in activation_map:
        raise ValueError(f"Ativação '{activation_name}' não suportada. Opções: {list(activation_map.keys())}")
    
    return activation_map[activation_name.lower()]

# Modelos
class AdaptiveActivation(nn.Module):
    """
    Função de ativação adaptativa (com parâmetro treinável 'a').
    Melhora a convergência em PINNs ao ajustar a inclinação/escala da ativação.
    Forma: f(x) = activation(a * x)
    """
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
        self.a = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.activation_fn(self.a * x)

# Fourier Features Layer
class FourierFeatureLayer(nn.Module):
    """
    Mapeia coordenadas de entrada para features de Fourier.
    Input: x (Batch, Dim)
    Output: [sin(B*x), cos(B*x)] (Batch, 2*Mapping_Size)
    """
    def __init__(self, input_dim, mapping_size, sigma=10.0):
        super().__init__()
        # B: Matriz de frequências aleatórias fixas (não treináveis)
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * sigma, requires_grad=False)
        self.mapping_size = mapping_size

    def forward(self, x):
        # Projeção: x @ B
        # x: [Batch, Dim], B: [Dim, Mapping] -> proj: [Batch, Mapping]
        proj = 2 * np.pi * (x @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    
class HestonParameterLayer(nn.Module):
    """
    Camada especializada para mapear a saída latente da LSTM para os
    intervalos físicos válidos dos parâmetros de Heston.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # Camada linear que recebe o estado oculto da LSTM
        # Saída tem dimensão 5: [nu, theta, kappa, xi, rho]
        self.linear = nn.Linear(hidden_size, 5)
        
    def forward(self, x):
        raw = self.linear(x)
        
        # Fatiamento para aplicar restrições específicas a cada parâmetro
        # nu (variância instantânea), theta (var média), kappa (mean rev), xi (vol of vol), rho (correlação)
        # Todos devem ser estritamente positivos -> Softplus (com estabilidade numérica mínima 1e-6)
        
        nu    = nn.functional.softplus(raw[:, 0:1]) + 1e-6    # Volatilidade Instantânea
        theta = nn.functional.softplus(raw[:, 1:2]) + 1e-6    # Volatilidade de Longo Prazo
        kappa = nn.functional.softplus(raw[:, 2:3]) + 1e-6    # Velocidade de Reversão
        xi    = nn.functional.softplus(raw[:, 3:4]) + 1e-6    # Volatilidade da Volatilidade     
        rho   = torch.tanh(raw[:, 4:5])                       # rho deve estar entre [-1, 1] -> Tanh
        
        return nu, theta, kappa, xi, rho

class DeepHestonHybrid(nn.Module):
    """
    Arquitetura Híbrida: LSTM (Time-Series) + FiLM (Modulation) + PINN (Physics-Informed).
    
    Fluxo de Dados:
    1. Sequence History (LSTM) -> Regime de Mercado Base
    2. Asset ID -> FiLM Layer -> Modula o Regime de Mercado (Escala e Shift Específicos do Ativo)
    3. Estado Modulado -> Parâmetros Heston
    4. [Parâmetros Heston + Dados do Contrato] -> PINN -> Preço
    """
    def __init__(self, config: dict, data_stats: dict):
        super().__init__()
        self.config = config
        self.stats = data_stats
        
        # --- Asset Embeddings & FiLM ---
        self.use_embedding = config.get('use_asset_embeddings', False)
        lstm_input_dim = config.get('lstm_input_size', 2)
        lstm_hidden_size = config['lstm_hidden_size']
        
        if self.use_embedding:
            num_assets = config.get('num_assets', 10)
            embedding_dim = config.get('asset_embedding_dim', 8)
            self.asset_embedding = nn.Embedding(num_assets, embedding_dim)
            
            # Input Conditioning: Concatena na entrada da LSTM
            lstm_input_dim += embedding_dim 
            
            # FiLM Generator: Gera Gamma (scale) e Beta (shift) a partir do embedding
            # Output size = 2 * hidden_size (um vetor para gamma, um para beta)
            self.film_generator = nn.Linear(embedding_dim, 2 * lstm_hidden_size)

        # --- MÓDULO 1: O Analista (LSTM) ---
        # Captura a dinâmica temporal do ativo subjacente
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=config['lstm_layers'],
            batch_first=True,
            dropout=config['lstm_dropout']
        )
        
        self.heston_head = HestonParameterLayer(lstm_hidden_size)
        
        
        # --- MÓDULO 2: O Físico (PINN) ---
        # Aproxima a solução V(S, K, T, r, params)
        pinn_input_size = 5 + 5  # 5 (S,K,T,r, q) + 5 (Heston)
        
        # Camadas Fourier
        self.use_fourier = config.get('use_fourier_features', True)
        if self.use_fourier:
            self.fourier_features = config.get('fourier_features', 128)
            self.fourier_sigma = config.get('fourier_sigma', 10.0)
            self.fourier_layer = FourierFeatureLayer(pinn_input_size, self.fourier_features, self.fourier_sigma)
            dense_input_size = 2 * self.fourier_features
        else:
            dense_input_size = pinn_input_size

        # Construção da MLP Densa (PINN)
        layers = []
        dropout_pinn = config.get('dropout', 0.0)
        
        deep_layers_config = config.get('deep_layers', None)
        if deep_layers_config is None:
            pinn_hidden_layers = config.get('pinn_hidden_layers', 6)
            pinn_neurons = config.get('pinn_neurons', 128)
            deep_layers_config = [pinn_neurons] * pinn_hidden_layers
        
        activation_str = config.get('activation', 'silu')
        activation_fn = get_activation_function(activation_str)
        
        # Camada de entrada
        layers.append(nn.Linear(dense_input_size, deep_layers_config[0]))
        layers.append(AdaptiveActivation(activation_fn))
        if dropout_pinn > 0: layers.append(nn.Dropout(dropout_pinn))
        
        # Camadas ocultas
        for i in range(len(deep_layers_config) - 1):
            layers.append(nn.Linear(deep_layers_config[i], deep_layers_config[i+1]))
            layers.append(AdaptiveActivation(activation_fn))
            if dropout_pinn > 0: layers.append(nn.Dropout(dropout_pinn))
            
        # Camada de saída (Preço)
        layers.append(nn.Linear(deep_layers_config[-1], 1))
        
        self.pricing_net = nn.Sequential(*layers)

    def forward(self, x_seq: torch.Tensor, x_phy: torch.Tensor, asset_ids: torch.Tensor = None):
        """
        Args:
            x_seq: Tensor [Batch, Seq_Len, Features]
            x_phy: Tensor [Batch, 5]
            asset_ids: Tensor [Batch]
        """
        # 1. Processamento Temporal (LSTM)
        self.lstm.flatten_parameters()
        
        # Lógica de Embedding + FiLM
        gamma = None
        beta = None
        
        if self.use_embedding and asset_ids is not None:
            # Proteção de bounds
            max_asset_id = asset_ids.max().item()
            num_embeddings = self.asset_embedding.num_embeddings
            if max_asset_id >= num_embeddings:
                 asset_ids = torch.clamp(asset_ids, max=num_embeddings - 1)
            
            # [Batch, Emb_Dim]
            emb = self.asset_embedding(asset_ids) 
            
            # A. Input Conditioning (Concatenação)
            emb_seq = emb.unsqueeze(1).repeat(1, x_seq.size(1), 1)
            lstm_input = torch.cat([x_seq, emb_seq], dim=2)
            
            # B. FiLM Generation (Modulação)
            # Gera [Batch, 2*Hidden]
            film_params = self.film_generator(emb)
            # Divide em Gamma (Escala) e Beta (Shift)
            gamma, beta = torch.split(film_params, self.config['lstm_hidden_size'], dim=1)
            
        else:
            lstm_input = x_seq

        # LSTM Pass
        _, (h_n, _) = self.lstm(lstm_input)
        market_state = h_n[-1] # [Batch, Hidden_Size]
        
        # Aplicação do FiLM (Feature-wise Linear Modulation)
        # Se tivermos embeddings, modulamos o estado oculto da LSTM
        # Isso permite que cada ativo tenha sua própria "escala" e "base" de volatilidade
        if gamma is not None and beta is not None:
            # Fórmula FiLM: h_new = h_old * (1 + gamma) + beta
            market_state = market_state * (1.0 + gamma) + beta
        
        # 2. Determinação dos Parâmetros de Heston (Baseado no estado modulado)
        nu, theta, kappa, xi, rho = self.heston_head(market_state)
        
        # 3. Preparação para a PINN
        pinn_input = torch.cat([x_phy, nu, theta, kappa, xi, rho], dim=1)
        
        # 4. Embeddings (Fourier ou Direto)
        if self.use_fourier:
            pinn_features = self.fourier_layer(pinn_input)
        else:
            pinn_features = pinn_input
            
        # 5. Precificação
        raw_output = self.pricing_net(pinn_features)
        
        # --- Hard Constraints com Payoff Físico Real ---
        # Desnormalizações
        S_norm = x_phy[:, 0:1]
        K_norm = x_phy[:, 1:2]
        T_norm = x_phy[:, 2:3]
        
        S_real = S_norm * self.stats['S_std'] + self.stats['S_mean']
        K_real = K_norm * self.stats['K_std'] + self.stats['K_mean']
        
        # Payoff físico
        intrinsic_value = torch.relu((S_real - K_real) / (K_real + 1e-8))
        
        # Fator de tempo
        T_real = T_norm * self.stats['T_std'] + self.stats['T_mean']
        time_factor = 1.0 - torch.exp(-10.0 * torch.relu(T_real))
        
        # Valor temporal (Time Value)
        time_value = time_factor * nn.functional.softplus(raw_output)
        
        # Preço final
        price = intrinsic_value + time_value

        return {
            'price': price,
            'heston_params': (nu, theta, kappa, xi, rho)
        }