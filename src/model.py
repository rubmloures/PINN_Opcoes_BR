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
        # nu (variância instantânea), theta (var média), kappa (mean rev), xi (vol of vol)
        # Todos devem ser estritamente positivos -> Softplus
        nu    = nn.functional.softplus(raw[:, 0:1]) + 1e-4  # Volatilidade Instantânea
        theta = nn.functional.softplus(raw[:, 1:2]) + 1e-4  # Volatilidade de Longo Prazo
        kappa = nn.functional.softplus(raw[:, 2:3]) + 1e-4  # Velocidade de Reversão
        xi    = nn.functional.softplus(raw[:, 3:4]) + 1e-4  # Volatilidade da Volatilidade     
        rho   = torch.tanh(raw[:, 4:5]) # rho (correlação ativo-vol) deve estar entre [-1, 1] -> Tanh
        return nu, theta, kappa, xi, rho

class DeepHestonHybrid(nn.Module):
    """
    Arquitetura Híbrida: LSTM (Time-Series) + PINN (Physics-Informed).
    
    Fluxo de Dados:
    1. Sequence History (LSTM) -> Regime de Mercado (Estado Oculto)
    2. Estado Oculto -> Parâmetros Heston (nu, theta, kappa, xi, rho)
    3. [Parâmetros Heston + Dados do Contrato (S,K,T)] -> PINN -> Preço da Opção
    """
    def __init__(self, config: dict, data_stats: dict):
        super().__init__()
        self.config = config
        self.stats = data_stats
        
        # --- Learnable Bias Parameter ---
        # Inicializado com valor negativo para ajudar a convergência (preços tendem a ser subestimados)
        self.price_bias = nn.Parameter(torch.tensor(-0.02))
        
        # --- Asset Embeddings ---
        self.use_embedding = config.get('use_asset_embeddings', False)
        lstm_input_dim = config.get('lstm_input_size', 2)
        if self.use_embedding:
            num_assets = config.get('num_assets', 10)
            embedding_dim = config.get('asset_embedding_dim', 8)
            self.asset_embedding = nn.Embedding(num_assets, embedding_dim)
            lstm_input_dim += embedding_dim  # Ajusta o input size da LSTM

        # --- MÓDULO 1: O Analista (LSTM) ---
        # Captura a dinâmica temporal do ativo subjacente
        self.lstm_input_size = lstm_input_dim
        self.lstm_hidden_size = config.get('lstm_hidden_size', 64)
        self.lstm_layers = config.get('lstm_layers', 2)
        dropout_lstm = config.get('lstm_dropout', 0.0)
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=dropout_lstm if self.lstm_layers > 1 else 0.0
        )
        # Cabeça que converte o pensamento da LSTM em parâmetros físicos
        self.heston_head = HestonParameterLayer(self.lstm_hidden_size)
        
        # --- MÓDULO 2: O Físico (PINN) ---
        # Aproxima a solução V(S, K, T, r, params)
        # Inputs da PINN:
        # 5 do contrato: S_norm, K_norm, T_norm, r, q
        # 5 de Heston: nu, theta, kappa, xi, rho
        pinn_input_size = 5 + 5  # 5 (S,K,T,r, q) + 5 (Heston)
        
        # Camadas Fourier, recomendado para alta frequência/precisão
        self.use_fourier = config.get('use_fourier_features', True)
        if self.use_fourier:
            self.fourier_features = config.get('fourier_features', 128)
            self.fourier_sigma = config.get('fourier_sigma', 10.0)
            self.fourier_layer = FourierFeatureLayer(pinn_input_size, self.fourier_features, self.fourier_sigma)
            
            # O input da próxima camada será sin() + cos() das features
            dense_input_size = 2 * self.fourier_features
        else:
            dense_input_size = pinn_input_size

        # Construção da MLP Densa (PINN)
        layers = []
        
        #Definir dropout_pinn antes do if para evitar erro de escopo
        dropout_pinn = config.get('dropout', 0.0)
        
        # Recupera configurações da rede densa com fallback
        deep_layers_config = config.get('deep_layers', None)
        if deep_layers_config is None:
            # Gera arquitetura automática baseado em pinn_hidden_layers e pinn_neurons
            pinn_hidden_layers = config.get('pinn_hidden_layers', 6)
            pinn_neurons = config.get('pinn_neurons', 128)
            deep_layers_config = [pinn_neurons] * pinn_hidden_layers
        
        # Função de ativação
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
            x_seq: Tensor [Batch, Seq_Len, Features] -> Histórico do ativo
            x_phy: Tensor [Batch, 5] -> [S_norm, K_norm, T_norm, r, q] (Z-score normalized)
            asset_ids: Tensor [Batch] -> IDs dos ativos para embeddings (opcional)
            
        Returns:
            dict: {'price': tensor, 'heston_params': tuple(nu, theta...)}
        """
        # 1. Processamento Temporal (LSTM)
        self.lstm.flatten_parameters()
        
        # Embeddings e LSTM
        if self.use_embedding and asset_ids is not None:
            # Validar bounds de asset_ids
            max_asset_id = asset_ids.max().item()
            num_embeddings = self.asset_embedding.num_embeddings
            
            if max_asset_id >= num_embeddings:
                import logging as _logging
                _logger = _logging.getLogger(__name__)
                _logger.warning(
                    f"Asset ID {max_asset_id} >= num_embeddings {num_embeddings}. "
                    f"Clampando para {num_embeddings - 1}."
                )
                asset_ids = torch.clamp(asset_ids, max=num_embeddings - 1)
            
            # [Batch, Emb_Dim]
            emb = self.asset_embedding(asset_ids) 
            # Repete para o tamanho da sequência: [Batch, Seq, Emb_Dim]
            emb_seq = emb.unsqueeze(1).repeat(1, x_seq.size(1), 1)
            # Concatena com features temporais
            lstm_input = torch.cat([x_seq, emb_seq], dim=2)
        else:
            lstm_input = x_seq

        # self.lstm retorna: output, (h_n, c_n)
        # Pegamos h_n[-1] que é o estado oculto da última camada no último passo de tempo
        _, (h_n, _) = self.lstm(lstm_input)
        market_state = h_n[-1] # Shape: [Batch, Hidden_Size]
        
        # 2. Determinação dos Parâmetros de Heston
        nu, theta, kappa, xi, rho = self.heston_head(market_state)
        
        # 3. Preparação para a PINN
        # Concatenar os inputs do contrato com os parâmetros inferidos
        # x_phy: [S_norm, K_norm, T_norm, r, q]
        pinn_input = torch.cat([x_phy, nu, theta, kappa, xi, rho], dim=1)
        
        # 4. Embeddings (Fourier ou Direto)
        if self.use_fourier:
            pinn_features = self.fourier_layer(pinn_input)
        else:
            pinn_features = pinn_input
            
        # 5. Precificação
        raw_output = self.pricing_net(pinn_features)
        
        # --- CORRECTED: Hard Constraints com Payoff Físico Real ---
        # Desnormalizar S e K para calcular moneyness correto
        S_norm = x_phy[:, 0:1]
        K_norm = x_phy[:, 1:2]
        T_norm = x_phy[:, 2:3]
        
        # Desnormalização Z-score -> escala real
        S_real = S_norm * self.stats['S_std'] + self.stats['S_mean']
        K_real = K_norm * self.stats['K_std'] + self.stats['K_mean']
        
        # Payoff físico correto: (S - K) / K
        # Isso garante que o payoff esteja na mesma escala do target (P/K)
        intrinsic_value = torch.relu((S_real - K_real) / (K_real + 1e-8))
        
        # Fator de tempo que zera no vencimento (T=0)
        # T_norm está em Z-score, então precisa ser desnormalizado para usar como fator
        T_real = T_norm * self.stats['T_std'] + self.stats['T_mean']
        time_factor = 1.0 - torch.exp(-10.0 * torch.relu(T_real))
        
        # Valor temporal aprendido pela rede (sempre positivo via softplus)
        time_value = time_factor * nn.functional.softplus(raw_output)
        
        # Preço final = Payoff intrínseco + Valor temporal + Bias aprendível
        price = intrinsic_value + time_value + self.price_bias

        return {
            'price': price,
            'heston_params': (nu, theta, kappa, xi, rho)
        }