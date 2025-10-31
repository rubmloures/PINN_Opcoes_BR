# /src/model.py

import torch
import torch.nn as nn

class AdaptiveActivation(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn
        # O parâmetro 'a' é inicializado como 1 e será ajustado durante o treinamento
        self.a = nn.Parameter(torch.ones(1)) 

    def forward(self, x):
        # move apenas para operação (não reatribui Parameter) para evitar problemas
        a = self.a.to(x.device)
        return self.activation_fn(a * x)

class PINN_BlackScholes(nn.Module):
    """
    Arquitetura de PINN híbrida para o problema de Black-Scholes.
    Pode operar em dois modos:
    1. FORWARD: Prevê o preço da opção dada a volatilidade.
    2. INVERSE: Prevê a volatilidade implícita dado o preço da opção.
    
    Também suporta duas arquiteturas de saída:
    1. ORIGINAL: Baseada no seu código inicial com hard constraints.
    2. PAYOFF_INSPIRED: Usa uma função de "gating" para modelar o payoff.
    """
    def __init__(self, config: dict, data_stats: dict):
        super().__init__()
        self.config = config
        self.stats = data_stats # Contém S_max, S_min, etc., para desnormalização
        self.problem_type = config.get('problem_type', 'INVERSE')
        self.architecture = config.get('architecture', 'PAYOFF_INSPIRED')
        
        input_size = self.config['input_size']
        fourier_features = self.config['fourier_features']
        fourier_sigma = self.config['fourier_sigma']
        
        # Camada de Mapeamento de Fourier
        self.fourier = nn.Linear(input_size, fourier_features)
        nn.init.normal_(self.fourier.weight, mean=0, std=fourier_sigma)
        self.fourier.weight.requires_grad = False # Congela os pesos para ser um mapeamento fixo

        # Componente Raso (Shallow)
        shallow_layers = self.config['shallow_layers']
        self.shallow = self._build_sequential([2 * fourier_features] + shallow_layers)

        # Componente Profundo (Deep)
        deep_layers = self.config['deep_layers']
        self.deep = self._build_sequential([2 * fourier_features] + deep_layers)
        
        # Camada Combinadora
        combiner_input_size = shallow_layers[-1] + deep_layers[-1]
        self.combiner = nn.Linear(combiner_input_size, 256) # Camada intermediária antes da saída
        # AdaptiveActivation para o combiner — criado no __init__ para garantir que seus
        # parâmetros sejam registrados e movidos com o modelo para o dispositivo correto.
        self._adaptive_combiner = AdaptiveActivation(self.config['activation_fn'])
        
        # Cabeças de Saída (Output Heads)
        if self.problem_type == 'INVERSE':
            # Saída para o preço (usado na perda de dados) e para a volatilidade
            self.price_head = nn.Linear(256, 1)
            self.vol_head = nn.Linear(256, 1)
        else: # FORWARD
            self.price_head = nn.Linear(256, 1)

    def _build_sequential(self, layer_dims):
        """Construtor auxiliar para criar blocos de camadas sequenciais."""
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(AdaptiveActivation(self.config['activation_fn']))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Executa a passagem forward da rede.
        
        Args:
            x (torch.Tensor): Tensor de entrada normalizado.
        
        Returns:
            dict: Um dicionário contendo as saídas relevantes (preço, volatilidade).
        """
        # Desnormaliza as entradas para cálculos físicos
        S_norm, K_norm, T_norm, r = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        
        S = S_norm * (self.stats['S_max'] - self.stats['S_min']) + self.stats['S_min']
        K = K_norm * (self.stats['K_max'] - self.stats['K_min']) + self.stats['K_min']
        T = T_norm * self.stats['T_max']

        # Mapeamento de Fourier
        x_fourier = torch.cat([
            torch.sin(self.fourier(x)),
            torch.cos(self.fourier(x))
        ], dim=1)

        # Passagem pelos componentes Shallow e Deep
        shallow_out = self.shallow(x_fourier)
        deep_out = self.deep(x_fourier)

        # Combinação e saída bruta
        combined = torch.cat([shallow_out, deep_out], dim=1)
        # usa a instância criada no __init__ (parâmetros já registrados e movidos com o modelo)
        combined_features = self._adaptive_combiner(self.combiner(combined))
        
        outputs = {}

        if self.problem_type == 'INVERSE':
            # A rede aprende a volatilidade
            # Usa softplus para garantir que a volatilidade seja sempre > 0
            sigma = nn.functional.softplus(self.vol_head(combined_features))
            outputs['sigma'] = sigma
        else: # FORWARD
            # A volatilidade é uma entrada (a 5ª coluna)
            sigma = x[:, 4:5]
            outputs['sigma'] = sigma

        # Calcula o preço usando a arquitetura de saída escolhida
        intrinsic_value = torch.relu(S - K)
        
        if self.architecture == 'PAYOFF_INSPIRED':
            gating_factor = torch.sigmoid(self.price_head(combined_features))
            price = (1 - gating_factor) * intrinsic_value + gating_factor * S
        
        elif self.architecture == 'ORIGINAL':
            time_decay = 1 - torch.exp(-r * T)
            # A constante 'A' pode ser um hiperparâmetro ou até mesmo aprendida
            A = 2.0 
            price_raw = self.price_head(combined_features)
            price = intrinsic_value + time_decay * (S * A * torch.sigmoid(price_raw))
        
        else:
            raise ValueError("Arquitetura de modelo desconhecida especificada na configuração.")

        outputs['price'] = price
        
        return outputs