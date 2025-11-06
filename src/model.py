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
        self.stats = data_stats
        self.problem_type = config.get('problem_type', 'INVERSE')  # 'INVERSE' ou 'FORWARD'
        self.architecture = config.get('architecture', 'PAYOFF_INSPIRED')

        # tamanhos
        input_size_full = self.config['input_size']            # ex: 5 (S,K,T,r,premium)
        input_size_price = input_size_full - 1                 # ex: 4 (S,K,T,r)
        fourier_features = self.config['fourier_features']
        fourier_sigma = self.config['fourier_sigma']
        shallow_layers = self.config['shallow_layers']
        deep_layers = self.config['deep_layers']

        # --- Map. Fourier separado para INVERSE (usa premium) e PRICE (sem premium) ---
        # Fourier for INVERSE route (input_size_full)
        self.fourier_full = nn.Linear(input_size_full, fourier_features)
        nn.init.normal_(self.fourier_full.weight, mean=0, std=fourier_sigma)
        self.fourier_full.weight.requires_grad = False  # mapeamento fixo

        # Fourier for PRICE route (input_size_price)
        self.fourier_price = nn.Linear(input_size_price, fourier_features)
        nn.init.normal_(self.fourier_price.weight, mean=0, std=fourier_sigma)
        self.fourier_price.weight.requires_grad = False

        # --- Blocos Shallow / Deep separados para cada rota (claridade/independência) ---
        self.shallow_inv = self._build_sequential([2 * fourier_features] + shallow_layers)
        self.deep_inv    = self._build_sequential([2 * fourier_features] + deep_layers)

        self.shallow_price = self._build_sequential([2 * fourier_features] + shallow_layers)
        self.deep_price    = self._build_sequential([2 * fourier_features] + deep_layers)

        # --- Camadas combinadoras separadas (inv / price) ---
        combiner_input_size = shallow_layers[-1] + deep_layers[-1]
        self.combiner_inv = nn.Linear(combiner_input_size, 256)
        self.combiner_price = nn.Linear(combiner_input_size, 256)

        # Adaptive activation (pode ser compartilhada)
        self._adaptive_combiner = AdaptiveActivation(self.config['activation_fn'])

        # --- Cabeças de saída ---
        # Em ambos os modos price_head espera receber (combined_price_features concat sigma) -> 256 + 1
        if self.problem_type == 'INVERSE':
            # inv: produz sigma (da rota que vê premium) e price (rota que NÃO vê premium)
            self.vol_head = nn.Linear(256, 1)           # sigma a partir de combined_inv_features
            self.price_head = nn.Linear(256 + 1, 1)     # recebe combined_price_features + sigma
        else:  # FORWARD
            # forward: volatilidade é input (col 5) e price_head usa combined_price_features + sigma_input
            self.price_head = nn.Linear(256 + 1, 1)

    def _build_sequential(self, layer_dims):
        """Construtor auxiliar para criar blocos de camadas sequenciais."""
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            layers.append(AdaptiveActivation(self.config['activation_fn']))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass do PINN.
        - x: tensor normalizado com 5 colunas (S_norm,K_norm,T_norm,r,premium) quando INVERSE.
        - Retorna dicionário com 'price' e 'sigma'.
        """
        # --- Slicing/Desnormalização básica ---
        # x layout esperado: [S_norm, K_norm, T_norm, r, premium]   # 5: S,K,T,r,premium
        S_norm = x[:, 0:1]
        K_norm = x[:, 1:2]
        T_norm = x[:, 2:3]
        r      = x[:, 3:4]
        # premium_col = x[:, 4:5]   # usado apenas na rota inverse (via fourier_full)

        # Desnormaliza para cálculos físicos (para uso no payoff/intrinsic)
        S = S_norm * (self.stats['S_max'] - self.stats['S_min']) + self.stats['S_min']
        K = K_norm * (self.stats['K_max'] - self.stats['K_min']) + self.stats['K_min']
        T = T_norm * self.stats['T_max']

        # --- Rota INVERSE: usa o vetor completo (incl. premium) para estimar sigma ---
        x_full = x  # S_norm, K_norm, T_norm, r, premium
        x_fourier_full = torch.cat([torch.sin(self.fourier_full(x_full)), torch.cos(self.fourier_full(x_full))], dim=1)
        shallow_out_inv = self.shallow_inv(x_fourier_full)
        deep_out_inv    = self.deep_inv(x_fourier_full)
        combined_inv = torch.cat([shallow_out_inv, deep_out_inv], dim=1)
        combined_inv_features = self._adaptive_combiner(self.combiner_inv(combined_inv))

        # Estima sigma apenas pela rota inverse (se aplicável)
        sigma = None
        if self.problem_type == 'INVERSE':
            # softplus garante positividade de sigma
            sigma = nn.functional.softplus(self.vol_head(combined_inv_features))

        # --- Rota PRICE: NÃO usa premium (apenas S,K,T,r) ---
        x_price = x[:, :4]  # S_norm, K_norm, T_norm, r  (sem premium)
        x_fourier_price = torch.cat([torch.sin(self.fourier_price(x_price)), torch.cos(self.fourier_price(x_price))], dim=1)
        shallow_out_price = self.shallow_price(x_fourier_price)
        deep_out_price    = self.deep_price(x_fourier_price)
        combined_price = torch.cat([shallow_out_price, deep_out_price], dim=1)
        combined_price_features = self._adaptive_combiner(self.combiner_price(combined_price))

        # --- Determinação de sigma em modo FORWARD (entra pela 5ª coluna em x) ---
        if self.problem_type != 'INVERSE':
            # FORWARD: sigma é fornecida como entrada (coluna 5)
            sigma = x[:, 4:5]

        # Assegure que sigma existe (deverá existir em ambos os modos)
        if sigma is None:
            raise RuntimeError("sigma não definido. Verifique problem_type e as entradas.")

        # --- Preço a partir de combined_price_features + sigma (sem ver o premium original) ---
        # concatenamos sigma (N x 1) com o embedding price (N x 256) -> (N x 257)
        price_input = torch.cat([combined_price_features, sigma], dim=1)
        price_raw = self.price_head(price_input)  # output shape (N,1)

        # --- Saída: arquitetura PAYOFF_INSPIRED ou ORIGINAL ---
        intrinsic_value = torch.relu(S - K)  # payoff no vencimento
        if self.architecture == 'PAYOFF_INSPIRED':
            gating_factor = torch.sigmoid(price_raw)  # 0..1
            price = (1 - gating_factor) * intrinsic_value + gating_factor * S
        elif self.architecture == 'ORIGINAL':
            time_decay = 1 - torch.exp(-r * T)
            A = 2.0
            price = intrinsic_value + time_decay * (S * A * torch.sigmoid(price_raw))
        else:
            raise ValueError("Arquitetura desconhecida na configuração.")

        # Saídas padronizadas
        outputs = {'sigma': sigma, 'price': price}
        return outputs