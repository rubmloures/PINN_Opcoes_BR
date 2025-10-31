# /src/physics.py

import torch
from torch import Tensor
from src.model import PINN_BlackScholes # Importamos o modelo para type hinting

def black_scholes_residual(
    model_output: dict,
    x_phy: Tensor,
    data_stats: dict
) -> Tensor:
    """
    Calcula o resíduo da EDP de Black-Scholes a partir das entradas normalizadas.
    
    Esta versão corrigida garante que o grafo computacional seja preservado ao
    realizar a desnormalização e a diferenciação dentro da mesma função.

    Args:
        model_output (dict): Dicionário de saída do modelo, contendo 'price' e 'sigma'.
        x_phy (Tensor): Pontos de colocação normalizados, com requires_grad=True.
        data_stats (dict): Dicionário com estatísticas para desnormalização.

    Returns:
        Tensor: O resíduo da EDP de Black-Scholes para cada ponto.
    """
    V = model_output['price']
    sigma = model_output['sigma']
    
    # --- Desnormalização ---
    # As entradas para a EDP (S, T) devem ser desnormalizadas.
    # Como 'x_phy' tem requires_grad=True, S e T também terão e estarão no grafo.
    S_norm, _, T_norm, r, _ = torch.split(x_phy, 1, dim=1)
    
    S = S_norm * (data_stats['S_max'] - data_stats['S_min']) + data_stats['S_min']
    T = T_norm * data_stats['T_max']

    # É crucial habilitar a criação do grafo para as derivadas de ordem superior
    grad_outputs = torch.ones_like(V)

    # --- Cálculo das Derivadas ---
    # Derivamos V em relação ao tensor de entrada original 'x_phy'
    grads = torch.autograd.grad(
        outputs=V,
        inputs=x_phy,
        grad_outputs=grad_outputs,
        create_graph=True
    )[0]

    # Extraímos as derivadas em relação às variáveis normalizadas
    V_S_norm = grads[:, 0:1]
    V_T_norm = grads[:, 2:3]

    # --- Aplicação da Regra da Cadeia para obter as derivadas reais ---
    # dV/dS = dV/dS_norm * dS_norm/dS = dV/dS_norm * (1 / (S_max - S_min))
    # dV/dT = dV/dT_norm * dT_norm/dT = dV/dT_norm * (1 / T_max)
    scale_S = data_stats['S_max'] - data_stats['S_min']
    scale_T = data_stats['T_max']
    
    V_S = V_S_norm / scale_S
    V_T = V_T_norm / scale_T

    # Derivada de segunda ordem (Gamma)
    # Derivamos V_S_norm em relação a x_phy e aplicamos a regra da cadeia novamente
    V_S_norm_grads = torch.autograd.grad(
        outputs=V_S_norm,
        inputs=x_phy,
        grad_outputs=torch.ones_like(V_S_norm),
        create_graph=True
    )[0]
    
    V_SS_norm = V_S_norm_grads[:, 0:1]
    V_SS = V_SS_norm / (scale_S**2)

    # --- Cálculo do Resíduo da EDP ---
    # ∂V/∂t = -∂V/∂T
    residual = (
        -V_T +
        0.5 * (sigma ** 2) * (S ** 2) * V_SS +
        r * S * V_S -
        r * V
    )
    
    # Escalonar o resíduo pode ajudar na estabilidade
    return residual / data_stats.get('S_max', 1.0)

def payoff_boundary_condition(S: Tensor, K: Tensor) -> Tensor:
    """
    Define a condição de contorno no vencimento para uma opção de compra (call).
    Payoff = max(S - K, 0)
    """
    return torch.relu(S - K)