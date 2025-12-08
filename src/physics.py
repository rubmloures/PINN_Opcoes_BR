# /src/physics.py

import torch
from torch import Tensor

class PhysicsUtils:
    """
    Utilitário para normalização/desnormalização consistente em todo o código físico.
    """
    @staticmethod
    def denormalize_zscore(x_norm: Tensor, mean: float, std: float) -> Tensor:
        """Desnormaliza variável usando Z-score."""
        return x_norm * std + mean
    
    @staticmethod
    def denormalize_inputs(x_phy: Tensor, data_stats: dict) -> tuple:
        """
        Desnormaliza todas as variáveis de input físico.
        
        Args:
            x_phy: [S_norm, K_norm, T_norm, r_norm, q_norm]
            data_stats: Dict com mean/std de todas as variáveis
            
        Returns:
            (S, K, T, r, q) em escala real
        """
        S = PhysicsUtils.denormalize_zscore(x_phy[:, 0:1], data_stats['S_mean'], data_stats['S_std'])
        K = PhysicsUtils.denormalize_zscore(x_phy[:, 1:2], data_stats['K_mean'], data_stats['K_std'])
        T = PhysicsUtils.denormalize_zscore(x_phy[:, 2:3], data_stats['T_mean'], data_stats['T_std'])
        r = PhysicsUtils.denormalize_zscore(x_phy[:, 3:4], data_stats['r_mean'], data_stats['r_std'])
        q = PhysicsUtils.denormalize_zscore(x_phy[:, 4:5], data_stats['q_mean'], data_stats['q_std'])
        
        return S, K, T, r, q

def _physical_scaling_factors(V: Tensor, S: Tensor, tau: Tensor) -> tuple:
    """
    Calcula fatores de escala baseados em análise dimensional.
    Baseado em Buckingham π theorem e Chen et al. (2020) - NeuroDiffEq
    
    Args:
        V: Option price
        S: Spot price
        tau: Time to maturity
        
    Returns:
        Tuple of scaling factors (price_scale, time_scale, spot_scale)
    """
    # Escala característica do problema (evita divisão por zero)
    price_scale = torch.mean(torch.abs(V)).clamp(min=1e-8)
    time_scale = torch.mean(tau).clamp(min=1e-8)
    spot_scale = torch.mean(S).clamp(min=1e-8)
    
    return price_scale, time_scale, spot_scale

def physics_regularization(model_output: dict, x_phy: Tensor) -> Tensor:
    """
    Regularização para evitar colapso dos parâmetros físicos.
    Baseado em Wang et al. (2020) - "Understanding and Mitigating Gradient Flow Pathologies"
    
    Penaliza parâmetros fora de ranges realistas para o modelo de Heston.
    
    Args:
        model_output: Dict contendo 'heston_params'
        x_phy: Tensor de entrada (não usado, mantido para compatibilidade)
        
    Returns:
        Tensor escalar com a penalidade de regularização
    """
    nu, theta, kappa, xi, rho = model_output['heston_params']
    
    # Penalizar parâmetros fora de ranges realistas
    # ν (variância instantânea) ∈ [0.01, 1.0]
    reg_nu = torch.mean(torch.relu(0.01 - nu) + torch.relu(nu - 1.0))
    
    # θ (variância de longo prazo) ∈ [0.01, 1.0]
    reg_theta = torch.mean(torch.relu(0.01 - theta) + torch.relu(theta - 1.0))
    
    # κ (velocidade de reversão) ∈ [0.1, 10.0]
    reg_kappa = torch.mean(torch.relu(0.1 - kappa) + torch.relu(kappa - 10.0))
    
    # ξ (vol of vol) ∈ [0.1, 2.0]
    reg_xi = torch.mean(torch.relu(0.1 - xi) + torch.relu(xi - 2.0))
    
    # ρ (correlação) ∈ [-0.99, 0.99]
    reg_rho = torch.mean(torch.relu(-0.99 - rho) + torch.relu(rho - 0.99))
    
    return reg_nu + reg_theta + reg_kappa + reg_xi + reg_rho

def heston_boundary_conditions(model_output: dict, x_phy: Tensor, 
                                data_stats: dict) -> Tensor:
    """
    Implementa condições de contorno físicas para o modelo de Heston.
    Baseado em Heston (1993) - "Closed-Form Solution" e Beck et al. (2019)
    
    Condições implementadas:
    1. S → 0: V → 0 (opção vale zero quando spot é zero)
    2. S → ∞: V → (S - K)/K (call vale o subjacente menos strike, normalizado)
    3. τ → 0: V → max((S-K)/K, 0) (payoff no vencimento, normalizado)
    
    Args:
        model_output: Dict contendo 'price' (normalizado como P/K)
        x_phy: Tensor [S_norm, K_norm, T_norm, r_norm, q_norm] (Z-score)
        data_stats: Estatísticas para desnormalização
        
    Returns:
        Tensor escalar com a perda de boundary conditions
    """
    V = model_output['price']  # Normalizado: V = P/K
    
    # Desnormalizar variáveis usando PhysicsUtils
    S, K, T, r, q = PhysicsUtils.denormalize_inputs(x_phy, data_stats)
    
    # === Condição 1: S → 0 ===
    S_threshold_low = 0.1 * torch.mean(S)
    mask_small_S = (S < S_threshold_low).float()
    bc_small_S = torch.mean(mask_small_S * V ** 2)
    
    # === Condição 2: S → ∞ (usando moneyness) ===
    moneyness = S / (K + 1e-8)
    mask_large_S = (moneyness > 2.0).float()
    # CORRECTED: V está normalizado (P/K), então comparar com (S-K)/K
    intrinsic_value_norm = (S - K) / (K + 1e-8)
    bc_large_S = torch.mean(mask_large_S * (V - intrinsic_value_norm) ** 2)
    
    # === Condição 3: τ → 0 ===
    tau_threshold = 0.01 * data_stats['T_max']
    mask_small_tau = (T < tau_threshold).float()
    # CORRECTED: Payoff normalizado max((S-K)/K, 0)
    payoff_norm = torch.relu((S - K) / (K + 1e-8))
    bc_small_tau = torch.mean(mask_small_tau * (V - payoff_norm) ** 2)
    
    # Combinar boundary conditions
    total_bc_loss = bc_small_S + bc_large_S + bc_small_tau
    
    return total_bc_loss

def heston_residual(
    model_output: dict,
    x_phy: Tensor,
    data_stats: dict,
    lambda_bc: float = 1.0,
    lambda_reg: float = 0.01,
    return_residuals: bool = False
) -> Tensor:
    """
    Calcula o resíduo corrigido da EDP de Heston com boundary conditions e regularização.
    
    Implementação baseada em:
    - Raissi et al. (2019) - Physics-Informed Neural Networks
    - Sirignano & Spiliopoulos (2018) - DGM Method
    
    Equação de Heston para Call Option V(t, S, nu):
    dV/dt + 0.5*S^2*nu*V_ss + rho*xi*S*nu*V_snu + 0.5*xi^2*nu*V_nunu 
    + (r-q)*S*V_s + kappa*(theta - nu)*V_nu - r*V = 0
    
    IMPORTANTE: V está normalizado como P/K, então a EDP é aplicada em V_norm.

    Args:
        model_output: Dict contendo 'price' (P/K) e 'heston_params' (tupla com nu, theta, etc).
        x_phy: Tensor de entrada da PINN [S_norm, K_norm, T_norm, r_norm, q_norm] (Z-score).
        data_stats: Estatísticas para desnormalização.
        lambda_bc: Peso para boundary conditions (default: 1.0)
        lambda_reg: Peso para regularização física (default: 0.01)
        return_residuals: Se True, retorna o tensor de resíduos ponto-a-ponto (default: False)
        
    Returns:
        Tensor escalar com o resíduo total (PDE + BC + Regularização) ou Tensor de resíduos.
    """
    V = model_output['price']  # Normalizado: P/K
    
    # Desempacota os parâmetros estocásticos previstos pela LSTM
    nu, theta, kappa, xi, rho = model_output['heston_params']

    # === 1. Desnormalização das Variáveis Físicas ===
    S, K, T, r, q = PhysicsUtils.denormalize_inputs(x_phy, data_stats)
    
    # === 2. Cálculo dos Gradientes de 1ª Ordem ===
    # Habilitar gradientes para input físico
    x_phy.requires_grad_(True)
    
    # Calcular gradientes em relação a x_phy e aos parâmetros Heston originais
    grads = torch.autograd.grad(
        outputs=V,
        inputs=[x_phy, nu, theta, kappa, xi, rho],
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        allow_unused=True,
        retain_graph=True
    )
    
    dV_dxphy = grads[0]
    dV_dnu = grads[1]
    dV_dtheta = grads[2]
    dV_dkappa = grads[3]
    dV_dxi = grads[4]
    dV_drho = grads[5]
    
    if dV_dxphy is None: dV_dxphy = torch.zeros_like(x_phy)
    if dV_dnu is None: dV_dnu = torch.zeros_like(nu)
    # Outros gradientes não são usados diretamente na EDP, mas dV_dnu é essencial

    # Extrai derivadas parciais NORMALIZADAS
    V_S_norm = dV_dxphy[:, 0:1]  # dV/dS_norm
    V_T_norm = dV_dxphy[:, 2:3]  # dV/dT_norm
    
    # === CORRECTED: Aplicação da Chain Rule ===
    # V = P/K é função de S, K, T
    # dV/dS = dV/dS_norm * dS_norm/dS = dV/dS_norm * (1/S_std)
    V_S = V_S_norm / (data_stats['S_std'] + 1e-8)
    
    # T_input é 'Time to Maturity' (tau)
    # A EDP usa tempo de calendário t. dt = -dtau
    V_tau = V_T_norm / (data_stats['T_std'] + 1e-8)
    V_t = -V_tau

    # === 3. Cálculo dos Gradientes de 2ª Ordem ===
    # A) V_SS (Convexidade / Gamma)
    grad_V_S = torch.autograd.grad(
        outputs=V_S,
        inputs=x_phy,
        grad_outputs=torch.ones_like(V_S),
        create_graph=True,
        allow_unused=True,
        retain_graph=True
    )[0]
    
    if grad_V_S is None:
        V_SS = torch.zeros_like(V_S)
    else:
        V_SS_norm = grad_V_S[:, 0:1]
        V_SS = V_SS_norm / (data_stats['S_std'] + 1e-8)

    # B) V_nu_nu (Vol of Vol curvature)
    grad_V_nu = torch.autograd.grad(
        outputs=dV_dnu,
        inputs=nu,
        grad_outputs=torch.ones_like(dV_dnu),
        create_graph=True,
        allow_unused=True,
        retain_graph=True
    )[0]
    
    if grad_V_nu is None:
        V_nu_nu = torch.zeros_like(dV_dnu)
    else:
        V_nu_nu = grad_V_nu

    # C) V_S_nu (Derivada Mista - captura correlação preço-vol)
    grad_V_S_mixed = torch.autograd.grad(
        outputs=V_S,
        inputs=nu,
        grad_outputs=torch.ones_like(V_S),
        create_graph=True,
        allow_unused=True,
        retain_graph=True
    )[0]
    
    if grad_V_S_mixed is None:
        V_S_nu = torch.zeros_like(V_S)
    else:
        V_S_nu = grad_V_S_mixed

    # === 4. Montagem do Resíduo da EDP ===
    # Termos da equação de Heston (com clamping para estabilidade numérica)
    diffusion_S = 0.5 * (S ** 2) * nu.clamp(min=1e-6) * V_SS
    diffusion_nu = 0.5 * (xi.clamp(min=1e-6) ** 2) * nu.clamp(min=1e-6) * V_nu_nu
    correlation_term = rho * xi.clamp(min=1e-6) * S * nu.clamp(min=1e-6) * V_S_nu
    drift_S = (r - q) * S * V_S
    drift_nu = kappa.clamp(min=1e-6) * (theta - nu) * dV_dnu
    
    pde_residual = (
        V_t + 
        diffusion_S + 
        diffusion_nu + 
        correlation_term + 
        drift_S + 
        drift_nu - 
        (r * V)
    )
    
    # === 5. Normalização Física do Resíduo ===
    # Baseado em Chen et al. (2020) - análise dimensional
    price_scale, time_scale, spot_scale = _physical_scaling_factors(V, S, T)
    pde_residual_normalized = pde_residual / price_scale
    
    if return_residuals:
        return pde_residual_normalized
    
    # MSE do resíduo da PDE
    loss_pde = torch.mean(pde_residual_normalized ** 2)
    
    # === 6. Boundary Conditions ===
    loss_bc = heston_boundary_conditions(model_output, x_phy, data_stats)
    
    # === 7. Regularização Física ===
    loss_reg = physics_regularization(model_output, x_phy)
    
    # === 8. Combinação Final (Multi-Task Learning - Kendall et al. 2018) ===
    total_residual = (
        loss_pde + 
        lambda_bc * loss_bc + 
        lambda_reg * loss_reg
    )
    
    # Retornar dicionário com componentes separados para logging
    return {
        'total': total_residual,
        'pde': loss_pde,
        'bc': loss_bc,
        'reg': loss_reg,
        'pde_residual_mean': torch.mean(torch.abs(pde_residual)),
        'pde_residual_max': torch.max(torch.abs(pde_residual)),
        'boundary_error_mean': torch.mean(torch.abs(heston_boundary_conditions(model_output, x_phy, data_stats))),
    } if not return_residuals else pde_residual_normalized


def payoff_boundary_condition(S: Tensor, K: Tensor) -> Tensor:
    """
    Condição de contorno no vencimento (Call Option).
    Max(S - K, 0)
    
    Args:
        S: Spot price
        K: Strike price
        
    Returns:
        Payoff tensor
    """
    return torch.relu(S - K)