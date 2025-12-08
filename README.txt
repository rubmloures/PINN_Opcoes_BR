-----

# DeepHeston Hybrid: Pipeline de Precificação de Opções (LSTM + PINN)

Este projeto implementa uma arquitetura híbrida de ponta para precificação de derivativos e calibração de volatilidade no mercado brasileiro (B3). O sistema funde **Deep Learning Sequencial (LSTM)** para inferência de regime de mercado com **Physics-Informed Neural Networks (PINNs)** para garantir robustez matemática e obediência à equação estocástica de Heston.

## Arquitetura do Sistema

O modelo não é uma "caixa preta" simples. Ele é composto por três módulos especializados que mimetizam o fluxo de trabalho de uma mesa de derivativos quantitativa.

```mermaid
graph LR
    A[Dados de Mercado] --> B(Módulo Analista: LSTM + Embeddings);
    B -->|Parâmetros Latentes| C(Módulo Físico: PINN Heston);
    D[Dados do Contrato] -->|Features Fourier| C;
    C -->|Valor Temporal| E[Hard Constraints];
    E -->|Preço Final| F[Loss: MSE + Resíduo PDE];
```

### 1\. Módulo Analista (O "Cérebro")

Responsável por observar o passado e inferir o estado latente atual do ativo, que não é diretamente observável.

  * **Entrada:** Sequência temporal ($t-29 \to t$) de 6 dimensões (Retornos, Volatilidade Histórica, Parkinson, EWMA, Volume Financeiro, IBOV) + **Asset Embedding** (Identidade do ativo).
  * **Mecanismo:**
      * **Asset Embeddings:** Um vetor aprendido que captura a "personalidade" do ativo (ex: se PETR4 tende a ter *mean reversion* rápida ou se VALE3 tem skew forte). Isso permite o **Meta-Learning** (treino conjunto).
      * **LSTM:** Processa a microestrutura do mercado (picos de volume, amplitude intraday) para projetar os parâmetros estocásticos futuros.
  * **Saída:** Parâmetros de Heston calibrados para o instante $t$:
      * $\nu_t$ (Volatilidade Instantânea), $\theta$ (Média Longo Prazo), $\kappa$ (Reversão), $\xi$ (Vol da Vol), $\rho$ (Correlação).

### 2\. Módulo de Lente (Fourier Features)

Responsável por resolver o "Viés Espectral" das redes neurais (dificuldade em aprender altas frequências).

  * **Lógica:** Mapeia as coordenadas físicas de baixa dimensão ($S, K, \tau, r, q$) para um espaço de alta dimensão usando funções senoidais.
  * **Objetivo:** Permite que a rede capture a **singularidade do Payoff** no vencimento (a "quina" onde o valor extrínseco colapsa a zero), que redes MLP normais suavizam excessivamente.

### 3\. Módulo Físico (A "Lei")

Uma rede densa (MLP) que funciona como um "Solver Universal" da equação de Heston.

  * **Entrada:** Estado físico ($S, K, \tau, r, q$) + Parâmetros Heston (vindos da LSTM).
  * **Hard Constraints (Garantia de Arbitragem):** A rede não prevê o preço livremente. Ela prevê apenas o **Valor Temporal**.
    $$V(S,t) = \text{Payoff}(S,K) + (1 - e^{-\lambda \tau}) \times \text{Rede}(S,t)$$
      * *Isso garante matematicamente que $V \to \text{Payoff}$ quando $\tau \to 0$.*
  * **Regularização Física (PDE Loss):** Durante o treino, calculamos as derivadas automáticas ($\frac{\partial V}{\partial t}, \frac{\partial^2 V}{\partial S^2}$) e penalizamos a rede se ela violar a **Equação Diferencial Parcial de Heston** (considerando dividendos $q$ e juros $r$).

-----

## Estratégia de Treinamento (Pipeline)

O treinamento ocorre em dois estágios para resolver o dilema "Generalização vs. Especialização".

### Estágio 1: Meta-Learning (Robustez)

  * **Dados:** Todos os ativos unificados.
  * **Objetivo:** Aprender a "Física das Opções". A rede aprende como o Theta corrói o prêmio, como o Vega infla o preço, etc.
  * **Curriculum Learning:** O peso da perda física ($\lambda_{PDE}$) começa em 0 e aumenta gradualmente. Primeiro o modelo aprende a ajustar os dados de mercado ("o que"), depois aprende a respeitar a equação ("o porquê").

### Estágio 2: Fine-Tuning (Precisão Cirúrgica)

  * **Processo:** Congela-se a PINN (a física não muda) e treina-se apenas a LSTM e os Embeddings para um ativo específico (ex: PETR4).
  * **Objetivo:** Ajustar a calibração de volatilidade para as idiossincrasias daquele papel (ex: dividendos implícitos não declarados, skew de liquidez específico).

-----

## Engenharia de Dados

O `DataLoader` processa dados brutos complexos para alimentar este sistema:

| Feature | Fonte/Cálculo | Justificativa para a Rede |
| :--- | :--- | :--- |
| **Volatilidade Parkinson** | High-Low Intraday | Captura o "nervosismo" intraday que o fechamento esconde. |
| **Volatilidade EWMA** | Média Exp. Ponderada | Detecta mudanças de tendência de vol mais rápido que a média simples. |
| **Log Volume Financeiro** | $\ln(Vol \times Preço)$ | Diferencia movimentos com convicção (alto volume) de ruído. |
| **Retorno IBOV** | Beta de Mercado | Permite à rede separar risco sistêmico de risco idiossincrático. |
| **Dividend Yield ($q$)** | Histórico de Proventos | **Crítico:** Corrige o viés de preço em ações pagadoras (PETR4, VALE3). |

-----

## Guia de Execução

### Pré-requisitos

  * Python 3.8+ com suporte a CUDA (recomendado).
  * Bibliotecas: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.

### 1\. Preparação

Gere o arquivo de dividendos (necessário para corrigir o viés de precificação):

```bash
python src/get_dividends.py
```

### 2\. Treinamento Completo (Pipeline Automático)

Executa o carregamento, treino robusto, fine-tuning automático para todos os ativos e gera relatórios.

```bash
python main.py
```

*Outputs:* Salva modelos em `resultados/modelo_final/` e gráficos em `resultados/plots/`.

### 3\. Validação e Backtest

Para abrir a "caixa preta" e ver a volatilidade latente ou testar em dados de 2024 (fora da amostra):

  * Abra o notebook `avaliacao_treino_lstm.ipynb`.
  * Execute a função `testar_ativo_2024('PETR4')`.

### 4\. Ajuste Manual (Opcional)

Para refinar um modelo específico sem rodar tudo de novo:

```bash
python src/fine_tune.py --asset VALE3 --epochs 100 --lr 1e-5
```

-----

## Estrutura do Projeto

```
PINN_Opcoes_BR/
├── src/                        # Núcleo do Sistema
│   ├── model.py                # DeepHestonHybrid (LSTM + Fourier + PINN)
│   ├── physics.py              # Equação de Heston e Derivadas
│   ├── trainer.py              # Loop de Treino com Curriculum Learning
│   ├── fine_tuner.py           # Motor de Especialização (Transfer Learning)
│   ├── data_loader.py          # Engenharia de Features e Normalização
│   └── visualization.py        # Diagnósticos (Superfícies 3D, Smiles)
├── dados/
│   ├── brutos/                 # CSVs originais (Opções, Selic, IBOV)
│   └── processados/            # Cache (opcional)
├── resultados/                 # Artefatos
│   ├── modelo_final/           # Pesos (.pth) e Stats (.json)
│   └── plots/                  # Relatórios visuais
└── notebooks/                  # Análise Exploratória e Backtest
```