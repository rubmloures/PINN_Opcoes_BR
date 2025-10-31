# -*- coding: utf-8 -*-
# [1. Imports e Configurações iniciais]
#---Install required packages---
#%pip install numpy pandas deepxde matplotlib scipy requests seaborn tqdm requests
#%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
##--- a) Imports das bibliotecas necessárias:

# Configuração do Ambiente
import os
os.environ["DDE_BACKEND"] = "pytorch"

import gc
import io
import time
from contextlib import redirect_stdout

# Manipulação de dados
import numpy as np
import pandas as pd

# Barra de progresso
from tqdm.notebook import tqdm

# Machine Learning e Deep Learning
import torch
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader

# DeepXDE - Framework para equações diferenciais
import deepxde as dde
from deepxde.backend import pytorch
from deepxde.geometry import Hypercube
from deepxde.data.pde import PDE
from deepxde.icbc import DirichletBC, NeumannBC, PointSetBC
from deepxde.callbacks import (
    PDEPointResampler,
    ModelCheckpoint,
    Callback,
    EarlyStopping
)
from deepxde.nn.pytorch.fnn import FNN

# Estatística e Modelagem
from scipy.stats import norm
from sklearn.model_selection import train_test_split

# Interface gráfica para seleção de arquivos
from tkinter import Tk, filedialog

# Configuração do PyTorch
print(
    "Dispositivo atual:",
    torch.cuda.current_device(),
    "-",
    torch.cuda.get_device_name(torch.cuda.current_device())
)
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Configuração do DeepXDE
dde.backend.set_default_backend("pytorch")
print("Imports for PINN model loaded successfully.")

import requests
# Credenciais do Telegram para notificações - robot_notify 
TELEGRAM_BOT_TOKEN = "8001785179:AAHfO4F9r9okKilpflnh5aIk8NWN4RL49tk"
TELEGRAM_CHAT_ID = "-1002806240423"
# Função para conectar com o bot de aviso:
def send_telegram_message(message):
    """Envia uma mensagem para um chat do Telegram."""
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Erro ao enviar mensagem para o Telegram: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Falha na conexão com a API do Telegram: {e}")

## Para rodar o PINN com os dados de um dataframe, é preciso mexer nos parâmetros 
# do tamanho do bach, além dos hiperparâmetros padrões do modelo.

#--- [2. Pré-processamento de Dados]
#-- 2.0 Configurações iniciais:

# a) Carregamento e pré-processamento dos dados e normalização:
# Função para processar e carregar os dados:
#path_adress = '../Data/DataHistory_Options/Dados_Unificados.csv'
path_adress = '../data/Dados_Unificados.csv'
def creat_dataframe (path_adress):
    df = pd.read_csv(path_adress, sep= ',') 
    df['time_to_maturity'] = df['days_to_maturity'] /252
    df['volatility'].fillna(df['volatility'].mean())
    df['vol']=df['volatility']/100.0
    df = df.dropna() 
    df['r'] = 13.75/100.0 
    epsilon = 1e-8
    df['moneyness_ratio'] = df['spot_price'] / (df['strike'] + epsilon)
    return df
data = creat_dataframe(path_adress)

# b) Normalização dos dados:
# Valores Máximos e Mínimos para a normalização:
S_max = data['spot_price'].max()
S_min = data['spot_price'].min()
T_max = data['time_to_maturity'].max()
K_max = data['strike'].max()
K_min = data['strike'].min()
moneyness_max = data['moneyness_ratio'].max()
moneyness_min = data['moneyness_ratio'].min()
# Dados Normalizados para o PINN:
data['S_norm'] = (data['spot_price']- S_min) /(S_max - S_min)
data['T_norm'] = (data['days_to_maturity']/252) / (T_max)
data['K_norm'] = (data['strike']- K_min) / (K_max - K_min)
data['moneyness_norm'] = (data['moneyness_ratio'] - moneyness_min) / (moneyness_max - moneyness_min)

# Volatilidade:
sigma = data['vol']  
# Taxa de juros livre de risco em 2023:
r = data['r']
# Strike Price:
premium = data['premium'] 

# Dados para a formulação do modelo PINN:
colunas_features = ['S_norm', 'T_norm', 'K_norm', 'moneyness_norm','vol', 'r', 'premium']
df_modelo = data[colunas_features].copy()

# c) Divisão Treino/Validação
# AMOSTRA que caiba na memória da sua GPU
num_samples = 400_000  # Comece com um valor e ajuste conforme necessário
data_sampled = data.sample(n=min(num_samples, len(data)), random_state=42)
train_data, val_data = train_test_split(df_modelo, test_size=0.2, random_state=42)

#--- [3. Definição da Geometria e Domínio] 
#-- 3.1 Domain: [S_norm, T_norm, K_norm, volatility, r] hypercube
domain = Hypercube([0.0] * 6, [1.0] * 6)

#--- [4. Formulação da PDE com Escalamento] 
def pde(x, y):
    S = x[:, 0:1] * (S_max - S_min) + S_min
    t = x[:, 1:2] * T_max
    K = x[:,2:3] * (K_max - K_min) + K_min
    # moneyness_norm = x[:, 3:4] # Não é usado diretamente na PDE, mas está aqui
    sigma = x[:, 4:5]  
    r = x[:, 5:6]
    scale_S = 1.0 / S_max
    scale_t = 1.0 / T_max
    scale_y = 1.0 / (S_max - K_min) if (S_max - K_min) > 1e-8 else 1.0
    with torch.enable_grad():
        y_scaled = y * scale_y
        grads = torch.autograd.grad(y_scaled, x, grad_outputs=torch.ones_like(y_scaled), create_graph=True, retain_graph=True)[0]
        C_t = grads[:, 1:2] * (scale_y / scale_t)
        C_S = grads[:, 0:1] * (scale_y / scale_S)
        grad_S = torch.autograd.grad(C_S, x, grad_outputs=torch.ones_like(C_S), create_graph=True, retain_graph=True)[0][:, 0:1]
        C_SS = grad_S * (scale_y / (scale_S ** 2))
    diffusion = 0.5 * (sigma ** 2) * (S ** 2) * C_SS
    convection = r * S * C_S
    decay = -r * y
    alpha = 1.0 + 99.0 * torch.exp(-10.0 * t)
    residual = alpha * C_t + diffusion + convection + decay
    return residual / scale_y
    

#--- [5. Condições de Contorno] 
# a) Payoff Final (t=T)
def payoff_func(x_bc):
    x_bc_tensor = torch.from_numpy(x_bc).float().to(device)
    S_bc = x_bc_tensor[:, 0:1] * (S_max - S_min) + S_min
    K_bc = x_bc_tensor[:, 2:3] * (K_max - K_min) + K_min
    # Payoff é max(S-K,0) para opções de compra (call)
    return torch.relu(S_bc - K_bc)

# b) Condição S=0:
def s0_func(x_bc):
    x_bc_tensor = torch.from_numpy(x_bc).float().to(device)
    return torch.zeros_like(x_bc_tensor[:, 0:1])

# c) Condição T_norm=1:
ic = dde.icbc.DirichletBC(
    domain,
    payoff_func,
    lambda x, on_boundary: on_boundary and np.isclose(x[1], 1.0), 
    component=0
)

# Condição S=0 - para quando o preço do ativo é zero:
bc_s0 = dde.icbc.DirichletBC(
    domain,
    s0_func,
    lambda x, on_boundary: on_boundary and np.isclose(x[0], 0.0),
    component=0
)

#---[6. Supervised Data BC] 
# Usando os dados de TREINAMENTO para criar o PointSetBC:
X_train = train_data.drop(columns=["premium"]).values
y_train = train_data["premium"].values.reshape(-1, 1)
# Usando os dados de VALIDAÇÃO para criar o PointSetBC:
X_val = val_data.drop(columns=['premium']).values
y_val = val_data['premium'].values.reshape(-1,1)

train_dataset = TensorDataset(
    torch.from_numpy(X_train).float()
    ,torch.from_numpy(y_train).float())

#--- [7. Arquitetura da Rede Neural - Combinação Linear de Redes Rasas e Profundas] 
class EuropeanCallPINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # a) Fourier Features (γ=10)
        self.regularizer = None
        self.fourier = torch.nn.Linear(6, 128)
        torch.nn.init.normal_(self.fourier.weight, mean=0, std=10.0)

        # b) Componente Raso - [sin+cos] das Fourier features
        self.shallow = torch.nn.Sequential(torch.nn.Linear(256, 64), torch.nn.SiLU())

        # c) Componente Profundo
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 512),
            torch.nn.SiLU()
        )

        # d) Camada Combinadora
        self.combiner = torch.nn.Linear(64 + 512, 1)


        # e) Parâmetros Financeiros 
        self.S_min = torch.tensor(S_min, device=device)
        self.S_max = torch.tensor(S_max, device=device)
        self.K_min = torch.tensor(K_min, device=device)
        self.K_max = torch.tensor(K_max, device=device)
        
    def forward(self, x):
        # f) Desnormalização
        S = x[:,0:1] * (self.S_max - self.S_min) + self.S_min
        t = x[:,1:2] * T_max
        K = x[:,2:3] * (self.K_max - self.K_min) + self.K_min
        # moneyness_norm = x[:, 3:4]
        # sigma (vol) = x[:, 4:5]
        r = x[:, 5:6]    

        # g) Mapeamento Fourier
        x_fourier = torch.cat([torch.sin(
            self.fourier(x))
            ,torch.cos(self.fourier(x))]
            ,dim=1
        ) 

        # h) Combinação Linear
        shallow_out = self.shallow(x_fourier)
        deep_out = self.deep(x_fourier)
        combined = torch.cat([shallow_out, deep_out], dim=1)
        C_raw = self.combiner(combined)  
        
        # i) Hard Constraints
        intrinsic = torch.relu(S - K)
        time_decay = 1 - torch.exp(-r * t)
        A = 2.0 
        return intrinsic + time_decay * (S * A * torch.sigmoid(C_raw))

#
#--- [8. Configuração do Modelo] 
# a) Dataset PDE
data_pde = dde.data.PDE(
    domain,
    pde=pde,
    bcs=[ic, bc_s0],
    num_domain=50_000,
    num_boundary=10_000,
)

# Limpeza de memória:
gc.collect()
torch.cuda.empty_cache()

# b) Inicialização da Rede e Validação e Monitoramento
net = EuropeanCallPINN().to(device)
model = dde.Model(data_pde, net)

#
#--- [9. Estratégia de Treinamento e Execução Robusta] ---

# Flag para controlar o status da conclusão do treinamento
training_completed_successfully = False
# Inicia o cronômetro fora do 'try' para estar disponível no 'except'
total_start_time = time.time() 

try:
    # a) Hiperparâmetros do Treinamento
    epochs = 12_000
    initial_lr = 1e-4
    batch_size = 4096
    phy_batch_size = 4096

    # b) Configurações dos Marcos Percentuais
    milestones_percent = [0.6, 0.85]
    phase1_end_pct = 0.20
    phase2_end_pct = 0.70
    patience = int(epochs * 0.20)
    min_delta = 1e-6

    # c) Cálculo dos Marcos em Épocas
    milestones_int = [int(p * epochs) for p in milestones_percent]
    phase1_end_epoch = int(epochs * phase1_end_pct)
    phase2_end_epoch = int(epochs * phase2_end_pct)
    all_milestones = sorted(list(set(milestones_int + [phase1_end_epoch, phase2_end_epoch])))

    # d) Preparação dos DataLoaders e Otimizador
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    generator_cuda = torch.Generator(device='cuda')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,generator=generator_cuda)
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr)
    scheduler = MultiStepLR(optimizer, milestones=milestones_int, gamma=0.1)
    
    # e) Inicialização das Variáveis de Controle
    epochs_no_improve = 0
    best_val_loss = float('inf')
    loss_functions = [dde.losses.mean_squared_error] * 3

    # f) Loop de Treinamento Principal
    print(f"Iniciando treinamento por {epochs} épocas com Early Stopping por Fases (Paciência={patience}).")
    print(f"Marcos de ajuste do treinamento (épocas): {all_milestones}")
    send_telegram_message(f"*Início do Treinamento do Modelo PINN*\n\nDispositivo: {torch.cuda.get_device_name(torch.cuda.current_device())}\nÉpocas configuradas: {epochs}")
    
    epoch = 0
    while epoch < epochs:
        epoch_start_time = time.time()
        net.train()

        # Ajuste dinâmico dos pesos
        if epoch < phase1_end_epoch:
            pde_weight, supervised_weight = 1.0, 100.0
        elif epoch < phase2_end_epoch:
            progress = (epoch - phase1_end_epoch) / (phase2_end_epoch - phase1_end_epoch)
            pde_weight = 1.0 + progress * 99.0
            supervised_weight = 100.0 - progress * 99.0
        else:
            pde_weight, supervised_weight = 100.0, 1.0
        
        epoch_losses = {'total': 0.0, 'pde': 0.0, 'ic': 0.0, 'bc_s0': 0.0, 'supervised': 0.0}

        # Loop de treinamento por mini-batch
        for i, (x_sup_batch, y_sup_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred_sup = net(x_sup_batch.to(device))
            loss_supervised_raw = torch.nn.functional.mse_loss(y_pred_sup, y_sup_batch.to(device))
            
            x_phy_batch = torch.from_numpy(domain.random_points(phy_batch_size)).float().to(device)
            x_phy_batch.requires_grad_()
            y_phy_pred = net(x_phy_batch)
            losses_phy = model.data.losses(np.zeros((phy_batch_size, 1)), y_phy_pred, loss_functions, x_phy_batch, model)
            loss_pde_raw, loss_ic_raw, loss_bc_s0_raw = losses_phy[0], losses_phy[1], losses_phy[2]
            
            total_loss = (loss_pde_raw * pde_weight) + loss_ic_raw + loss_bc_s0_raw + (loss_supervised_raw * supervised_weight)
            total_loss.backward()
            optimizer.step()
            
            for key, loss in zip(epoch_losses.keys(), [total_loss, loss_pde_raw, loss_ic_raw, loss_bc_s0_raw, loss_supervised_raw]):
                epoch_losses[key] += loss.item()

        # Fase de Validação (executada uma vez por época)
        net.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_loader:
                total_val_loss += torch.nn.functional.mse_loss(net(x_val_batch.to(device)), y_val_batch.to(device)).item()
        avg_val_loss = total_val_loss / len(val_loader)

        # Logging e Early Stopping
        print(f"Época: {epoch + 1:04d}/{epochs} | T: {time.time() - epoch_start_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.1e} | "
            f"Val Loss: {avg_val_loss:.4e} | Train Loss: {epoch_losses['total'] / len(train_loader):.4e} | "
            f"Pesos (PDE/Sup): {pde_weight:.1f}/{supervised_weight:.1f}")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(net.state_dict(), "best_model_weights.pt")
            print(f"    ==> Nova melhor Val Loss: {best_val_loss:.4e}. Modelo salvo!")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            next_milestone = next((m for m in all_milestones if m > epoch), -1)
            if next_milestone != -1:
                print(f"\n!! Paciência atingida. Saltando para a próxima fase no marco {next_milestone} !!\n")
                epoch = next_milestone - 1
                epochs_no_improve = 0
                best_val_loss = float('inf')
                while scheduler.last_epoch < epoch:
                    scheduler.step()
            else:
                print(f"\nParada antecipada final acionada após {patience} épocas sem melhoria.")
                training_completed_successfully = True
                break
                
        scheduler.step()
        epoch += 1

    # Se o loop terminar naturalmente (sem o break), marca como sucesso
    if epoch == epochs:
        training_completed_successfully = True

# --- [10. Captura de Exceções e Alertas] ---
except Exception as e:
    # Este bloco só executa se uma exceção ocorrer no bloco 'try'.
    duration_minutes = (time.time() - total_start_time) / 60
    error_message = f"*ERRO no Treinamento*\n\nOcorreu uma exceção:\n`{str(e)}`\n\nO processo foi interrompido após `{duration_minutes:.2f}` minutos."
    print(error_message)
    send_telegram_message(error_message)
    # Re-lança a exceção para que o script pare com um código de erro.
    raise

finally:
    # Este bloco executa SEMPRE, independentemente de ter havido erro ou não.
    
    # Envia a mensagem de sucesso e salva o modelo APENAS se o treinamento foi concluído.
    if training_completed_successfully:
        duration_minutes = (time.time() - total_start_time) / 60
        print(f"\nTreinamento concluído em {duration_minutes:.2f} minutos.")
        
        success_message = (
            f"*Treinamento Concluído com Sucesso*\n\n"
            f"Duração total: `{duration_minutes:.2f}` minutos.\n"
            f"Melhor loss de validação: `{best_val_loss:.4e}`"
        )
        print(success_message)
        send_telegram_message(success_message)
        
        # Salvando os modelos e dados finais
        print("\nSalvando os dados do modelo final...")
        # Carrega os pesos do melhor modelo encontrado
        net.load_state_dict(torch.load("best_model_weights.pt"))
        # Salva o modelo completo (arquitetura + melhores pesos)
        torch.save(net, "pinn_model_final.pt")
        # Salva apenas os melhores pesos
        torch.save(net.state_dict(), "pinn_weights_final.pt")
        print("Modelos salvos com sucesso.")
        send_telegram_message("Modelos e dados salvos com sucesso!")



