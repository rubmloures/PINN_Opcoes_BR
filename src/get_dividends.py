import yfinance as yf
import pandas as pd

ativos = [
    "PETR4.SA",      # Petrobras
    "VALE3.SA",      # Vale
    "BOVA11.SA",     # Indice Bovespa
    "ITUB4.SA",      # Itaú Unibanco
    "BBDC4.SA",      # Bradesco
    "BBAS3.SA",      # Banco do Brasil
    "ABEV3.SA",      # Ambev
    "B3SA3.SA",      # B3
    "WEGE3.SA",      # Weg
    "GGBR4.SA",      # Gerdau
    "CSNA3.SA",      # CSN
    "SUZB3.SA",      # Suzano
    "LREN3.SA",      # Lojas Renner
    "MGLU3.SA",      # Magazine Luiza
]
start_date = "2018-01-01"
end_date = "2025-12-31"

lista_dy = []

for ticker in ativos:
    print(f"Baixando {ticker}...")
    stock = yf.Ticker(ticker)
    
    # Histórico de Preços e Dividendos
    hist = stock.history(start=start_date, end=end_date)
    divs = stock.dividends
    
    # Merge
    df = hist[['Close']].copy()
    df['Dividends'] = divs
    df['Dividends'] = df['Dividends'].fillna(0)
    
    # Cálculo do Yield Anualizado (Janela Móvel de 12 meses)
    # Soma dividendos dos últimos 252 dias úteis
    df['Rolling_Div_1Y'] = df['Dividends'].rolling(window=252).sum()
    df['Dividend_Yield'] = df['Rolling_Div_1Y'] / df['Close']
    
    # Limpeza
    df['ativo'] = ticker.replace('.SA', '')
    df = df.reset_index()[['Date', 'Dividend_Yield', 'ativo']]
    df.rename(columns={'Date': 'data_only'}, inplace=True)
    df['data_only'] = df['data_only'].dt.normalize()
    
    lista_dy.append(df)

df_final = pd.concat(lista_dy)
import os

# ... (rest of the code remains the same until the save part)

# Define o caminho correto independente de onde o script é executado
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_path = os.path.join(project_root, 'dados', 'brutos', 'dividend_yields.csv')

# Garante que o diretório existe
os.makedirs(os.path.dirname(output_path), exist_ok=True)

df_final.to_csv(output_path, index=False)
print("Dados de Dividendos Salvos!")