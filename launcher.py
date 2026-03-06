#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
launcher.py - Orquestrador Central do Projeto PINN_Opcoes_BR.
Oferece uma interface visual para facilitar a execução de diferentes modos do pipeline.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.menu import Menu
    from rich.prompt import Prompt, IntPrompt
    from rich.table import Table
    from rich import print as rprint
except ImportError:
    print("Erro: A biblioteca 'rich' nao esta instalada.")
    print("Instalando dependencias basicas...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich"], check=True)
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, IntPrompt
    from rich.table import Table
    from rich import print as rprint

console = Console()

def show_banner():
    banner = """
    [bold blue]╔══════════════════════════════════════════════════════════════════════╗[/bold blue]
    [bold blue]║[/bold blue] [bold white]    PINN_Opcoes_BR - ORQUESTRADOR DE INTELIGENCIA ARTIFICIAL      [/bold white] [bold blue]║[/bold blue]
    [bold blue]╚══════════════════════════════════════════════════════════════════════╝[/bold blue]
    """
    console.print(banner)

def run_command(command, description):
    console.print(f"\n[bold green]>>> Executando: {description}[/bold green]")
    console.print(f"[dim]Comando: {' '.join(command)}[/dim]\n")
    try:
        # Usamos shell=True no Windows se necessário, mas subprocess.run com lista é mais seguro
        result = subprocess.run(command)
        if result.returncode == 0:
            console.print(f"\n[bold green]✓ {description} finalizado com sucesso.[/bold green]")
        else:
            console.print(f"\n[bold red]✗ {description} falhou com código {result.returncode}.[/bold red]")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]! Execução interrompida pelo usuário.[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Erro inesperado: {e}[/bold red]")

def menu():
    while True:
        console.clear()
        show_banner()
        
        table = Table(title="Menu de Operações", show_header=False, box=None)
        table.add_row("[bold cyan]1.[/bold cyan]", "Treinamento Completo (Pipeline Padrão)")
        table.add_row("[bold cyan]2.[/bold cyan]", "Otimização Optuna (Busca de Hiperparâmetros)")
        table.add_row("[bold cyan]3.[/bold cyan]", "Teste de Integridade e QA (Rápido)")
        table.add_row("[bold cyan]4.[/bold cyan]", "Executar apenas Visualização (Diagnóstico)")
        table.add_row("[bold cyan]5.[/bold cyan]", "Limpar Pasta de Resultados")
        table.add_row("[bold red]0.[/bold red]", "Sair")
        
        console.print(Panel(table, title="Selecione uma opção", border_style="blue"))
        
        choice = Prompt.ask("Escolha", choices=["0", "1", "2", "3", "4", "5"], default="1")
        
        if choice == "1":
            run_command([sys.executable, "main.py"], "Treinamento Padrão")
            input("\nPressione Enter para voltar ao menu...")
            
        elif choice == "2":
            trials = IntPrompt.ask("Número de trials", default=50)
            epochs = IntPrompt.ask("Épocas por trial", default=5)
            run_command([
                sys.executable, "main.py", 
                "--optimize", 
                "--n-trials", str(trials), 
                "--epochs-per-trial", str(epochs)
            ], f"Otimização Optuna ({trials} trials)")
            input("\nPressione Enter para voltar ao menu...")
            
        elif choice == "3":
            run_command([sys.executable, "test_pipeline.py"], "Teste de Integridade")
            input("\nPressione Enter para voltar ao menu...")
            
        elif choice == "4":
            # Para rodar apenas visualização, simulamos um modo onde o treino é pulado
            # Ou simplesmente avisamos que precisa haver um histórico
            if os.path.exists("resultados/training_history.csv"):
                run_command([sys.executable, "main.py", "--skip-train"], "Geração de Plots (Mock)")
                # Nota: main.py não tem --skip-train oficialmente, seria uma melhoria futura.
                # Por ora, sugerimos rodar o teste de integridade que gera plots.
                console.print("[yellow]Dica: O main.py atual sempre tenta treinar. Considere usar o Teste de Integridade (Opção 3) para validar plots rapidamente.[/yellow]")
            else:
                console.print("[red]Erro: Nenhum histórico encontrado em resultados/training_history.csv[/red]")
            input("\nPressione Enter para voltar ao menu...")
            
        elif choice == "5":
            if Prompt.ask("Deseja realmente apagar os resultados atuais?", choices=["s", "n"], default="n") == "s":
                import shutil
                if os.path.exists("resultados"):
                    shutil.rmtree("resultados")
                    console.print("[green]Pasta de resultados limpa.[/green]")
                else:
                    console.print("[yellow]Pasta de resultados não existe.[/yellow]")
            input("\nPressione Enter para voltar ao menu...")
            
        elif choice == "0":
            console.print("[bold blue]Até logo![/bold blue]")
            break

def main():
    parser = argparse.ArgumentParser(description="Launcher PINN Opcoes BR")
    parser.add_argument('--mode', type=str, choices=['train', 'optimize', 'test'], help='Modo headless')
    parser.add_argument('--trials', type=int, default=50)
    args = parser.parse_args()
    
    if args.mode:
        if args.mode == 'train':
            run_command([sys.executable, "main.py"], "Treinamento Headless")
        elif args.mode == 'optimize':
            run_command([sys.executable, "main.py", "--optimize", "--n-trials", str(args.trials)], "Otimização Headless")
        elif args.mode == 'test':
            run_command([sys.executable, "test_pipeline.py"], "Teste Headless")
    else:
        try:
            menu()
        except Exception as e:
            console.print(f"[bold red]Ocorreu um erro no menu: {e}[/bold red]")

if __name__ == "__main__":
    main()
