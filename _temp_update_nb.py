"""
Script para integrar análises avançadas no notebook implementar.ipynb
"""
import json

# Lê o notebook atual  
with open('implementar.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Células a serem adicionadas na seção 7 (antes da Conclusão)
new_cells = [
    # Seção 7: Análises Avançadas
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Análises Visuais Avançadas\n",
            "\n",
            "Análises complementares para avaliação detalhada:\n",
            "1. Séries Temporais\n",
            "2. An��lise Detalhada de Resíduos\n",
            "3. Cross-Sections (Moneyness × Maturidade)\n",
            "4. Heatmaps 2D\n",
            "5. Rolling Window Performance"
        ]
    }
]

# Atualizar título da seção Conclusão para 8
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and '## 8. Conclusão' in cell['source'][0]:
        cell['source'][0] = '## 9. Conclusão\n'
        break

# Inserir antes da conclusão  
conclusion_index = next(i for i, cell in enumerate(nb['cells']) if '## 8. Conclusão' in str(cell.get('source', '')))
nb['cells'] = nb['cells'][:conclusion_index] + new_cells + nb['cells'][conclusion_index:]

# Salva
with open('implementar.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✓ Notebook atualizado com sucesso!")
