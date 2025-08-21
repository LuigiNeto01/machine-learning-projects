# Titanic: Machine Learning from Disaster — Projeto Didático (pt-BR)

Pipeline completo e reprodutível para prever `Survived` no dataset Titanic (Kaggle).

## Estrutura
```text
titanic-ml/
├── notebooks/
│   └── 01_titanic_ml.ipynb
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   └── models/
│       └── modeling.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── submission.csv
├── requirements.txt
└── README.md
```

## Como rodar

1. **Python** 3.10+ e `pip` instalados.
2. (Opcional) Crie um ambiente virtual.
3. Instale dependências:  
   ```bash
   pip install -r requirements.txt
   ```
4. Baixe os dados do Kaggle (ou manualmente):
   - Com **Kaggle API** (precisa configurar `~/.kaggle/kaggle.json`):
     ```bash
     kaggle competitions download -c titanic -p data/raw
     unzip -o data/raw/titanic.zip -d data/raw
     ```
   - Ou faça download manual de `train.csv` e `test.csv` e coloque em `data/raw/`.
5. Abra o notebook:
   ```bash
   jupyter lab notebooks/01_titanic_ml.ipynb
   ```
6. Execute todas as células **de cima para baixo**. Ao final, será gerado:
   - `outputs/submission.csv`
   - `outputs/models/model.joblib`
   - gráficos em `outputs/figures/`

## Observações
- `random_state=42` fixado para reprodutibilidade.
- Sem vazamento: transformações aplicadas via `Pipeline` + `ColumnTransformer`.
- Notebook contém justificativas e comentários didáticos.