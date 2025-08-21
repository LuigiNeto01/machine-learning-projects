"""
Módulo de pré-processamento para o projeto Titanic.
Contém a função que constrói um ColumnTransformer com pipelines
para variáveis numéricas e categóricas.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RANDOM_STATE = 42

def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scaler: bool = True
) -> ColumnTransformer:
    """
    Constrói o ColumnTransformer do scikit-learn para pré-processamento.

    - Numéricas: imputação por mediana + (opcional) StandardScaler
    - Categóricas: imputação por mais frequente + OneHotEncoder(handle_unknown="ignore")

    Parameters
    ----------
    numeric_features : list of str
        Nome das colunas numéricas.
    categorical_features : list of str
        Nome das colunas categóricas.
    scaler : bool, default=True
        Se True, aplica StandardScaler nas numéricas.

    Returns
    -------
    ColumnTransformer
    """
    num_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scaler:
        num_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    from sklearn.pipeline import Pipeline
    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor