"""
Modelagem para o Titanic: definição de pipelines, avaliação com CV,
busca de hiperparâmetros, e utilitários para salvar/interpretar modelos.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from joblib import dump

RANDOM_STATE = 42

def build_estimators() -> Dict[str, object]:
    """
    Retorna um dicionário com estimadores base para comparação.
    """
    estimators = {
        "logreg": LogisticRegression(max_iter=1000, class_weight=None, random_state=RANDOM_STATE),
        "rf": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    return estimators

def build_pipelines(preprocessor) -> Dict[str, Pipeline]:
    """
    Anexa o preprocessor a cada estimador em um Pipeline completo.
    """
    pipes = {}
    for name, est in build_estimators().items():
        pipes[name] = Pipeline(steps=[("pre", preprocessor), ("clf", est)])
    return pipes

def get_scoring() -> Dict[str, object]:
    return {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
        "roc_auc": make_scorer(roc_auc_score)
    }

def evaluate_pipelines(pipes: Dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """
    Roda CV estratificada e retorna tabela com métricas médias e std.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for name, pipe in pipes.items():
        cvres = cross_validate(pipe, X, y, cv=cv, scoring=get_scoring(), n_jobs=-1, return_estimator=False)
        record = {"model": name}
        for k, vals in cvres.items():
            if k.startswith("test_"):
                metric = k.replace("test_", "")
                record[f"{metric}_mean"] = np.mean(vals)
                record[f"{metric}_std"] = np.std(vals)
        records.append(record)
    return pd.DataFrame(records).sort_values("f1_mean", ascending=False)

def small_param_grids() -> Dict[str, dict]:
    """
    Espaços de busca pequenos e bem documentados para GridSearchCV.
    """
    grids = {
        "logreg": {
            "clf__C": [0.1, 0.5, 1.0, 2.0],
            "clf__penalty": ["l2"],
            "clf__class_weight": [None, "balanced"]
        },
        "rf": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__class_weight": [None, "balanced"]
        },
        "gb": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3]
        }
    }
    return grids

def run_grid_search(pipe: Pipeline, grid: dict, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """
    Executa GridSearchCV com ROC AUC como métrica principal (binária).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe, param_grid=grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True, return_train_score=False
    )
    gs.fit(X, y)
    return gs

def save_best_model(model, path: str):
    dump(model, path)

def get_feature_names_from_preprocessor(preprocessor, cat_prefix="cat__", num_prefix="num__"):
    """
    Extrai nomes das features após o ColumnTransformer + OneHotEncoder.
    Essa função tenta obter os nomes de saída para interpretabilidade.
    """
    feature_names = []
    # Numéricas
    num = preprocessor.transformers_[0]
    num_cols = num[2]
    feature_names.extend(num_cols)
    # Categóricas (One-Hot)
    cat = preprocessor.transformers_[1]
    ohe = cat[1].named_steps["onehot"]
    cat_cols = cat[2]
    ohe_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names.extend(ohe_names)
    return feature_names