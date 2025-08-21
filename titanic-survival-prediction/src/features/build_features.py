"""
Feature engineering para o dataset Titanic.
Aqui extraímos Title a partir de Name, calculamos FamilySize/IsAlone,
derivamos CabinInitial e TicketPrefix, e aplicamos winsorização leve em Age e Fare.
"""
from __future__ import annotations
import re
from typing import Tuple
import pandas as pd
import numpy as np

RANDOM_STATE = 42

_TITLE_MAP = {
    "Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs",
    "Lady":"Rare", "Countess":"Rare", "Sir":"Rare",
    "Jonkheer":"Rare", "Don":"Rare", "Dona":"Rare",
    "Dr":"Rare", "Rev":"Rare", "Col":"Rare", "Major":"Rare", "Capt":"Rare"
}

def _extract_title(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    m = re.search(r",\s*([^\.]+)\.", name)
    if not m:
        return "Unknown"
    title = m.group(1).strip()
    return _TITLE_MAP.get(title, title)

def _ticket_prefix(ticket: str) -> str:
    if not isinstance(ticket, str):
        return "None"
    prefix = re.sub(r"[\.\/]", "", ticket).strip()
    m = re.match(r"([A-Za-z]+)", prefix)
    return m.group(1) if m else "None"

def _cabin_initial(cabin: str) -> str:
    if not isinstance(cabin, str) or len(cabin) == 0:
        return "NA"
    return cabin[0]

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria colunas derivadas no DataFrame (cópia).
    Retorna um novo DataFrame com as novas features.
    """
    d = df.copy()

    # Title (agrupar raros)
    d["Title"] = d["Name"].apply(_extract_title)
    rare_mask = ~d["Title"].isin(["Mr","Mrs","Miss","Master"])
    d.loc[rare_mask, "Title"] = "Rare"

    # FamilySize e IsAlone
    d["FamilySize"] = d["SibSp"].fillna(0) + d["Parch"].fillna(0) + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)

    # CabinInitial
    d["CabinInitial"] = d["Cabin"].apply(_cabin_initial)

    # TicketPrefix
    d["TicketPrefix"] = d["Ticket"].apply(_ticket_prefix)

    # Winsorização leve para Age e Fare (1% e 99%)
    for col in ["Age", "Fare"]:
        if col in d.columns:
            low, high = d[col].quantile([0.01, 0.99])
            d[col] = d[col].clip(lower=low, upper=high)

    return d

def get_feature_lists(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Retorna listas de colunas numéricas e categóricas após a engenharia de features.
    """
    numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    categorical_features = ["Sex", "Pclass", "Embarked", "CabinInitial", "TicketPrefix", "Title", "IsAlone"]
    # Filtra somente as que existem no df (robustez)
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    return numeric_features, categorical_features