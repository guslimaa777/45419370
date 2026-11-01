"""Transformações customizadas usadas no pipeline do modelo.

Ficam num módulo próprio porque o joblib precisa importá-las
para carregar o modelo salvo (tanto no treino quanto na API).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# `regime_operacional_referencia` ficou de fora: pode ser conhecida só depois do evento
# (vazamento) e, nos testes, tirá-la não piorou o modelo.
CATEGORICAS = [
    "linha_producao", "turno", "produto", "reagente_principal",
    "tipo_valvula", "status_sensor", "modo_operacao",
]

NUMERICAS_BRUTAS = [
    "vazao_setpoint_l_min", "vazao_real_l_min", "pressao_bar", "temperatura_c",
    "abertura_valvula_pct", "vibracao_bomba_mm_s", "consumo_energia_kwh",
    "tempo_resposta_s", "ph_produto", "viscosidade_cp", "densidade_g_cm3",
    "horas_operacao_bomba", "leituras_invalidas", "umidade_ambiente_pct",
    "erro_vazao_pct", "indice_estabilidade",
]

ENGENHEIRADAS = ["razao_vazao", "desvio_vazao_abs"]
NUMERICAS = NUMERICAS_BRUTAS + ENGENHEIRADAS

# Colunas com capping por IQR (mesmas do notebook)
COLUNAS_CAPPING = ["temperatura_c", "vibracao_bomba_mm_s", "pressao_bar", "consumo_energia_kwh"]

# Variáveis usadas no K-Means (mesmas do notebook)
FEATURES_CLUSTER = [
    "vazao_real_l_min", "pressao_bar", "temperatura_c",
    "vibracao_bomba_mm_s", "erro_vazao_pct", "indice_estabilidade",
]

ALVO = "falha_vazao"


def adicionar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engenharia de atributos do notebook: razão e desvio absoluto da vazão."""
    df = df.copy()
    df["razao_vazao"] = df["vazao_real_l_min"] / (df["vazao_setpoint_l_min"] + 1e-5)
    df["desvio_vazao_abs"] = (df["vazao_real_l_min"] - df["vazao_setpoint_l_min"]).abs()
    return df


class IQRClipper(BaseEstimator, TransformerMixin):
    """Capping por 1.5*IQR. Os limites são aprendidos só no treino (evita vazamento)."""

    def __init__(self, colunas=None, fator=1.5):
        self.colunas = colunas
        self.fator = fator

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = self.colunas or list(X.columns)
        q1, q3 = X[cols].quantile(0.25), X[cols].quantile(0.75)
        iqr = q3 - q1
        self.lim_inf_ = (q1 - self.fator * iqr).to_dict()
        self.lim_sup_ = (q3 + self.fator * iqr).to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, inf in self.lim_inf_.items():
            X[col] = np.clip(X[col], inf, self.lim_sup_[col])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [])
