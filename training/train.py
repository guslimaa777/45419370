"""Treina os modelos do projeto Nexus-Chem e salva o artefato usado pela API.

Reproduz as decisões do notebook (projeto_integrador_final_nexus_chem_aluno.ipynb),
mas dentro de Pipelines do scikit-learn: assim imputação, capping, encoding e
escala são aprendidos só no treino e reaplicados igualzinho na API.

Uso:
    python -m training.train --csv data/dataset_projeto_integrador_nexus_chem.csv
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from app.features import (ALVO, CATEGORICAS, COLUNAS_CAPPING, FEATURES_CLUSTER,
                          NUMERICAS, NUMERICAS_BRUTAS, IQRClipper, adicionar_features)

RAIZ = Path(__file__).resolve().parents[1]
SAIDA_PADRAO = RAIZ / "artifacts" / "modelo_nexus_chem.joblib"


def montar_classificador() -> Pipeline:
    preprocessador = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERICAS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAS),
    ])
    return Pipeline([
        ("features", FunctionTransformer(adicionar_features, validate=False)),
        ("capping", IQRClipper(colunas=COLUNAS_CAPPING)),
        ("prep", preprocessador),
        ("modelo", RandomForestClassifier(random_state=42)),
    ])


def montar_kmeans(k: int) -> Pipeline:
    return Pipeline([
        ("selecao", ColumnTransformer([("sel", "passthrough", FEATURES_CLUSTER)])),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10)),
    ])


def metricas(y_true, y_pred, y_proba) -> dict:
    return {
        "acuracia": round(accuracy_score(y_true, y_pred), 4),
        "precisao_falha": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_falha": round(recall_score(y_true, y_pred), 4),
        "f1_falha": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "matriz_confusao": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Caminho do CSV da Nexus-Chem")
    parser.add_argument("--saida", default=str(SAIDA_PADRAO))
    parser.add_argument("--k", type=int, default=3, help="Número de clusters do K-Means")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    linhas_originais = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Base: {linhas_originais} linhas -> {len(df)} após remover duplicatas")

    X = df[NUMERICAS_BRUTAS + CATEGORICAS]  # as features engenheiradas nascem dentro do pipeline
    y = df[ALVO].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # ---------- Classificação supervisionada (Random Forest) ----------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        montar_classificador(),
        param_grid={
            "modelo__n_estimators": [50, 100, 150],
            "modelo__max_depth": [5, 8, 12, None],
            "modelo__min_samples_split": [2, 5, 10],
            "modelo__class_weight": ["balanced", None],
        },
        cv=cv, scoring="f1", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print("Melhores hiperparâmetros:", grid.best_params_)

    proba_test = clf.predict_proba(X_test)[:, 1]
    proba_train = clf.predict_proba(X_train)[:, 1]
    met_teste = metricas(y_test, (proba_test >= 0.5).astype(int), proba_test)
    met_treino = metricas(y_train, (proba_train >= 0.5).astype(int), proba_train)
    print("Teste:", met_teste)

    # Re-treina com a base inteira para produção, mantendo os melhores hiperparâmetros
    clf_final = montar_classificador().set_params(**grid.best_params_).fit(X, y)

    # Importância das variáveis (top 10)
    nomes = clf_final.named_steps["prep"].get_feature_names_out()
    importancias = clf_final.named_steps["modelo"].feature_importances_
    top = sorted(zip(nomes, importancias), key=lambda t: -t[1])[:10]

    # ---------- Segmentação não supervisionada (K-Means) ----------
    km = montar_kmeans(args.k).fit(X)
    clusters = km.predict(X)
    Z = km[:-1].transform(X)
    silhueta = round(float(silhouette_score(Z, clusters)), 4)

    # Nomeia os clusters pela taxa de falha real: menor = Estável, maior = Alto risco
    taxa = pd.Series(y.values).groupby(clusters).mean().sort_values()
    nomes_regime = (["Estável", "Atenção", "Alto risco"] if args.k == 3
                    else [f"Regime {i + 1}" for i in range(args.k)])
    mapa_clusters = {
        int(c): {"regime": nomes_regime[i], "taxa_falha_historica": round(float(t), 4)}
        for i, (c, t) in enumerate(taxa.items())
    }
    print("Clusters:", mapa_clusters, "| silhueta:", silhueta)

    metadata = {
        "projeto": "Nexus-Chem — previsão de falha de vazão",
        "treinado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versao_sklearn": sklearn.__version__,
        "base_treino": Path(args.csv).name,
        "linhas_treino": int(len(df)),
        "taxa_falha_base": round(float(y.mean()), 4),
        "limiar_padrao": 0.5,
        "melhores_hiperparametros": {k.replace("modelo__", ""): v for k, v in grid.best_params_.items()},
        "f1_validacao_cruzada": round(float(grid.best_score_), 4),
        "metricas_teste": met_teste,
        "metricas_treino": met_treino,
        "top_variaveis": [{"variavel": n, "importancia": round(float(v), 4)} for n, v in top],
        "kmeans": {"k": args.k, "silhueta": silhueta, "clusters": mapa_clusters},
    }

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classificador": clf_final, "kmeans": km, "metadata": metadata}, saida)
    saida.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Modelo salvo em {saida}")


if __name__ == "__main__":
    main()
