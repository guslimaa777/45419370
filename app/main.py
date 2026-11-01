"""API de previsão de falha de vazão — Nexus-Chem.

Rodar local:  uvicorn app.main:app --reload
Documentação: http://localhost:8000/docs
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader

from app import features  # noqa: F401  (necessário para o joblib carregar o modelo)
from app.features import CATEGORICAS, NUMERICAS_BRUTAS
from app.schemas import Leitura, LoteLeituras, Predicao, RespostaLote

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parents[1] / "artifacts" / "modelo_nexus_chem.joblib"))
API_KEY = os.getenv("API_KEY")  # se definida, exige o header X-API-Key

estado: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        estado.update(joblib.load(MODEL_PATH))
    yield
    estado.clear()


app = FastAPI(
    title="Nexus-Chem — API de Falha de Vazão",
    description=(
        "Prevê falhas de vazão (Random Forest) e identifica o regime operacional "
        "(K-Means) a partir das leituras de sensores da planta química Nexus-Chem."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_header_chave = APIKeyHeader(name="X-API-Key", auto_error=False)


def verificar_chave(chave: Optional[str] = Security(_header_chave)):
    if API_KEY and chave != API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key inválida ou ausente")


def modelo_carregado():
    if "classificador" not in estado:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo não encontrado em {MODEL_PATH}. Rode training/train.py com o CSV da base.",
        )
    return estado


def _nivel_risco(p: float, limiar: float) -> str:
    if p >= limiar:
        return "alto"
    return "medio" if p >= limiar * 0.6 else "baixo"


def _prever(leituras: list[Leitura], limiar: float) -> list[Predicao]:
    df = pd.DataFrame([l.model_dump() for l in leituras])
    X = df[NUMERICAS_BRUTAS + CATEGORICAS]
    proba = estado["classificador"].predict_proba(X)[:, 1]
    clusters = estado["kmeans"].predict(X)
    mapa = estado["metadata"]["kmeans"]["clusters"]

    saida = []
    for id_reg, p, c in zip(df["id_registro"], proba, clusters):
        info = mapa.get(int(c)) or mapa.get(str(int(c)), {})
        saida.append(Predicao(
            id_registro=id_reg,
            falha_prevista=bool(p >= limiar),
            probabilidade_falha=round(float(p), 4),
            nivel_risco=_nivel_risco(float(p), limiar),
            limiar_usado=limiar,
            cluster=int(c),
            regime_operacional=info.get("regime", f"Cluster {int(c)}"),
            taxa_falha_historica_regime=info.get("taxa_falha_historica", 0.0),
        ))
    return saida


_limiar = Query(None, ge=0.01, le=0.99, description="Limiar de decisão. Mais baixo = pega mais falhas, com mais alarmes falsos.")


@app.get("/", include_in_schema=False)
def raiz():
    return RedirectResponse("/docs")


@app.get("/health", tags=["status"])
def health():
    meta = estado.get("metadata", {})
    return {"status": "ok", "modelo_carregado": "classificador" in estado,
            "base_treino": meta.get("base_treino"), "treinado_em": meta.get("treinado_em")}


@app.get("/modelo", tags=["status"], dependencies=[Depends(verificar_chave)])
def info_modelo(est=Depends(modelo_carregado)):
    """Metadados do treino: métricas, hiperparâmetros, variáveis mais importantes e clusters."""
    return est["metadata"]


@app.post("/predict", response_model=Predicao, tags=["previsão"], dependencies=[Depends(verificar_chave)])
def predict(leitura: Leitura, limiar: Optional[float] = _limiar, est=Depends(modelo_carregado)):
    """Prevê falha de vazão para uma leitura."""
    return _prever([leitura], limiar or est["metadata"]["limiar_padrao"])[0]


@app.post("/predict/lote", response_model=RespostaLote, tags=["previsão"], dependencies=[Depends(verificar_chave)])
def predict_lote(lote: LoteLeituras, limiar: Optional[float] = _limiar, est=Depends(modelo_carregado)):
    """Prevê falha de vazão para até 1000 leituras de uma vez."""
    preds = _prever(lote.leituras, limiar or est["metadata"]["limiar_padrao"])
    return RespostaLote(total=len(preds), falhas_previstas=sum(p.falha_prevista for p in preds), predicoes=preds)
