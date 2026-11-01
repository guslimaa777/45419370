"""Testes da API. Rodar com: pytest -q

Precisa de um modelo em artifacts/ (treinado com a base real ou com o CSV sintético).
"""
import pytest
from fastapi.testclient import TestClient

from app.main import MODEL_PATH, app
from app.schemas import EXEMPLO

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="treine o modelo antes (training/train.py)")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["modelo_carregado"] is True


def test_modelo(client):
    r = client.get("/modelo")
    assert r.status_code == 200
    assert "metricas_teste" in r.json()


def test_predict(client):
    r = client.post("/predict", json=EXEMPLO)
    assert r.status_code == 200, r.text
    corpo = r.json()
    assert 0 <= corpo["probabilidade_falha"] <= 1
    assert corpo["nivel_risco"] in {"baixo", "medio", "alto"}
    assert corpo["id_registro"] == "NEX_00001"


def test_predict_com_faltantes(client):
    leitura = {**EXEMPLO, "produto": None, "pressao_bar": None, "ph_produto": None}
    assert client.post("/predict", json=leitura).status_code == 200


def test_limiar_muda_decisao(client):
    p = client.post("/predict", json=EXEMPLO).json()["probabilidade_falha"]
    baixo = client.post("/predict", params={"limiar": 0.01}, json=EXEMPLO).json()
    alto = client.post("/predict", params={"limiar": 0.99}, json=EXEMPLO).json()
    assert baixo["falha_prevista"] is (p >= 0.01)
    assert alto["falha_prevista"] is (p >= 0.99)


def test_lote(client):
    r = client.post("/predict/lote", json={"leituras": [EXEMPLO, {**EXEMPLO, "id_registro": "NEX_2"}]})
    assert r.status_code == 200
    assert r.json()["total"] == 2


def test_categoria_invalida(client):
    r = client.post("/predict", json={**EXEMPLO, "turno": "Madrugada"})
    assert r.status_code == 422


def test_campo_obrigatorio(client):
    leitura = {k: v for k, v in EXEMPLO.items() if k != "vazao_real_l_min"}
    assert client.post("/predict", json=leitura).status_code == 422
