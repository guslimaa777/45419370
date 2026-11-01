from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

LinhaProducao = Literal["Linha_A", "Linha_B", "Linha_C", "Linha_D"]
Turno = Literal["Manhã", "Tarde", "Noite"]
Produto = Literal["Polimero_A", "Polimero_B", "Polimero_C"]
Reagente = Literal["R1", "R2", "R3", "R4"]
TipoValvula = Literal["Globo", "Esfera", "Borboleta"]
StatusSensor = Literal["Normal", "Suspeito", "Descalibrado"]
ModoOperacao = Literal["Contínuo", "Lote"]

EXEMPLO = {
    "id_registro": "NEX_00001",
    "linha_producao": "Linha_B",
    "turno": "Tarde",
    "produto": "Polimero_C",
    "reagente_principal": "R3",
    "tipo_valvula": "Globo",
    "status_sensor": "Descalibrado",
    "modo_operacao": "Contínuo",
    "vazao_setpoint_l_min": 108.36,
    "vazao_real_l_min": 60.82,
    "pressao_bar": 3.1,
    "temperatura_c": 54.2,
    "abertura_valvula_pct": 66.0,
    "vibracao_bomba_mm_s": 2.6,
    "consumo_energia_kwh": 42.5,
    "tempo_resposta_s": 5.8,
    "ph_produto": 6.05,
    "viscosidade_cp": 295.1,
    "densidade_g_cm3": 1.248,
    "horas_operacao_bomba": 517.7,
    "leituras_invalidas": 1,
    "umidade_ambiente_pct": 45.3,
    "erro_vazao_pct": 43.9,
    "indice_estabilidade": 41.0,
}


class Leitura(BaseModel):
    """Uma leitura de sensores da planta. Campos opcionais aceitam null (são imputados)."""

    model_config = ConfigDict(json_schema_extra={"example": EXEMPLO})

    id_registro: Optional[str] = Field(None, description="Identificador opcional, devolvido na resposta")
    linha_producao: LinhaProducao
    turno: Turno
    produto: Optional[Produto] = None
    reagente_principal: Reagente
    tipo_valvula: TipoValvula
    status_sensor: Optional[StatusSensor] = None
    modo_operacao: ModoOperacao

    vazao_setpoint_l_min: float = Field(..., gt=0)
    vazao_real_l_min: float = Field(..., ge=0)
    pressao_bar: Optional[float] = None
    temperatura_c: Optional[float] = None
    abertura_valvula_pct: float = Field(..., ge=0, le=100)
    vibracao_bomba_mm_s: float = Field(..., ge=0)
    consumo_energia_kwh: float = Field(..., ge=0)
    tempo_resposta_s: Optional[float] = Field(None, ge=0)
    ph_produto: Optional[float] = Field(None, ge=0, le=14)
    viscosidade_cp: Optional[float] = Field(None, ge=0)
    densidade_g_cm3: float = Field(..., gt=0)
    horas_operacao_bomba: float = Field(..., ge=0)
    leituras_invalidas: int = Field(..., ge=0)
    umidade_ambiente_pct: float = Field(..., ge=0, le=100)
    erro_vazao_pct: float
    indice_estabilidade: float


class LoteLeituras(BaseModel):
    leituras: list[Leitura] = Field(..., min_length=1, max_length=1000)


class Predicao(BaseModel):
    id_registro: Optional[str]
    falha_prevista: bool
    probabilidade_falha: float = Field(..., description="Probabilidade estimada de falha de vazão (0 a 1)")
    nivel_risco: Literal["baixo", "medio", "alto"]
    limiar_usado: float
    cluster: int
    regime_operacional: str = Field(..., description="Regime identificado pelo K-Means")
    taxa_falha_historica_regime: float


class RespostaLote(BaseModel):
    total: int
    falhas_previstas: int
    predicoes: list[Predicao]
