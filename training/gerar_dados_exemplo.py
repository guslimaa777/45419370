"""Gera um CSV SINTÉTICO com o mesmo formato da base Nexus-Chem.

Serve só para testar a API sem a base real. Os números NÃO representam a planta;
para o modelo valer alguma coisa, treine com o CSV original do projeto.

Uso:
    python -m training.gerar_dados_exemplo --saida data/exemplo_sintetico.csv
"""
import argparse

import numpy as np
import pandas as pd


def gerar(n: int = 640, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    escolha = lambda opcoes, p=None: rng.choice(opcoes, size=n, p=p)

    status = escolha(["Normal", "Suspeito", "Descalibrado"], [0.6, 0.2, 0.2])
    setpoint = rng.normal(120, 15, n).round(2)
    drift = np.where(status == "Descalibrado", rng.normal(0.35, 0.1, n),
                     np.where(status == "Suspeito", rng.normal(0.15, 0.08, n), rng.normal(0.05, 0.04, n)))
    real = (setpoint * (1 - drift)).round(2)
    erro = (np.abs(setpoint - real) / setpoint * 100).round(2)
    vibracao = (rng.normal(2.2, 0.6, n) + drift * 2).round(2)
    estabilidade = (75 - erro * 1.2 + rng.normal(0, 5, n)).round(1)

    risco = -4 + 0.12 * erro + 0.8 * (vibracao - 2.2) + (status == "Descalibrado") * 0.8
    falha = (rng.random(n) < 1 / (1 + np.exp(-risco))).astype(int)
    regime = np.where(estabilidade > 62, "Estável", np.where(estabilidade > 50, "Atenção", "Crítico"))

    df = pd.DataFrame({
        "id_registro": [f"NEX_{i:05d}" for i in range(1, n + 1)],
        "linha_producao": escolha(["Linha_A", "Linha_B", "Linha_C", "Linha_D"]),
        "turno": escolha(["Manhã", "Tarde", "Noite"]),
        "produto": escolha(["Polimero_A", "Polimero_B", "Polimero_C"]),
        "reagente_principal": escolha(["R1", "R2", "R3", "R4"]),
        "tipo_valvula": escolha(["Globo", "Esfera", "Borboleta"]),
        "status_sensor": status,
        "modo_operacao": escolha(["Contínuo", "Lote"]),
        "vazao_setpoint_l_min": setpoint,
        "vazao_real_l_min": real,
        "pressao_bar": rng.normal(4.2, 0.8, n).round(2),
        "temperatura_c": rng.normal(58, 6, n).round(1),
        "abertura_valvula_pct": rng.normal(65, 10, n).round(1),
        "vibracao_bomba_mm_s": vibracao,
        "consumo_energia_kwh": rng.normal(42, 6, n).round(2),
        "tempo_resposta_s": rng.normal(3.5, 1, n).round(2),
        "ph_produto": rng.normal(6.6, 0.4, n).round(2),
        "viscosidade_cp": rng.normal(290, 20, n).round(1),
        "densidade_g_cm3": rng.normal(1.28, 0.05, n).round(3),
        "horas_operacao_bomba": rng.uniform(100, 900, n).round(1),
        "leituras_invalidas": rng.poisson(0.8, n),
        "umidade_ambiente_pct": rng.normal(55, 8, n).round(1),
        "erro_vazao_pct": erro,
        "indice_estabilidade": estabilidade,
        "falha_vazao": falha,
        "regime_operacional_referencia": regime,
    })
    # ~2% de faltantes, como na base original
    for col in ["produto", "status_sensor", "pressao_bar", "temperatura_c",
                "tempo_resposta_s", "ph_produto", "viscosidade_cp"]:
        df.loc[rng.random(n) < 0.022, col] = np.nan
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saida", default="data/exemplo_sintetico.csv")
    parser.add_argument("--n", type=int, default=640)
    args = parser.parse_args()
    gerar(args.n).to_csv(args.saida, index=False)
    print(f"CSV sintético salvo em {args.saida}")
