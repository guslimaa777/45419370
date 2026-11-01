# Nexus-Chem — API de Falha de Vazão

API REST (FastAPI) que coloca em produção o projeto integrador **Otimização da Planta Química Nexus-Chem**:

- **Random Forest**: prevê se uma leitura de sensores indica **falha de vazão** (`falha_vazao`) e devolve a probabilidade.
- **K-Means**: identifica o **regime operacional** da leitura (Estável / Atenção / Alto risco), nomeado pela taxa de falha histórica de cada cluster.

O pré-processamento do notebook (remoção de duplicatas, mediana e moda nos faltantes, capping por IQR, `razao_vazao`, `desvio_vazao_abs`, one-hot e StandardScaler) roda dentro de um `Pipeline` do scikit-learn. Assim a API recebe os dados **brutos**, do jeito que vêm da planta, e aplica exatamente as mesmas transformações do treino.

> O modelo em `artifacts/` foi treinado com o arquivo `dataset_projeto_integrador_nexus_chem.csv` fornecido para este projeto: 638 linhas lidas, com 18 duplicatas removidas (620 registros no treino). O `GET /health` informa em `base_treino` qual arquivo originou o modelo carregado.

## Estrutura

```
app/
  main.py        # endpoints
  schemas.py     # validação de entrada e saída (Pydantic)
  features.py    # engenharia de atributos e capping (usados no treino e na API)
training/
  train.py                 # treina e salva artifacts/modelo_nexus_chem.joblib
  gerar_dados_exemplo.py   # CSV sintético para testes
artifacts/       # modelo treinado + metadados (.json)
tests/           # pytest
treinar_no_colab.ipynb     # treino com a base real direto no Colab
Dockerfile
```

## 1. Treinar com a base real

**No Colab (sem instalar nada):** suba este projeto para o seu GitHub, abra `treinar_no_colab.ipynb` no Colab, troque o link do repositório na célula 2 e rode tudo. Na célula 3 ele pede o `dataset_projeto_integrador_nexus_chem.csv`. No fim, baixe os dois arquivos gerados e substitua os que estão em `artifacts/`.

**Ou local:**
```bash
pip install -r requirements-dev.txt
python -m training.train --csv data/dataset_projeto_integrador_nexus_chem.csv
```

## 2. Rodar local (opcional)

```bash
uvicorn app.main:app --reload
```
Depois abra http://localhost:8000/docs, que tem a documentação interativa com botão "Try it out". Para rodar os testes: `pytest -q`.

## 3. Publicar

**Render** (mais simples)
1. Suba o projeto para o GitHub, já com o modelo treinado em `artifacts/`.
2. No Render, escolha *New → Blueprint* e selecione o repositório. O `render.yaml` configura o serviço Docker, o plano Free e a rota de health check.
3. (Opcional) Em *Environment*, crie a variável `API_KEY` para exigir senha.
4. A URL sai no formato `https://nexus-chem-api.onrender.com/docs`. No plano free, o serviço pode dormir após inatividade; a primeira chamada pode demorar enquanto ele inicia.

**Hugging Face Spaces**
1. *New Space* → SDK **Docker** → *Blank*.
2. Envie os arquivos do projeto (pelo site ou com `git push`). A porta 7860 já está configurada.

## Endpoints

| Método | Rota | O que faz |
|---|---|---|
| GET | `/health` | Status e qual base treinou o modelo |
| GET | `/modelo` | Métricas de teste, hiperparâmetros, top variáveis, clusters |
| POST | `/predict` | Previsão para uma leitura |
| POST | `/predict/lote` | Previsão para até 1000 leituras |

Os dois POSTs aceitam `?limiar=0.3`, por exemplo. Um limiar mais baixo captura mais falhas, ao custo de mais alarmes falsos. No contexto da planta, deixar passar uma falha costuma custar mais caro que um alarme falso.

Se a variável `API_KEY` estiver definida, envie o header `X-API-Key`.

### Exemplo

```bash
curl -X POST https://SUA-URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "linha_producao": "Linha_B", "turno": "Tarde", "produto": "Polimero_C",
    "reagente_principal": "R3", "tipo_valvula": "Globo", "status_sensor": "Descalibrado",
    "modo_operacao": "Contínuo", "regime_operacional_referencia": "Crítico",
    "vazao_setpoint_l_min": 108.36, "vazao_real_l_min": 60.82, "pressao_bar": 3.1,
    "temperatura_c": 54.2, "abertura_valvula_pct": 66.0, "vibracao_bomba_mm_s": 2.6,
    "consumo_energia_kwh": 42.5, "tempo_resposta_s": 5.8, "ph_produto": 6.05,
    "viscosidade_cp": 295.1, "densidade_g_cm3": 1.248, "horas_operacao_bomba": 517.7,
    "leituras_invalidas": 1, "umidade_ambiente_pct": 45.3,
    "erro_vazao_pct": 43.9, "indice_estabilidade": 41.0
  }'
```

Resposta:
```json
{
  "id_registro": null,
  "falha_prevista": true,
  "probabilidade_falha": 0.9,
  "nivel_risco": "alto",
  "limiar_usado": 0.5,
  "cluster": 1,
  "regime_operacional": "Alto risco",
  "taxa_falha_historica_regime": 0.7328
}
```

Os campos `produto`, `status_sensor`, `pressao_bar`, `temperatura_c`, `tempo_resposta_s`, `ph_produto` e `viscosidade_cp` aceitam `null`. São as colunas que tinham faltantes na base, e o pipeline imputa esses valores. Uma categoria fora da lista (ex.: `"turno": "Madrugada"`) volta com erro 422, explicando o que é aceito.

## Pontos de atenção

- **`regime_operacional_referencia` como entrada:** o notebook usa essa coluna como preditora, e a API mantém isso. Se em operação real ela só for conhecida *depois* do evento, é vazamento de dados. Vale retreinar sem ela (basta tirá-la de `CATEGORICAS` em `app/features.py`).
- **Overfitting:** no notebook, o F1 de treino foi 0,92 contra 0,50 no teste. O `GET /modelo` mostra as métricas de treino e de teste lado a lado, para acompanhar isso a cada novo treino.
- **Versões:** o modelo `.joblib` só carrega com a mesma versão do scikit-learn usada no treino (fixada no `requirements.txt` e no notebook do Colab).
