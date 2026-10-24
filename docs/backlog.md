# Backlog de mejoras — d2ql

Documento único de mejoras, decisiones y deuda técnica del proyecto (DDQN sobre CloudSimPlus
para *load balancing* en la nube, eje **precisión × capacidad**).

Reemplaza a los históricos `h4_early_stopping_y_resultados.md`, `h6_energy_reward.md`,
`r1_recommendations.md` y `r2_decisiones_precision_capacidad.md` (eliminados). `h5_heuristics.md`
sigue siendo el documento operativo de ese experimento.

> Última revisión: **2026-10-24** (cuarta tanda: EVAL-04, LOG-04, LOG-05, LOG-06, LOG-07, LOG-08,
> SIM-08, REPO-09 + export de TensorBoard a series CSV; se descubrió **SIM-09**). Alcance:
> `python-agent/`, `java-sim/`, `configs/`, `tests/`, `docs/`, `notebooks/`, `docker-compose*.yml`,
> `Dockerfile.python`, `pyproject.toml`.

---

## 1. Lineamientos

1. **Fuente única.** Toda sugerencia, decisión o deuda vive aquí; no se abren documentos por sprint.
   Orden: propuestos → hechos → descartados → decisiones.
2. **ID estable.** `ÁREA-nn` con áreas `LOG` (TensorBoard/series), `CSV` (resultados persistidos),
   `EVAL` (evaluación/comparabilidad), `SIM` (simulador + entorno), `REPO` (reproducibilidad/higiene),
   `PERF` (precisión y kernels). Nunca se reutiliza ni se renombra.
3. **Estados:** `propuesto` → `en curso` → `hecho` | `descartado` | `bloqueado`.
4. **Prioridades:** `P0` invalida cifras publicadas; `P1` comparabilidad/reproducibilidad/corrección
   latente; `P2` deuda sin efecto en las cifras; `P3` idea opcional.
5. **Definición de hecho:** código/config aplicado **y** evidencia reproducible (test, comando con
   salida, o fila de CSV). Sin evidencia el ítem no sale de "propuesto".
6. **Descartar es una decisión:** lleva porqué y sustituto. Si se retoma, vuelve a `propuesto` con el
   mismo ID.
7. **Un ítem = un problema.** Si un cambio toca dos problemas, se abren dos ítems y se referencian.
8. **Procedencia obligatoria.** Toda fila persistida lleva `schema_version` + `git_sha` +
   `config_sha256`; sin eso no se puede afirmar que dos cifras sean comparables.
9. **Series crudas siempre exportadas.** Cada run deja su TensorBoard **y** su versión CSV sin
   downsample (ver §7); el CSV agregado (`h4_results.csv`) es el resumen, no la fuente de las curvas.

### Verificación

```bash
docker run --rm -v $PWD:/work -w /work --entrypoint python d2ql-python-agent -m pytest tests/ -o addopts=""
docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work --entrypoint ruff d2ql-python-agent check --no-cache .
docker build -f Dockerfile.java -t d2ql-java-sim:latest .        # tras tocar el gateway
GIT_SHA=$(git rev-parse --short HEAD) docker compose build python-agent
```

---

## 2. Resumen

| Estado | P0 | P1 | P2 | P3 | Total |
|---|---|---|---|---|---|
| propuesto | 1 | 0 | 3 | 2 | 6 |
| hecho | 0 | 0 | 0 | 0 | 38 |
| descartado | 1 | 1 | 0 | 0 | 2 + 5 históricos |
| bloqueado | 0 | 0 | 0 | 0 | 0 (4 decisiones abiertas) |

Queda un solo P0 abierto y es nuevo: **SIM-09** (la energía se cobra por paso como si fuera 1 s de
simulación, pero el reloj avanza ~0.01 s por paso). Cualquier cifra de energía/reward publicada hoy
está inflada ~100× respecto al tiempo simulado.

---

## 3. PROPUESTOS

### P0 — invalidan cifras publicadas

#### SIM-09 · La energía se cobra por paso asumiendo 1 s, pero el reloj avanza ~0.01 s
- **Estado:** propuesto · **P0**
- **Evidencia (diagnóstico con el gateway actual, 10 cloudlets, 18 pasos):**
  `clock` avanza 0.000 → 0.010 → 0.048 → … → **0.158 s en total**, mientras
  `getTotalEnergyConsumed()` sube ~0.108–0.12 Wh por paso (≈ **18 s** de potencia). Es decir, la
  energía acumulada corresponde a ~114× el tiempo simulado transcurrido.
  Causa: `SimulationGateway.step()` hace `simulation.runFor(1.0)` pero el reloj avanza solo hasta el
  siguiente evento (≈0.01 s); `updateEnergyAndSla()` suma `power × 1 s / 3600` con un 1 s fijo.
- **Impacto:** `episode/energy`, `episode/total_reward`, `eval_*` y cualquier comparación energética
  están en una escala que no es tiempo simulado. Además `norm_reward` (reward por paso) no es
  "reward por segundo".
- **Fix propuesto:** (a) acumular energía con el **dt real** (`clock − clock_prev`) en
  `updateEnergyAndSla`, que es físicamente correcto sea cual sea la mecánica de `runFor`; y (b) revisar
  el ciclo de vida `startSync()` + `runFor()` para que cada paso avance exactamente 1 s si eso es lo
  que se quiere simular.
- **Criterio:** `Σ energía` de un episodio ≈ `P_medio × (clock_final − clock_inicial)` y
  `Δclock == 1.0` por paso.

### P2 — deuda

#### SIM-06 · El gateway todavía calcula el SLA y emite la 12.ª dimensión
- **Estado:** propuesto · **P2** (requiere reconstruir `java-sim`; el build tarda ~1–2 min y funciona)
- **Evidencia:** `updateEnergyAndSla` mantiene `slaViolationCount`/`countedSlaCloudlets`,
  `buildObservation()` devuelve `N_HOSTS*2 + 1 + 3` con `time_to_deadline` constante y
  `addWorkloadRow(submitted, deadline, mi, pes)` arrastra un `deadline` que ya nadie usa. Python ignora
  el contador y recorta a 11 dims (`env._parse_obs`).
- **Fix:** borrar el contador SLA, emitir 11 dims y quitar `deadline` del protocolo/records; ajustar
  `SimulationGatewayRegressionTest`.

#### EVAL-07 · El split temporal existe pero nadie lo mide
- **Estado:** propuesto · **P2**
- **Evidencia:** `workload.temporal_episodes()` conserva la cola cronológica como test de
  generalización temporal, pero ningún camino de `train.py` lo llama ni lo persiste.
- **Fix:** `evaluation.split: stratified|temporal` y columnas `eval_temporal_*` aparte.

#### PERF-04 · El eval estratificado encarece la evaluación
- **Estado:** propuesto · **P2**
- **Evidencia:** la cola cronológica tenía MI medio 25; el split estratificado, 221 (EVAL-01). Cada
  episodio de evaluación tiene ~10× más trabajo y el eval de 66 celdas × 2 semillas pesa más.
- **Fix:** medir `_run_heldout_eval` y, si pesa, bajar `evaluation.n_episodes` (10 → 5) o cachear las
  ventanas por corrida.

### P3 — opcional

#### PERF-02 · El notebook mide "tamaño" con `capas × neuronas × bits`
- **Estado:** propuesto · **P3**
- **Evidencia:** `notebooks/results_analisis.ipynb` usa `n_layers * hidden_size * bits`, que ignora
  que los parámetros crecen con `hidden²` dentro de cada capa oculta.
- **Fix:** usar las columnas persistidas `params`, `flops` o `effective_capacity_bits`.

#### REPO-10 · `analyze_pareto.py` no es instalable como comando
- **Estado:** propuesto · **P3**
- **Evidencia:** vive en `python-agent/` (fuera del paquete), así que no hay entry point.
- **Fix:** moverlo a `d2ql/pareto.py` y añadir `pareto = "d2ql.pareto:main"`.

---

## 4. HECHOS

| Ítem | Qué se hizo | Dónde | Evidencia |
|---|---|---|---|
| **LOG-09** series TB → CSV | Export automático al terminar cada run: `series_long.csv` (todos los eventos), un CSV ancho por namespace, `text.csv`, `run.json` y copia de `result.json`; sin downsample y sin enmascarar NaN | `tb_export.py`, `train.py` | smoke: 574 eventos/27 tags → `tb_series/smoke/…/{episode,train,buffer,stopper,eval,perf}.csv`; `tests/test_tb_export.py` (7) |
| **EVAL-04** profundidad en el Pareto | `GROUP_KEYS` incluye `n_hidden_layers`; las celdas ya no mezclan 1 y 2 capas | `analyze_pareto.py` | cabecera imprime "celdas (precisión×device×ancho×profundidad)" |
| **LOG-04** pérdida/Q por updates | `episode/avg_loss` y `episode/mean_q` dividen por nº de updates; si no hubo updates no se registra | `train.py`, `metrics.py` | episodio 1 del smoke: 0.1321 (= media real de sus 11 updates) frente a 0.0799 antes (×0.59) |
| **LOG-05** reward/energía por paso | Nuevos tags `episode/steps`, `episode/reward_per_step`, `episode/energy_per_step` | `metrics.py`, `train.py` | `episode.csv` del smoke: steps 16–20, reward/step −0.459…−0.488 |
| **LOG-06** sin NaN en TB | `log_scalar` descarta valores no finitos; `log_eval`/`log_perf` usan esa ruta | `metrics.py` | `tests/test_tb_export.py` + runs sin eval ya no crean tags vacíos |
| **LOG-07** energía dinámica | Nuevo `power.idle_wh_per_step()` (390 W → 0.10833 Wh/paso) y tag `episode/dynamic_energy = energy − idle×steps` | `power.py`, `train.py`, `metrics.py` | `episode.csv`: energy 2.474 Wh, dynamic 0.307 Wh (idle 2.167 = 20 × 0.10833) |
| **LOG-08** diagnóstico de entrenamiento | `train/grad_norm`, `train/td_error_abs`, `train/lr`, `stopper/window_mean`, `stopper/best_window_mean` y `run/es_reason` (texto) | `agent.py`, `metrics.py`, `train.py` | `train.csv` con 5 columnas; `text.csv` con el motivo de parada |
| **SIM-08** fin de episodio en el makespan | El gateway expone `isEpisodeComplete()` (todos despachados y con `finishTime > 0`) y el env lo usa; fallback a `isFinished()` con aviso | `SimulationGateway.java`, `env.py` | episodios del smoke 16–20 pasos (antes 18–22); el gateway se reconstruyó y el env detecta la capacidad |
| **REPO-09** SHA dentro de la imagen | `ARG GIT_SHA` + `ENV D2QL_GIT_SHA` en `Dockerfile.python`; `build.args` en `docker-compose.yml` (python-agent y tensorboard) | `Dockerfile.python`, `docker-compose.yml` | `docker compose config` válido; `provenance.git_sha()` lee la env var |
| **EVAL-01** hold-out estratificado | Muestra sistemática sobre el orden por MI, con offset de medio bloque; cola temporal separada en `temporal_episodes()` | `workload.py` | MI medio train 221.4 vs eval 221.1 (0.14 %) en la traza real; `tests/test_holdout.py` |
| **EVAL-06** esquema/procedencia | `schema_version` por fila; parquet actuales documentados como v1; `analyze_pareto` avisa si se mezclan versiones | `provenance.py`, `analyze_pareto.py`, `notebooks/data/README.md` | `extra.schema_version = 2` en el smoke |
| **CSV-04** procedencia | `git_sha`, `config_sha256`, `run_timestamp_utc`, `obs_dim`, `dispatch_order`, `eval_split`, `tb_series_dir` | `provenance.py`, `train.py` | `tests/test_provenance.py` (5) |
| **REPO-01** entry points | `train = "d2ql.train:main"`, fuera `build-deps`; orquestador en `d2ql/train.py` y `main.py` como shim | `pyproject.toml`, `d2ql/train.py` | `importlib.import_module("d2ql.train")` OK |
| **REPO-02** Triton opcional | Extra `kernels = ["triton>=3.0"]`; fallback con aviso explícito | `pyproject.toml`, `precision.py` | extras `['dev','kernels']` |
| **REPO-04** env muertas | 6 variables eliminadas de `docker-compose.yml` | `docker-compose.yml` | `docker compose config` válido |
| **REPO-05** código muerto | `h4_capacity_plan` eliminado | `precision.py` | sin referencias |
| **SIM-05** reward en el env | `CloudSimEnv` construye su `RewardManager` y `step()` devuelve `r_t ∈ [-1,0]` | `env.py`, `train.py` | `tests/test_env_order.py` (delta 0 → 0.0; full → −1.0) |
| **PERF-03** coste del padding | `benchmark_padding_overhead()` (batch 1 vs 17) + columna `padding_overhead_ms` + tag `perf/padding_overhead_ms` | `agent.py`, `results.py`, `train.py` | smoke CPU: 0.0195 ms |
| **CSV-01** reward normalizado | `best_mean_reward` = media por episodio de la mejor ventana; `norm_reward` por paso; `eval_avg_steps`/`eval_norm_reward`; fuera `best_eval_*` | `train.py`, `results.py` | smoke: −9.29 / 20.25 = −0.4588 |
| **CSV-02** upsert | Reemplaza la fila con la misma `(experiment_id, seed)` | `results.py` | `tests/test_results.py` |
| **CSV-03** sin evaluación | `NaN` en vez de `0.0` | `train.py` | smoke 1 episodio → `NaN` |
| **LOG-01** eje de `train/*` | Una serie por tag con x = `agent.total_steps`; promedios por episodio en `episode/*` | `metrics.py`, `train.py` | TB: `train/loss` pasos 1…N únicos |
| **LOG-02** pesos adaptativos | Fuera `log_weights`, `delta_*` y la API no-op del reward | `metrics.py`, `reward.py` | sin tags `weights/*` |
| **LOG-03** telemetría de eval/despliegue | `eval/*` y `perf/*` al cerrar el run | `metrics.py` | TB del smoke |
| **EVAL-02** H5 vs H4 | `simulation_time_scale: 0.01` en H5 | `configs/h5_heuristics.yaml` | smoke H5 corre con esa escala |
| **EVAL-05** semillas | `[42, 41]` → 132 corridas | `configs/h4_native_precision.yaml`, `tests/test_h4_plan.py` | guard en verde |
| **SIM-02** topología fija | `N_HOSTS = 4` en `env.py`; fuera `n_cloud_hosts` | `env.py`, configs | `tests/test_env_order.py` |
| **SIM-03** potencia de fuente única | `d2ql/power.py` + `RewardManager` que solo recibe `e_step_max` | `power.py`, `reward.py` | `tests/test_power.py` compara los 4 arrays con el Java |
| **SIM-04** orden aplicado | LPT por trabajo/PE enviado al gateway en ese orden | `queue.py`, `env.py` | `tests/test_queue.py`, `tests/test_env_order.py` |
| **SLA retirado** | Fuera del CSV/JSON/TB/docs/notebooks/reward | `results.py`, `env.py`, `reward.py`, docs | `tests/test_results.py` |
| **SIM-07** el gateway se apagaba | `close()` usa `gateway.close()` en vez de `shutdown()` | `env.py` | 3 corridas seguidas con el contenedor `Up` |
| **REPO-06** lint | `ruff --fix` + excepciones documentadas | repo | `ruff check` limpio |
| **REPO-07** API no-op del reward | `RewardManager` solo `compute_step_reward` + `e_step_max` | `reward.py` | `tests/test_reward.py` |
| **REPO-08** doc de H5 | Sin `notes` ni métricas SLA | `docs/h5_heuristics.md` | columnas del doc = CSV |
| **PERF-01** eje de batch | `[1, 8, 32, 128, 256, 512, 1024]` | `configs/h4_native_precision.yaml` | curva en `throughput_by_batch` |
| **REPO-03** guard del plan | 66 celdas × 2 semillas = 132 | `tests/test_h4_plan.py` | 80 tests en verde |
| **H1/H2 retirados** | Configs y `d2ql/quantization.py` eliminados | `configs/`, `d2ql/` | sin referencias |
| **Docs unificados** | H4, H6, R1 y R2 absorbidos aquí | `docs/` | quedan `backlog.md` y `h5_heuristics.md` |

---

## 5. DESCARTADOS

| Ítem | Porqué | Sustituto |
|---|---|---|
| **SIM-01** SLA como eje | Deadline absoluto (~1e6 s) contra un reloj de ~0.2 s: el contador daba 0 en 132/132 filas | energía/tamaño/latencia; métrica eliminada del CSV |
| **EVAL-03** SLA del último episodio | El contador ya no existe | — |
| Migración real de cloudlets | CloudSimPlus no la expone de forma simple y no aporta al objetivo energético | scheduling de colocación |
| int8 como precisión estudiada | Sin VNNI en Zen 3, un GEMM int8 solo se emula | ternario bit-packed en CPU |
| Triton como backend por defecto | Triton 3.2 corrompe `tl.dot` int8 con ±127 | `int_mm` por defecto, Triton opt-in |
| PTQ (H2) | No captura "entrenar desde cero a baja precisión" | H4 (entrenamiento nativo) |
| bf16 en CPU | Zen 3 sin AVX512-BF16 → emulación | bf16 solo en GPU |

---

## 6. Decisiones abiertas

1. **Re-correr el sweep:** regenerar `data_h4.parquet` como v2 (11 dims, LPT, split estratificado,
   132 celdas) o conservar el v1 como histórico. **Ahora además** depende de decidir SIM-09 (escala de
   energía), porque cambia todas las magnitudes.
2. **SIM-09:** ¿se corrige la energía con el `dt` real (magnitudes bajan ~100×) y/o se arregla el ciclo
   `startSync`/`runFor` para que cada paso sea 1 s?
3. **Reconstruir `java-sim` (SIM-06):** quitar el SLA y la 12.ª dimensión (el build ya está verificado:
   ~1–2 min con Maven y red).
4. **Hardware objetivo del despliegue:** decide si el eje es `packed_size_mb` (memoria/edge) o latencia
   real (servidor), y si tiene sentido seguir escalando batch.
5. **Orden de despacho (SIM-04):** LPT por trabajo/PE; alternativas SPT o el orden del trace como
   baseline explícito.

---

## 7. Anexo — export de TensorBoard a series CSV

Cada run escribe, al terminar (y también por CLI para runs antiguos):

```
outputs/results/tb_series/<experiment_id>/<stamp>/
├── series_long.csv     tag, step, wall_time_unix, wall_time_utc, value   (todos los eventos)
├── episode.csv         step + una columna por tag de episode/*
├── train.csv           step + loss, mean_q, grad_norm, td_error_abs, lr
├── buffer.csv          step + size
├── stopper.csv         step + window_mean, best_window_mean
├── eval.csv            step + mean_reward, norm_reward, makespan, avg_steps
├── perf.csv            step + latency_*, padding_overhead_ms, throughput_pps
├── text.csv            tag, step, wall_time_utc, text   (run/es_reason)
├── result.json         copia del result.json del checkpoint (float64 autoritativo)
└── run.json            recuentos, rangos de step, no finitos por tag, metadatos
```

Garantías: sin downsample (`size_guidance=0`), orden determinista por `(tag, step)`, un fichero ancho
por namespace (nunca se mezclan ejes x distintos), `nan`/`inf` visibles, export idempotente.
**Aviso:** TensorBoard guarda los escalares en **float32**; el CSV es exacto respecto al event file,
pero la referencia float64 es `result.json`/`h4_results.csv`.

```bash
# Re-exportar un run o un árbol completo de runs antiguos
docker compose run --rm python-agent python -m d2ql.tb_export \
  --logdir outputs/tensorboard --outdir outputs/results/tb_series
```
