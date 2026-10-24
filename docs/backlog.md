# Backlog de mejoras — d2ql

Documento único de mejoras, decisiones y deuda técnica del proyecto (DDQN sobre CloudSimPlus
para *load balancing* en la nube, eje **precisión × capacidad**).

Reemplaza a los históricos `h4_early_stopping_y_resultados.md`, `h6_energy_reward.md`,
`r1_recommendations.md` y `r2_decisiones_precision_capacidad.md` (eliminados). `h5_heuristics.md`
sigue siendo el documento operativo de ese experimento.

> Última revisión: **2026-10-24** (aplicación de CSV-01, CSV-02, CSV-03, EVAL-02, EVAL-05, LOG-01,
> LOG-02, LOG-03, SIM-02, SIM-03, SIM-04, REPO-06, REPO-07, REPO-08, PERF-01 y retiro del eje SLA).
> Alcance: `python-agent/`, `java-sim/`, `configs/`, `tests/`, `docs/`, `notebooks/`.

---

## 1. Lineamientos

1. **Fuente única.** Toda sugerencia, decisión o deuda vive aquí; no se abren documentos nuevos por
   sprint. El documento está **ordenado por estado**: propuestos → hechos → descartados.
2. **ID estable.** `ÁREA-nn`, con áreas `LOG` (TensorBoard), `CSV` (resultados persistidos), `EVAL`
   (evaluación/comparabilidad), `SIM` (simulador + entorno), `REPO` (reproducibilidad/higiene),
   `PERF` (precisión y kernels). Nunca se reutiliza ni se renombra.
3. **Estados:** `propuesto` → `en curso` → `hecho` | `descartado` | `bloqueado`.
4. **Prioridades:** `P0` invalida cifras publicadas; `P1` comparabilidad/reproducibilidad/corrección
   latente; `P2` deuda sin efecto en las cifras; `P3` idea opcional.
5. **Definición de hecho:** código/config aplicado **y** evidencia reproducible (test, comando con
   salida, o fila de CSV). Sin evidencia el ítem no se mueve de "propuesto".
6. **Descartar es una decisión:** todo descartado lleva el porqué y el sustituto. Si se retoma, vuelve
   a `propuesto` con el mismo ID.
7. **Un ítem = un problema.** Si un cambio toca dos problemas, se abren dos ítems y se referencian.

### Verificación (comandos usados en esta revisión)

```bash
# tests (el torch del host no importa: falta libnccl)
docker run --rm -v $PWD:/work -w /work --entrypoint python d2ql-python-agent -m pytest tests/ -o addopts=""
# lint (--user para no dejar archivos root en el repo)
docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work --entrypoint ruff d2ql-python-agent check --no-cache .
# análisis del sweep
docker compose run --rm python-agent python analyze_pareto.py --results outputs/results/h4_results.csv
```

---

## 2. Resumen

| Estado | P0 | P1 | P2 | P3 | Total |
|---|---|---|---|---|---|
| propuesto | 2 | 2 | 5 | 1 | 10 |
| hecho | 0 | 0 | 0 | 0 | 16 |
| descartado | 1 | 1 | 0 | 0 | 2 + 4 decisiones |
| bloqueado | 0 | 0 | 0 | 0 | 0 |

Lo que bloquea volver a publicar cifras: **EVAL-01** (hold-out 10.7× más ligero) y **EVAL-06**
(el código actual ya no reproduce el parquet: observación 12→11, orden traza→LPT, semillas 3→2 y SLA
fuera del CSV).

---

## 3. PROPUESTOS

### P0 — invalidan cifras publicadas

#### EVAL-01 · El hold-out cronológico es 10.7× más ligero que el entrenamiento
- **Estado:** propuesto · **P0**
- **Evidencia:** `workload.py:109-120` reserva el 20 % final de ventanas. Con `episode_length=50`:
  **train 40 817 ventanas, MI medio 270** (p95 1685); **test 10 204 ventanas, MI medio 25** (p95 129)
  → ratio **10.7×**. Coincide con los datos: `eval_makespan` ≈ 0.59 s frente a episodios de
  entrenamiento de ~166 pasos.
- **Impacto:** `eval_*` mide otra carga, así que no es comparable con ninguna métrica de
  entrenamiento y no sirve como estimación del régimen entrenado.
- **Fix propuesto:** hold-out **estratificado por MI** (o k-fold con la misma distribución) para la
  métrica de calidad; conservar el corte temporal como test de generalización aparte.
- **Criterio:** `|MI_train − MI_eval| / MI_train < 10 %` y `eval_makespan` del mismo orden que el de
  entrenamiento.

#### EVAL-06 · El código actual no reproduce el sweep publicado (132 filas)
- **Estado:** propuesto · **P0**
- **Evidencia:** las 132 filas de `notebooks/data/data_h4.parquet` se generaron con: observación de
  **12 dims** (incluida `time_to_deadline`), despacho en **orden del trace**, **2 semillas** (41, 42) y
  columnas `eval_sla_*`. El código fusionado ahora usa **11 dims** (SIM-01), **orden LPT** (SIM-04),
  el config pide `[42, 41]` (EVAL-05) y el CSV ya no tiene SLA. Cambiar el espacio de observación
  además invalida los checkpoints previos (`agent.load` falla por `state_dim`).
- **Impacto:** cualquier cifra nueva no es comparable con las publicadas; el parquet actual describe
  otro entorno.
- **Fix propuesto:** decidir si se re-corre el sweep (66 celdas × 2 semillas) y se regenera
  `data_h4.parquet`, o si se conserva el parquet como "resultados v1" etiquetado con su configuración.
- **Criterio:** el parquet declara en su metadata/`extra` la versión de código y coincide con
  `len(plan) × len(seeds)`.

### P1 — comparabilidad / reproducibilidad

#### EVAL-04 · `analyze_pareto.py` agrega sin la profundidad
- **Estado:** propuesto · **P1**
- **Evidencia:** `analyze_pareto.py:47` → `GROUP_KEYS = ["precision","device","hidden_size"]`, pero el
  sweep barre `n_hidden_layers: [1, 2]`. Cada "celda" promedia modelos de 1 y 2 capas (capacidad y
  FLOPs distintos).
- **Fix:** incluir `n_hidden_layers` en `GROUP_KEYS`.
- **Criterio:** nº de celdas = precisiones × devices × anchos × profundidades.

#### CSV-04 · No queda registrada la procedencia de cada corrida
- **Estado:** propuesto · **P1**
- **Evidencia:** `RunResult.extra` guarda `eval_every`, `n_episodes_budget`, `es_reason` y
  `simulation_time_scale`; no hay commit, ni hash del config, ni fecha de corrida. Dos filas con el
  mismo `experiment_id` pueden venir de código distinto (justo lo que pasó entre `c4063eb` y esta
  revisión).
- **Fix:** añadir `git_sha`, `config_sha256` y `run_timestamp` a `extra`.
- **Criterio:** una fila del CSV se puede reproducir sabiendo commit + config.

### P2 — deuda

#### REPO-01 · Entry points del paquete rotos
- **Estado:** propuesto · **P2**
- **Evidencia:** `pyproject.toml` → `train = "main:main"` (el wheel solo empaqueta `python-agent/d2ql`;
  `main` no queda instalado) y `build-deps = "scripts.build_deps:main"` (ese módulo no existe).
- **Fix:** borrar `build-deps`; mover el orquestador a `d2ql.cli:main` o ajustar el wheel.

#### REPO-02 · `triton` se importa pero no se declara
- **Estado:** propuesto · **P2**
- **Evidencia:** `kernels.py:27` importa `triton` en el top-level; no está en `dependencies` ni en un
  extra, y el `try/except` de `precision.py` convierte el fallo en un *fallback* silencioso.
- **Fix:** extra opcional `triton` o import perezoso dentro de `lowbit_matmul_triton`.

#### REPO-04 · Variables de entorno muertas en `docker-compose*.yml`
- **Estado:** propuesto · **P2** (la parte de configs ya está hecha)
- **Evidencia:** `DATASET_PATH`, `WORKLOAD_TIME_SCALE`, `WORKLOAD_MIPS`, `WINDOW_DURATION`,
  `MAX_CLOUDLETS_PER_WINDOW`, `INITIAL_WINDOW_START` no los lee Java (`GATEWAY_PORT` es lo único) ni
  Python (`JAVA_HOST`/`JAVA_PORT`).
- **Fix:** borrarlas de `docker-compose.yml` y `docker-compose.h5.yml`.

#### REPO-05 · `h4_capacity_plan` es código muerto
- **Estado:** propuesto · **P2**
- **Evidencia:** `precision.py:354-368` no tiene llamadores; el plan lo construye
  `main._build_native_plan`.

#### SIM-05 · `env.step()` sigue devolviendo reward 0.0
- **Estado:** propuesto (parcial) · **P2**
- **Evidencia:** `env.py:147` (`reward = 0.0`; el cálculo vive en `main.py`). El reloj ya se lee
  **después** del paso (`info["sim_time"]`, corregido en esta revisión).
- **Fix:** mover `RewardManager` al env o documentar el contrato en la docstring del módulo.

#### SIM-06 · El gateway todavía calcula el SLA y emite la 12.ª dimensión
- **Estado:** propuesto · **P2** (requiere reconstruir `java-sim`)
- **Evidencia:** `SimulationGateway.updateEnergyAndSla` mantiene `slaViolationCount` y
  `countedSlaCloudlets`, y `buildObservation()` sigue devolviendo
  `N_HOSTS*2 + 1 + 3` con `time_to_deadline` constante. Python ya ignora el contador y recorta a 11
  dims (`env._parse_obs`), pero el trabajo muerto sigue en Java y el cliente paga el valor extra.
- **Fix:** al reconstruir la imagen, borrar el contador SLA y emitir 11 dims; ajustar
  `SimulationGatewayRegressionTest` (hoy compara `getSlaViolationCount`).
- **Criterio:** la observación del gateway mide `OBS_DIM` y no existe `getSlaViolationCount`.

#### PERF-03 · La latencia de despliegue mide el padding del kernel, no solo el cómputo
- **Estado:** propuesto · **P2**
- **Evidencia:** `precision._int8_binary_matmul` rellena `M` a 17 filas (exigencia de `torch._int_mm`)
  en cada forward de batch=1; `benchmark_inference` mide ese camino completo.
- **Fix:** reportar por separado latencia con y sin padding (p. ej. backend Triton sin padding) o
  descontar el padding explícitamente en el análisis.

### P3 — opcional

#### PERF-02 · El notebook mide "tamaño" con `capas × neuronas × bits`
- **Estado:** propuesto · **P3**
- **Evidencia:** `notebooks/results_analisis.ipynb` usa `n_layers * hidden_size * bits`, que ignora
  que los parámetros crecen con `hidden²` dentro de cada capa oculta.
- **Fix:** usar las columnas persistidas `params`, `flops` o `effective_capacity_bits`.

---

## 4. HECHOS

| Ítem | Qué se hizo | Dónde | Evidencia |
|---|---|---|---|
| **CSV-01** reward normalizado | `best_mean_reward` es ahora la **media por episodio** de la mejor ventana (antes era la suma de 50); `norm_reward = best_mean_reward / avg_steps ∈ [-1,0]`; se añaden `eval_avg_steps` y `eval_norm_reward`; se eliminan `best_eval_reward`/`best_eval_episode` (duplicados) | `main.py`, `results.py` | smoke: `best_mean_reward=-9.29`, `avg_steps=20.25` → `norm_reward=-0.4588`; `eval_norm_reward=-0.4512` |
| **CSV-02** upsert | El CSV se lee, se reemplaza la fila con la misma `(experiment_id, seed)` y se reescribe (antes: `append` puro) | `results.py` | `tests/test_results.py` (3 tests) + smoke re-ejecutado dos veces → sigue habiendo **1** fila |
| **CSV-03** sin evaluación | `best_mean_reward`/`norm_reward` son `NaN` si el run nunca llegó a un punto de evaluación (antes `0.0`, que se leía como "mejor que −4597") | `main.py` | smoke con `n_episodes=1 < eval_every=5` → `NaN` en el CSV |
| **LOG-01** eje de `train/*` | Una sola serie en `train/loss` y `train/mean_q`, con x = `agent.total_steps` global; `mean_q` real (media de `max_a Q`) devuelta por `train_step`; el promedio por episodio vive en `episode/avg_loss` y `episode/mean_q` | `agent.py`, `metrics.py`, `main.py` | `EventAccumulator`: `train/loss` n=62 con pasos 1…62 monótonos y únicos; `train/mean_q` con varianza; `episode/mean_q` aparte |
| **LOG-02** pesos adaptativos | Eliminados `log_weights`, el bloque `delta_*`/`prev_*` y la API no-op del reward (upstream `284ac2b`/`f74001d` + esta revisión) | `metrics.py`, `main.py`, `reward.py` | no existen tags `weights/*` ni `update_weights`/`get_current_weights` |
| **LOG-03** telemetría de eval y despliegue | Al cerrar el run se registran `eval/mean_reward`, `eval/norm_reward`, `eval/makespan`, `eval/avg_steps` y `perf/latency_*`, `perf/throughput_pps`; se quitó `run/started` | `metrics.py`, `main.py` | `EventAccumulator` lista los 7 tags nuevos |
| **EVAL-02** H5 vs H4 | H5 pasa de `mi_scale: 0.1` a `simulation_time_scale: 0.01`, igual que H4 | `configs/h5_heuristics.yaml`, `docs/h5_heuristics.md` | ambos configs con el mismo valor; smoke H5 corre con esa escala |
| **EVAL-05** semillas | `experiment.seeds: [42, 41]` → 66 celdas × 2 = **132** corridas (3 semillas costaban +66 corridas) | `configs/h4_native_precision.yaml`, `tests/test_h4_plan.py` | guard del test en verde con 132 |
| **SIM-02** nº de hosts | `n_cloud_hosts` desaparece: `env.N_HOSTS = 4` y la topología queda fija como en el gateway; se quitó `datacenter` de `DEFAULT_CONFIG` y de H3/H4/H5 | `env.py`, `main.py`, configs | `tests/test_env_order.py::test_observation_drops_the_constant_deadline_feature` |
| **SIM-03** fuente única de potencia | Nuevo `d2ql/power.py` con el modelo (`idle`, `max`, `gamma`, precio, `e_step_max_wh`); `RewardManager` lo usa y ya no acepta argumentos muertos; `e_step_max` es inyectable | `power.py`, `reward.py` | `tests/test_power.py` compara los 4 arrays con `SimulationGateway.java`; `test_reward.py` verifica 0.2694 Wh |
| **SIM-04** el orden se aplica | La cola ahora **ordena de verdad**: LPT por trabajo por PE (`mi/num_pes`) y `env.reset` envía la ventana al gateway en ese orden (antes se calculaba y se ignoraba). Se eliminaron las claves `queue.*` | `queue.py`, `env.py`, configs | `tests/test_queue.py` (4) + `tests/test_env_order.py` (el stub registra el orden LPT de `addWorkloadRow`) |
| **SLA retirado** | Fuera del CSV/JSON/TB/docs: `eval_sla_violations`, `eval_sla_rate`, `info["sla_violations"]` y los argumentos `sla_*`/`cpu_*` del reward; notebooks y `analyze_pareto.py` sin esas columnas | `results.py`, `main.py`, `env.py`, `reward.py`, `analyze_pareto.py`, docs, notebooks | `tests/test_results.py` verifica que ninguna columna contiene "sla"; CSV del smoke con 31 columnas, ninguna SLA |
| **REPO-06** lint | `ruff --fix` sobre el repo + `NOQA` documentado en el alias `F` de PyTorch | `python-agent/`, `tests/` | `ruff check --no-cache .` → *All checks passed!* |
| **REPO-07** API no-op del reward | `RewardManager` expone solo `compute_step_reward` y `e_step_max` | `reward.py` | `tests/test_reward.py` (6 tests) |
| **REPO-08** doc de H5 | `h5_heuristics.md` ya no lista la columna inexistente `notes` ni métricas SLA | `docs/h5_heuristics.md` | columnas del doc = columnas de `h5_results.csv` |
| **PERF-01** eje de batch | `throughput_batch_sizes: [1, 8, 32, 128, 256, 512, 1024]` en H4 | `configs/h4_native_precision.yaml` | el CSV guarda la curva completa por corrida (smoke con `[1, 8]`) |
| **SIM-07** el gateway se apagaba al cerrar el env | `CloudSimEnv.close()` llamaba `JavaGateway.shutdown()`, que **mata el servidor**; la siguiente corrida fallaba con socket muerto. Ahora usa `gateway.close()` (solo el canal cliente) | `env.py` | tras 3 corridas seguidas el contenedor `java-sim` seguía `Up` |
| **REPO-03** guard del plan | El test del plan H4 se alinea con el config vigente (66 celdas × 2 semillas) | `tests/test_h4_plan.py` | 60 tests en verde |
| **H1/H2 retirados** | Configs H1/H2 y `d2ql/quantization.py` eliminados (la fusión con `284ac2b` los había limpiado pero no borrado) | `configs/`, `d2ql/` | `grep -r quantization` sin resultados de código |
| **Docs unificados** | H4, H6, R1 y R2 absorbidos aquí y eliminados | `docs/` | quedan `backlog.md` y `h5_heuristics.md` |

---

## 5. DESCARTADOS

| Ítem | Porqué | Sustituto |
|---|---|---|
| **SIM-01** SLA como eje | El deadline del trace es un timestamp absoluto (~1e6 s) frente a un reloj de ~170 s: `finish > deadline` nunca se cumple y el contador daba **0 en 132/132 filas**. Además la feature `time_to_deadline` era constante 1.0 | eje energía/tamaño/latencia; métrica eliminada del CSV |
| **EVAL-03** SLA contado en el último episodio | Deja de existir el contador: el SLA se retiró del proyecto (SIM-01) | — |
| Migración real de cloudlets | CloudSimPlus no la expone de forma simple y no aporta al objetivo energético actual | scheduling de colocación (nombre honesto en docs) |
| int8 como precisión estudiada | Sin VNNI en Zen 3 un GEMM int8 solo se emula: siempre más lento que fp32 | ternario (kernel bit-packed) en CPU |
| Triton como backend por defecto | Triton 3.2 corrompe `tl.dot` int8 con magnitudes ±127 | `int_mm` por defecto, Triton opt-in |
| PTQ (H2) | No captura el objetivo (entrenar desde cero a baja precisión) | H4 (entrenamiento nativo) |
| bf16 en CPU | Zen 3 sin AVX512-BF16 → emulación: artefacto del hardware | bf16 solo en GPU |

---

## 6. Decisiones abiertas

1. **Re-correr el sweep (EVAL-06):** ¿se regenera `data_h4.parquet` con el código actual (11 dims +
   LPT + 2 semillas) o se conserva como "resultados v1" etiquetado?
2. **Hold-out (EVAL-01):** ¿estratificado por MI o se asume el corte temporal como "otra carga"?
3. **Reconstruir `java-sim` (SIM-06):** limpiar el SLA y la 12.ª dimensión del gateway requiere
   rebuild de la imagen Java.
4. **Hardware objetivo del despliegue:** decide si el eje es `packed_size_mb` (memoria/edge) o
   latencia real (servidor), y si tiene sentido seguir escalando batch.
5. **Orden de despacho (SIM-04):** hoy es LPT por trabajo por PE; alternativas: SPT (minimiza el
   tiempo medio de finalización) o mantener el orden del trace como baseline explícito.
