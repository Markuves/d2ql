# Backlog de mejoras — d2ql

Documento único de mejoras, decisiones y deuda técnica del proyecto (DDQN sobre CloudSimPlus
para *load balancing* en la nube, eje **precisión × capacidad**).

Reemplaza a los históricos `h4_early_stopping_y_resultados.md`, `h6_energy_reward.md`,
`r1_recommendations.md` y `r2_decisiones_precision_capacidad.md` (eliminados). `h5_heuristics.md`
sigue siendo el documento operativo de ese experimento.

> Última revisión: **2026-10-24** (tercera tanda: auditoría de TensorBoard con dos corridas reales —
> 6 hallazgos nuevos, sin cambios de código todavía). Alcance: `python-agent/`, `java-sim/`,
> `configs/`, `tests/`, `docs/`, `notebooks/`, `docker-compose*.yml`, `pyproject.toml`.

---

## 1. Lineamientos

1. **Fuente única.** Toda sugerencia, decisión o deuda vive aquí; no se abren documentos por sprint.
   El documento está **ordenado por estado**: propuestos → hechos → descartados → decisiones.
2. **ID estable.** `ÁREA-nn` con áreas `LOG` (TensorBoard), `CSV` (resultados persistidos), `EVAL`
   (evaluación/comparabilidad), `SIM` (simulador + entorno), `REPO` (reproducibilidad/higiene),
   `PERF` (precisión y kernels). Nunca se reutiliza ni se renombra.
3. **Estados:** `propuesto` → `en curso` → `hecho` | `descartado` | `bloqueado`.
4. **Prioridades:** `P0` invalida cifras publicadas; `P1` comparabilidad/reproducibilidad/corrección
   latente; `P2` deuda sin efecto en las cifras; `P3` idea opcional.
5. **Definición de hecho:** código/config aplicado **y** evidencia reproducible (test, comando con
   salida, o fila de CSV). Sin evidencia el ítem no sale de "propuesto".
6. **Descartar es una decisión:** todo descartado lleva el porqué y el sustituto. Si se retoma, vuelve
   a `propuesto` con el mismo ID.
7. **Un ítem = un problema.** Si un cambio toca dos problemas, se abren dos ítems y se referencian.
8. **Procedencia obligatoria.** Toda fila persistida lleva `schema_version` + `git_sha` +
   `config_sha256`; sin eso no se puede afirmar que dos cifras sean comparables.

### Verificación

```bash
# tests (el torch del host no importa: falta libnccl)
docker run --rm -v $PWD:/work -w /work --entrypoint python d2ql-python-agent -m pytest tests/ -o addopts=""
# lint (--user para no dejar archivos root en el repo)
docker run --rm --user $(id -u):$(id -g) -v $PWD:/work -w /work --entrypoint ruff d2ql-python-agent check --no-cache .
# smoke end-to-end (gateway real + config mínima; ver commit 804c295 para el YAML)
docker network create d2ql-smoke
docker run -d --name java-sim --network d2ql-smoke -e GATEWAY_PORT=25333 -v $PWD/data:/app/data d2ql-java-sim:latest
docker run --rm --network d2ql-smoke --user $(id -u):$(id -g) \
  -e JAVA_HOST=java-sim -e JAVA_PORT=25333 -e D2QL_GIT_SHA=$(git rev-parse --short HEAD) \
  -v $PWD/python-agent/main.py:/app/main.py:ro -v $PWD/python-agent/d2ql:/app/d2ql:ro \
  -v $PWD/configs:/app/configs:ro -v $PWD/data:/app/data:ro -v $PWD/outputs:/app/outputs \
  d2ql-python-agent python main.py --config configs/<smoke>.yaml
```

---

## 2. Resumen

| Estado | P0 | P1 | P2 | P3 | Total |
|---|---|---|---|---|---|
| propuesto | 0 | 5 | 5 | 3 | 13 |
| hecho | 0 | 0 | 0 | 0 | 29 |
| descartado | 1 | 1 | 0 | 0 | 2 + 5 históricos |
| bloqueado | 0 | 0 | 0 | 0 | 0 (4 decisiones abiertas) |

Ya no queda ningún P0 abierto: EVAL-01 (hold-out) y EVAL-06 (procedencia/esquema) están resueltos.
Lo que falta para publicar cifras nuevas es **re-correr el sweep** con este código (decisión 1).

---

## 3. PROPUESTOS

### P1 — comparabilidad / reproducibilidad

#### EVAL-04 · `analyze_pareto.py` agrega sin la profundidad
- **Estado:** propuesto · **P1**
- **Evidencia:** `analyze_pareto.py:47` → `GROUP_KEYS = ["precision","device","hidden_size"]`, pero el
  sweep barre `n_hidden_layers: [1, 2]`; cada "celda" promedia modelos de 1 y 2 capas (capacidad y
  FLOPs distintos).
- **Fix:** incluir `n_hidden_layers` en `GROUP_KEYS`.
- **Criterio:** nº de celdas = precisiones × devices × anchos × profundidades.

#### LOG-04 · `episode/avg_loss` y `episode/mean_q` incluyen el warm-up
- **Estado:** propuesto · **P1**
- **Evidencia (auditoría TB, 6 episodios, `replay_buffer_warmup=10`):** el primer episodio registró
  `episode/avg_loss=0.0799` cuando la media real de sus 12 updates (`train/loss` en los pasos
  globales 1–12) es **0.1355** → el valor publicado es **0.59×** el real. Lo mismo en
  `episode/mean_q` (0.0290 vs 0.0521). En los episodios 2–6 la razón es 0.99–1.01.
  Causa: `episode_loss / step_count` y `episode_mean_q / step_count` dividen por **pasos de entorno**
  (22 en el episodio 1), pero solo hay 12 updates; los 10 pasos de warm-up aportan 0.
- **Impacto:** el primer punto de las curvas de pérdida/Q es ~1.7× más bajo que la realidad; cualquier
  episodio más corto que el warm-up se registraría como 0.
- **Fix:** dividir por el número de updates del episodio (contador que solo avanza cuando
  `train_step` actualiza) y, si es 0, no registrar el punto.
- **Criterio:** `episode/avg_loss` == media de `train/loss` del mismo rango de pasos (|Δ| < 1e-6).

#### LOG-05 · `episode/total_reward` es redundante con `episode/energy` y mide la duración
- **Estado:** propuesto · **P1**
- **Evidencia:** `reward == -energy / E_step_max` exacto en los 6 episodios (comprobado con
  `np.allclose`): los dos tags son la misma información reescalada. Además
  `corr(nº de pasos, total_reward) = -0.933`, con `reward/paso = -0.468 ± 0.013` (CV 2.8 %) y
  episodios de 18–22 pasos. Es decir: la curva de reward enseña sobre todo **cuánto duró el episodio**.
- **Fix:** registrar `episode/steps` y `episode/reward_per_step` (y `episode/energy_per_step`), dejando
  el total como acumulado; opcionalmente marcar `episode/energy` como derivada del reward.
- **Criterio:** `reward_per_step` tiene varianza propia y la correlación reward↔steps deja de dominar.

#### LOG-07 · El 86 % de la energía registrada es el piso idle
- **Estado:** propuesto · **P1**
- **Evidencia:** piso idle = `sum(HOST_IDLE_WATT)=390 W` → **0.10833 Wh/paso**; la energía medida por
  paso es 0.1213–0.1308 Wh, así que el **82.8–89.3 %** de `episode/energy` es una constante que no
  depende de la política. La parte dinámica (lo único que el agente puede mover) es
  **0.013–0.022 Wh/paso**. Coherente con SIM-08: el makespan es 0.6–1.2 % de la duración del episodio.
- **Impacto:** el reward discrimina poco entre políticas y su varianza está dominada por la duración
  del episodio, no por la colocación.
- **Fix:** registrar `episode/dynamic_energy` (= `energy − idle_floor × steps`) y `episode/steps`;
  usarlo como eje de calidad secundario.

#### SIM-08 · El episodio sigue vivo después del último cloudlet
- **Estado:** propuesto · **P1**
- **Evidencia:** con 10 cloudlets por episodio, el `makespan` es 0.12–0.27 s mientras el episodio dura
  18–22 s simulados: se despacha un cloudlet por paso (10 pasos) y el simulador sigue avanzando ~10 s
  más sin trabajo pendiente (VM/datacenter con eventos). La mitad de los pasos y el 86 % de la energía
  son piso idle.
- **Impacto:** coste de entrenamiento y ruido en el reward (LOG-07); el agente no puede influir en esa
  fracción.
- **Fix:** terminar el episodio cuando no queden cloudlets pendientes y todos hayan terminado (o no
  emitir eventos de VM al reconstruir), midiendo el efecto en `energy` y `makespan`.
- **Criterio:** duración del episodio ≈ `makespan` + margen pequeño y `reward/paso` estable.

### P2 — deuda

#### SIM-06 · El gateway todavía calcula el SLA y emite la 12.ª dimensión
- **Estado:** propuesto · **P2** (requiere reconstruir `java-sim`)
- **Evidencia:** `SimulationGateway.updateEnergyAndSla` mantiene `slaViolationCount` y
  `countedSlaCloudlets`; `buildObservation()` sigue devolviendo `N_HOSTS*2 + 1 + 3` con
  `time_to_deadline` constante; y `addWorkloadRow(submitted, deadline, mi, pes)` sigue arrastrando un
  `deadline` que ya nadie usa (el SLA se retiró y la cola ordena por trabajo/PE). Python ignora el
  contador y recorta a 11 dims (`env._parse_obs`), pero el trabajo muerto sigue en el JVM.
- **Fix:** al reconstruir la imagen, borrar el contador SLA, emitir 11 dims y quitar `deadline` del
  protocolo/records; ajustar `SimulationGatewayRegressionTest` (hoy compara `getSlaViolationCount`).
- **Criterio:** la observación del gateway mide `OBS_DIM` y no existe `getSlaViolationCount`.

#### EVAL-07 · El split temporal existe pero nadie lo mide
- **Estado:** propuesto · **P2**
- **Evidencia:** `workload.temporal_episodes()` conserva la cola cronológica como test de
  generalización temporal, pero ningún camino de `main.py` lo llama ni lo persiste.
- **Fix:** exponer `evaluation.split: stratified|temporal` en el config y, si se pide, persistir
  `eval_temporal_mean_reward`/`eval_temporal_makespan` como columnas aparte (nunca mezcladas con las
  del split estratificado).
- **Criterio:** un run con `split: temporal` escribe sus propias columnas y no sobrescribe las del
  split estratificado.

#### REPO-09 · `git_sha` es `"unknown"` dentro del contenedor
- **Estado:** propuesto · **P2**
- **Evidencia:** `Dockerfile.python` copia `main.py` y `d2ql/` pero no `.git`, así que
  `provenance.git_sha()` no puede ejecutar `git rev-parse`; en el smoke solo apareció
  `smoke123` porque se inyectó `D2QL_GIT_SHA` a mano.
- **Fix:** `ARG GIT_SHA` en el `Dockerfile.python` + `ENV D2QL_GIT_SHA=$GIT_SHA`, y en compose
  `build.args.GIT_SHA: ${GIT_SHA:-unknown}`.
- **Criterio:** un run dentro de `docker compose` escribe un SHA real sin variables manuales.

#### PERF-04 · El eval estratificado encarece la evaluación
- **Estado:** propuesto · **P2**
- **Evidencia:** el split anterior (cola cronológica) tenía MI medio 25; el estratificado, 221
  (EVAL-01). Cada episodio de evaluación tiene por tanto ~10× más trabajo y el eval de las 66 celdas ×
  2 semillas pasa a ser una fracción no trivial del coste del sweep.
- **Fix:** medir el tiempo de `_run_heldout_eval` y, si pesa, bajar `evaluation.n_episodes`
  (10 → 5) manteniendo la cobertura, o cachear las ventanas de evaluación por corrida.

#### LOG-06 · Se escriben escalares `NaN` cuando no hay eval o no hay benchmark
- **Estado:** propuesto · **P2**
- **Evidencia (auditoría TB, corrida con `evaluation.n_episodes=0` y `latency.benchmark: false`):** los
  8 tags de `eval/*` y `perf/*` se crean con **1 punto NaN** cada uno. En TensorBoard eso produce
  series vacías/saltos en vez de "sin dato".
- **Fix:** filtrar los valores no finitos en `MetricsLogger.log_eval`/`log_perf` (y en
  `log_episode_diagnostics`) y avisar por log de que la métrica no se midió.
- **Criterio:** un run sin evaluación ni benchmark no genera ningún tag `eval/*`/`perf/*` (o ninguno con
  valores no finitos).

### P3 — opcional

#### PERF-02 · El notebook mide "tamaño" con `capas × neuronas × bits`
- **Estado:** propuesto · **P3**
- **Evidencia:** `notebooks/results_analisis.ipynb` usa `n_layers * hidden_size * bits`, que ignora
  que los parámetros crecen con `hidden²` dentro de cada capa oculta.
- **Fix:** usar las columnas persistidas `params`, `flops` o `effective_capacity_bits`.

#### REPO-10 · `analyze_pareto.py` no es instalable como comando
- **Estado:** propuesto · **P3**
- **Evidencia:** vive en `python-agent/` (fuera del paquete), así que no hay entry point y hay que
  invocarlo con `python analyze_pareto.py` desde ese directorio.
- **Fix:** moverlo a `d2ql/pareto.py` y añadir `pareto = "d2ql.pareto:main"` en `[project.scripts]`.

#### LOG-08 · Faltan series para diagnosticar el entrenamiento
- **Estado:** propuesto · **P3**
- **Evidencia:** TB registra `train/loss` y `train/mean_q`, pero no `learning_rate`, norma de gradiente,
  TD-error medio, ni la **ventana de reward que decide el early stopping**; el `es_reason` solo queda en
  el CSV. Tampoco hay `episode/steps` (ver LOG-05).
- **Fix:** añadir `train/lr`, `train/grad_norm`, `train/td_error_abs_mean`, `eval/window_mean` (la
  métrica real del stopper) y texto del motivo de parada.
- **Criterio:** se puede explicar en TB por qué una corrida paró antes y con qué magnitud de gradiente.

---

## 4. HECHOS

| Ítem | Qué se hizo | Dónde | Evidencia |
|---|---|---|---|
| **EVAL-01** hold-out estratificado | El split de calidad pasa de "cola cronológica" (MI 10.7× menor) a una muestra sistemática sobre el orden por MI, con offset de medio bloque para no sesgar a la ventana más ligera; la cola temporal queda accesible vía `temporal_episodes()` | `workload.py` | log real: **MI medio train 221.4 vs eval 221.1 (0.14 %)**; `tests/test_holdout.py` (5 tests: ratio < 10 %, tamaños, disjunción, cola temporal) |
| **EVAL-06** esquema/procedencia | Cada fila lleva `schema_version` y se documenta que los parquet actuales son **v1** (12 dims, orden traza, SLA, suma de ventana) e incompatibles con el código actual; `analyze_pareto` etiqueta las filas por versión y avisa si hay mezcla | `provenance.py`, `analyze_pareto.py`, `notebooks/data/README.md` | smoke: `extra` con `schema_version: 2`; el análisis imprime `schema_version: [2]` o el aviso |
| **CSV-04** procedencia por corrida | `extra` incluye `git_sha`, `config_sha256` (hash del config efectivo), `run_timestamp_utc`, `obs_dim`, `dispatch_order`, `eval_split` | `provenance.py`, `main.py` | `tests/test_provenance.py` (5) + fila del smoke con `git_sha=smoke123`, `config_sha256=355f0725ddf33037`, `obs_dim=11` |
| **REPO-01** entry points | `train = "d2ql.train:main"`, borrado `build-deps` (módulo inexistente); el orquestador se movió a `d2ql/train.py` y `python-agent/main.py` queda como shim compatible | `pyproject.toml`, `d2ql/train.py`, `main.py` | `importlib.import_module("d2ql.train")` OK; `python main.py --config ...` corre el smoke |
| **REPO-02** Triton opcional | Extra `kernels = ["triton>=3.0"]`; el fallback ya no es silencioso: avisa con la instrucción de instalación | `pyproject.toml`, `precision.py` | `tomllib` lista los extras `['dev', 'kernels']`; ruff limpio |
| **REPO-04** env muertas | Borradas `DATASET_PATH`, `WORKLOAD_TIME_SCALE`, `WORKLOAD_MIPS`, `WINDOW_DURATION`, `MAX_CLOUDLETS_PER_WINDOW`, `INITIAL_WINDOW_START` (6 líneas) | `docker-compose.yml` | `docker compose config` válido para el compose base y el override H5; grep sin resultados |
| **REPO-05** código muerto | Eliminada `h4_capacity_plan` (sin llamadores; el plan lo arma `_build_native_plan`) | `precision.py` | `grep -r h4_capacity_plan` solo encuentra el `.pyc` viejo |
| **SIM-05** reward en el env | `CloudSimEnv` construye su propio `RewardManager` y `step()` devuelve `r_t ∈ [-1,0]`; `main.py` ya no calcula la recompensa (usa la devuelta) | `env.py`, `train.py` | `tests/test_env_order.py`: delta 0 → 0.0; delta = `E_step_max` → −1.0 |
| **PERF-03** coste del padding | Nuevo `benchmark_padding_overhead()` (batch=1 vs lote de 17 sin padding) + columna `padding_overhead_ms` y tag `perf/padding_overhead_ms` | `agent.py`, `results.py`, `train.py`, `metrics.py` | smoke CPU: `padding_overhead_ms=0.0195`; TB lista `perf/padding_overhead_ms` |
| **CSV-01** reward normalizado | `best_mean_reward` = media por episodio de la mejor ventana; `norm_reward` por paso en [-1,0]; nuevas `eval_avg_steps`/`eval_norm_reward`; fuera `best_eval_*` | `train.py`, `results.py` | smoke: `best_mean_reward=-9.29`, `avg_steps=20.25` → `norm_reward=-0.4588` |
| **CSV-02** upsert | El CSV se reescribe reemplazando la fila con la misma `(experiment_id, seed)` | `results.py` | `tests/test_results.py`; smoke re-ejecutado → 1 fila |
| **CSV-03** sin evaluación | `NaN` en vez de `0.0` cuando el run no llegó a evaluar | `train.py` | smoke `n_episodes=1 < eval_every=5` → `NaN` |
| **LOG-01** eje de `train/*` | Una serie por tag, x = `agent.total_steps`; `mean_q` real; promedios por episodio en `episode/avg_loss` y `episode/mean_q` | `agent.py`, `metrics.py` | TB: `train/loss` 62 puntos con pasos 1…62 únicos y monótonos |
| **LOG-02** pesos adaptativos | Fuera `log_weights`, `delta_*` y la API no-op del reward (upstream + esta revisión) | `metrics.py`, `train.py`, `reward.py` | sin tags `weights/*` |
| **LOG-03** telemetría de eval/despliegue | `eval/*` y `perf/*` al cerrar el run; sin `run/started` | `metrics.py` | TB del smoke lista 4 tags `eval/*` y 4 `perf/*` |
| **EVAL-02** H5 vs H4 | `simulation_time_scale: 0.01` en H5 | `configs/h5_heuristics.yaml` | smoke H5 corre con esa escala |
| **EVAL-05** semillas | `[42, 41]` → 132 corridas | `configs/h4_native_precision.yaml`, `tests/test_h4_plan.py` | guard en verde |
| **SIM-02** topología fija | `N_HOSTS = 4` en `env.py`; fuera `n_cloud_hosts` de configs y `DEFAULT_CONFIG` | `env.py`, `train.py`, configs | `tests/test_env_order.py` |
| **SIM-03** fuente única de potencia | `d2ql/power.py` + `RewardManager` que solo recibe `e_step_max` | `power.py`, `reward.py` | `tests/test_power.py` compara los 4 arrays con el Java |
| **SIM-04** orden aplicado | LPT por trabajo/PE y envío al gateway en ese orden | `queue.py`, `env.py` | `tests/test_queue.py` + `tests/test_env_order.py` |
| **SLA retirado** | Fuera del CSV/JSON/TB/docs/notebooks/reward | `results.py`, `env.py`, `reward.py`, `analyze_pareto.py`, docs | `tests/test_results.py`; CSV del smoke con 32 columnas sin SLA |
| **SIM-07** el gateway se apagaba | `close()` usa `gateway.close()` (solo canal cliente) en vez de `shutdown()` | `env.py` | 3 corridas seguidas con el contenedor `Up` |
| **REPO-06** lint | `ruff --fix` + excepciones documentadas | repo | `ruff check` → *All checks passed!* |
| **REPO-07** API no-op del reward | `RewardManager` expone solo `compute_step_reward` y `e_step_max` | `reward.py` | `tests/test_reward.py` |
| **REPO-08** doc de H5 | Sin columna `notes` ni métricas SLA | `docs/h5_heuristics.md` | columnas del doc = CSV |
| **PERF-01** eje de batch | `[1, 8, 32, 128, 256, 512, 1024]` | `configs/h4_native_precision.yaml` | curva guardada en `throughput_by_batch` |
| **REPO-03** guard del plan | Test alineado al config (66 × 2 = 132) | `tests/test_h4_plan.py` | 72 tests en verde |
| **H1/H2 retirados** | Configs y `d2ql/quantization.py` eliminados | `configs/`, `d2ql/` | sin referencias de código |
| **Docs unificados** | H4, H6, R1 y R2 absorbidos aquí | `docs/` | quedan `backlog.md` y `h5_heuristics.md` |

---

## 5. DESCARTADOS

| Ítem | Porqué | Sustituto |
|---|---|---|
| **SIM-01** SLA como eje | Deadline absoluto (~1e6 s) contra un reloj de ~170 s: el contador daba 0 en 132/132 filas | energía/tamaño/latencia; métrica eliminada del CSV |
| **EVAL-03** SLA del último episodio | El contador ya no existe | — |
| Migración real de cloudlets | CloudSimPlus no la expone de forma simple y no aporta al objetivo energético | scheduling de colocación |
| int8 como precisión estudiada | Sin VNNI en Zen 3, un GEMM int8 solo se emula | ternario bit-packed en CPU |
| Triton como backend por defecto | Triton 3.2 corrompe `tl.dot` int8 con ±127 | `int_mm` por defecto, Triton opt-in |
| PTQ (H2) | No captura "entrenar desde cero a baja precisión" | H4 (entrenamiento nativo) |
| bf16 en CPU | Zen 3 sin AVX512-BF16 → emulación | bf16 solo en GPU |

---

## 6. Decisiones abiertas

1. **Re-correr el sweep:** ¿se regenera `data_h4.parquet` como v2 (11 dims, LPT, split estratificado,
   132 celdas) o se conserva el snapshot v1 solo como histórico?
2. **Reconstruir `java-sim` (SIM-06):** limpiar el SLA y la 12.ª dimensión del gateway exige rebuild
   de la imagen Java (Maven).
3. **Hardware objetivo del despliegue:** decide si el eje es `packed_size_mb` (memoria/edge) o
   latencia real (servidor), y si tiene sentido seguir escalando batch.
4. **Orden de despacho (SIM-04):** hoy LPT por trabajo/PE; alternativas: SPT (minimiza el tiempo medio
   de finalización) o mantener el orden del trace como baseline explícito.
