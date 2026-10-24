# Backlog de mejoras — d2ql

Documento único de mejoras, decisiones y deuda técnica del proyecto (DDQN sobre CloudSimPlus
para *load balancing* en la nube, eje **precisión × capacidad**).

Reemplaza y absorbe los documentos históricos `h4_early_stopping_y_resultados.md`,
`h6_energy_reward.md`, `r1_recommendations.md` y `r2_decisiones_precision_capacidad.md`
(eliminados). `h5_heuristics.md` sigue siendo el documento operativo de ese experimento.

> Última revisión: **2026-10-19**. Alcance: `python-agent/`, `java-sim/`, `configs/`, `tests/`, `docs/`.

---

## 1. Lineamientos

1. **Fuente única.** Toda sugerencia, decisión o deuda vive aquí. No se crean documentos nuevos de
   cambios por sprint; se actualizan los ítems existentes.
2. **ID estable.** `ÁREA-nn`, áreas: `LOG` (TensorBoard), `CSV` (resultados persistidos), `EVAL`
   (evaluación/comparabilidad), `SIM` (simulador Java + entorno), `REPO` (higiene/reproducibilidad),
   `PERF` (precisión y kernels). El ID no se reutiliza ni se renombra.
3. **Estados:** `propuesto` → `en curso` → `hecho` | `descartado` | `bloqueado`.
4. **Prioridades:**
   - `P0` invalida o ensucia resultados ya publicados (hay que re-correr o re-analizar).
   - `P1` afecta comparabilidad, reproducibilidad o corrección latente.
   - `P2` deuda/limpieza sin efecto sobre las cifras.
   - `P3` idea opcional.
5. **Definición de hecho:** código/config aplicado **y** evidencia reproducible (comando + salida, o
   dato del CSV/parquet). Sin evidencia el ítem no se cierra.
6. **Descartar es una decisión, no un olvido:** todo `descartado` lleva el porqué y con qué se
   sustituye. Un ítem descartado que se retoma vuelve a `propuesto` con la misma ID.
7. **Un ítem = un problema.** Si un fix toca dos problemas, se abren dos ítems y se referencian.
8. **Cifras siempre con fuente.** Las afirmaciones numéricas de este documento salen de
   `notebooks/data/data_h4.parquet` (132 corridas H4) o de comandos listados en el anexo.

### Verificación rápida

```bash
# tests (el torch del host no importa: falta libnccl)
docker run --rm -v $PWD:/work -w /work --entrypoint python d2ql-python-agent -m pytest tests/ -o addopts=""
# lint
docker run --rm -v $PWD:/work -w /work --entrypoint ruff d2ql-python-agent check --no-cache .
# análisis del sweep
docker compose run --rm python-agent python analyze_pareto.py --results outputs/results/h4_results.csv
```

---

## 2. Resumen

| Estado | P0 | P1 | P2 | P3 | Total |
|---|---|---|---|---|---|
| hecho | 1 | 0 | 2 | 0 | 3 |
| propuesto | 4 | 8 | 5 | 2 | 19 |
| descartado | 1 | 0 | 0 | 0 | 1 |
| bloqueado | 0 | 0 | 0 | 0 | 0 (4 decisiones abiertas) |

Lo urgente antes de volver a publicar cifras: **CSV-01**, **EVAL-01** y **LOG-01**.
**EVAL-02** ya quedó corregido en esta revisión (H5 estaba 10× desalineado de H4).

---

## 3. P0 — invalida o ensucia los resultados

### CSV-01 · `norm_reward` es dimensionalmente inválido y `best_mean_reward` es una suma
- **Estado:** propuesto · **P0**
- **Evidencia:** `main.py:807-815` → `norm_reward = best_mean_reward / avg_steps`, exacto en las 132
  filas (máx. diferencia `7.1e-15`). `best_mean_reward` = `EarlyStopper.best_window_reward` = **suma**
  de la ventana de `eval_every=50` episodios (≈ −4597), no una media: equivale a
  `eval_every × reward medio por episodio` (≈ −91.9), mientras `eval_mean_reward` ≈ −31.5.
- **Impacto:** el eje de calidad del Pareto (`norm_reward`, −27.5 ± 0.6) está **50× escalado** y
  normalizado por los pasos de *un* episodio, no de la ventana. El ranking no cambia (todas las filas
  comparten factor), pero el valor absoluto y la interpretación ("reward por paso") son falsos.
- **Fix:** guardar `best_window_mean_reward = window_sum / eval_every`; calcular
  `norm_reward = best_window_mean_reward / avg_steps` (o, mejor, `eval_mean_reward / eval_steps`);
  renombrar la columna a `best_window_mean_reward`.
- **Criterio:** `norm_reward ∈ [-1, 0]` (el reward por paso está acotado así por construcción) y
  reproducible a mano desde `eval_mean_reward` y los pasos del eval.

### EVAL-01 · El hold-out cronológico es 10.7× más ligero que el entrenamiento
- **Estado:** propuesto · **P0**
- **Evidencia:** `workload.py:109-120` reserva el 20 % final de ventanas. Sobre
  `data/workload.csv.gz` con `episode_length=50`: **train 40 817 ventanas, MI medio 270** (p95 1685);
  **test 10 204 ventanas, MI medio 25** (p95 129) → ratio **10.7×**. Coincide con los resultados:
  `eval_makespan` ≈ 0.59 s de media frente a episodios de entrenamiento de ~166 pasos.
- **Impacto:** `eval_*` mide otra carga (mucho más ligera), así que no es comparable con ninguna
  métrica de entrenamiento — ni con `best_mean_reward` (EVAL/CSV-01) ni con el reward de la curva de
  TB. "Held-out" es correcto como test temporal, incorrecto como estimación del régimen entrenado.
- **Fix:** hold-out **estratificado por MI** (o k-fold con la misma distribución) para la métrica de
  calidad, y conservar el corte temporal como test de generalización aparte.
- **Criterio:** `|MI_train − MI_eval| / MI_train < 10 %` y `eval_makespan` del mismo orden que el
  makespan de entrenamiento.

### LOG-01 · `train/loss` mezcla tres ejes x y `train/mean_q` es constante
- **Estado:** propuesto · **P0** (telemetría no interpretable)
- **Evidencia:** `main.py:633-637` escribe `train/*` con `x = max(agent.total_steps, step_count)` y
  `step_count` **se reinicia en cada episodio**; `main.py:698-702` vuelve a escribir el mismo tag con
  `x = episode+1`; `main.py:612,670` — `episode_mean_q` nunca se acumula, siempre `0.0`.
- **Impacto:** el panel de loss es un diente de sierra con una rama por episodio más una rama por
  episodio-index; `train/mean_q` no aporta información. No se puede diagnosticar convergencia.
- **Fix:** un único contador global (`agent.total_steps`) como x; borrar la re-escritura por episodio
  (o moverla a `episode/avg_loss`); devolver desde `train_step()` `q_values.max(dim=1).values.mean()`
  y promediarlo por episodio.
- **Criterio:** x monótona y sin duplicados, un punto por update, `train/mean_q` con varianza > 0.

### LOG-02 · `weights/*` registra constantes de un experimento retirado (H1)
- **Estado:** propuesto · **P0** (paneles engañosos)
- **Evidencia:** `main.py:659-663,692-697` calcula deltas y llama `update_weights` (no-op desde el
  reward energy-only) y `metrics.log_weights` (`metrics.py:44-55`) escribe `w_sla=0, w_energy=1,
  w_cost=0` en cada episodio.
- **Impacto:** tres curvas planas que sugieren un mecanismo adaptativo que ya no existe.
- **Fix:** eliminar `log_weights`, el bloque `delta_perf/delta_energy/delta_cost`,
  `RewardManager.update_weights` y `get_current_weights` (y su test), o dejarlos con un
  `logger.debug`.
- **Criterio:** no quedan tags `weights/*` en un run nuevo.

### EVAL-02 · H5 evaluaba con un `simulation_time_scale` 10× distinto al de H4
- **Estado:** **hecho** (2026-10-19) · **P0**
- **Evidencia:** `configs/h4_native_precision.yaml` usa `simulation_time_scale: 0.01` (y las 132 filas
  del parquet registran ese valor), mientras `configs/h5_heuristics.yaml` usaba el alias
  `mi_scale: 0.1`, con un comentario que afirmaba que H4 valía 0.1. El MI escala el tiempo de
  ejecución 1:1 (`workload.py:70-86`), así que energía, makespan y reward se multiplicaban por 10.
- **Fix aplicado:** H5 pasa a `simulation_time_scale: 0.01` y se corrige el comentario y
  `docs/h5_heuristics.md`.
- **Criterio:** ambos configs con el mismo valor. **Consecuencia:** cualquier `h5_results.csv`
  generado antes de este cambio **no es comparable** con las filas DDQN y debe re-ejecutarse.

### SIM-01 · El SLA vale 0 por construcción (deadline absoluto del trace)
- **Estado:** **descartado** por decisión (eje SLA fuera de alcance) — documentado · **P0 si se retoma**
- **Evidencia:** `SimulationGateway.java:356-368` compara `cloudlet.getFinishTime()` (reloj de
  simulación, 0…~170 s) contra `deadline` del trace (absoluto, p. ej. 1 673 700 s). La condición
  `finish > deadline` nunca se cumple: **132/132 filas con `eval_sla_rate = 0`**. El mismo deadline
  absoluto hace que la 12.ª dimensión de la observación (`time_to_deadline`, `java:392`) sea
  **siempre 1.0** (`(deadline − 0)/3600 ≥ 464`, luego `clip` a 1).
- **Porqué se descartó:** el eje de comparación actual es energía/tamaño/latencia; arreglar el SLA
  exige re-basar los deadlines y cambiaría también la observación y el reward.
- **Fix si se retoma:** `deadline_rel = deadline − min(submitted_at de la ventana)`, escalado por
  `simulation_time_scale`; usar ese valor en `addWorkloadRow` y `createCloudlet`.
- **Criterio:** ≥1 política con `sla_rate > 0`, varianza entre políticas, y la 12.ª feature deja de ser
  constante.

---

## 4. P1 — comparabilidad, reproducibilidad, corrección latente

### CSV-02 · Re-ejecutar el sweep duplica filas en `h4_results.csv`
- **Estado:** propuesto · **P1**
- **Evidencia:** `results.py:226-238` abre el CSV en modo `"a"` sin deduplicar. `experiment_id`
  incluye precisión/device/ancho/profundidad/semilla, así que una segunda corrida inserta filas con la
  misma clave; cualquier agregación posterior cuenta doble.
- **Fix:** *upsert* por `(experiment_id, seed)` (leer, reemplazar, reescribir) o escribir un CSV por
  corrida y agregar en el análisis. Añadir `run_timestamp` a la fila.
- **Criterio:** correr dos veces el mismo plan deja el mismo número de filas.

### CSV-03 · `best_eval_reward = 0.0` cuando el run nunca evaluó
- **Estado:** propuesto · **P1**
- **Evidencia:** `main.py:830-832` sustituye `-inf` por `0.0`, y `main.py:813-814` hace lo mismo con
  `best_mean_reward`. Con `n_episodes < eval_every` (o `n_episodes=0`) el CSV reporta 0, que se lee
  como "mejor que −4597" en cualquier ordenamiento.
- **Fix:** persistir `NaN` (o `null`) y documentarlo; nunca 0 para "sin dato".
- **Criterio:** un run sin puntos de evaluación produce `NaN`, no 0.

### EVAL-03 · `eval_sla_violations` cuenta solo el último episodio
- **Estado:** propuesto · **P1** (latente mientras SIM-01 no se arregle)
- **Evidencia:** `rebuildSimulation()` resetea `slaViolationCount` en cada `resetEpisode()`
  (`java:205`), pero `_run_heldout_eval` (`main.py:140`) lee el contador **una vez, al final** de los
  10 episodios, y luego divide por los cloudlets de los 10 (`main.py:143`).
- **Fix:** acumular `env.sim.getSlaViolationCount()` al cerrar cada episodio.
- **Criterio:** `eval_sla_violations == Σ episodios` y `sla_rate` estable al cambiar `n_episodes`.

### EVAL-04 · `analyze_pareto.py` agrega sin la profundidad
- **Estado:** propuesto · **P1**
- **Evidencia:** `analyze_pareto.py:43` → `GROUP_KEYS = ["precision","device","hidden_size"]`, pero el
  sweep barre `n_hidden_layers: [1, 2]`. Cada "celda" promedia modelos de 1 y 2 capas (capacidad y
  FLOPs distintos) y el frente compara medias de poblaciones mezcladas.
- **Fix:** incluir `n_hidden_layers` en `GROUP_KEYS`.
- **Criterio:** nº de celdas = precisiones × devices × anchos × profundidades.

### SIM-02 · `n_cloud_hosts` es configurable en Python y fijo en Java
- **Estado:** propuesto · **P1**
- **Evidencia:** `SimulationGateway.java:56-57` (`NUM_HOSTS = 4`, `NUM_VMS = 4`) contra
  `env.py:17-29` (acción y observación derivadas de `config["datacenter"]["n_cloud_hosts"]`) y
  `env.py:170-174` (la observación se recorta/rellena **en silencio** si no coincide).
- **Impacto:** con cualquier valor ≠ 4 el agente recibe una observación y un espacio de acciones que
  no corresponden al simulador, sin error.
- **Fix:** pasar el número de hosts al gateway en `resetEpisode()` o validar `n_cloud_hosts == 4` al
  construir el env.
- **Criterio:** error explícito (o soporte real) si el config difiere de 4.

### SIM-03 · `E_step_max` del reward ignora la config y el nº de hosts
- **Estado:** propuesto · **P1**
- **Evidencia:** `reward.py:11-13` fija `host_max_watts = [250, 200, 300, 220]`; la clave
  `e_step_max_wh` que existía en los configs **no se leía en ningún sitio** (verificado por grep);
  los argumentos `e_ref` y `v_max` del constructor no se usan.
- **Impacto:** si cambia el modelo de potencia (`HOST_MAX_WATT`) o el número de hosts, el reward deja
  de estar en `[-1, 0]` y las corridas no son comparables entre sí.
- **Fix:** una sola fuente de verdad (leer `e_step_max_wh` del config, o derivarlo de los parámetros
  del gateway) y eliminar los argumentos muertos.
- **Criterio:** un test que cambie el nº de hosts/potencias y vea cambiar `E_step_max`.

### SIM-04 · La cola de prioridad no influye en el despacho
- **Estado:** propuesto · **P1**
- **Evidencia:** `env.py:85-92` encola cada cloudlet y `env.py:114` recalcula prioridades, pero **nunca
  se hace `pop()`/`drain()`**; el gateway despacha siempre `cloudlets[currentCloudletIndex++]` en el
  orden del trace (`java:249-255`).
- **Impacto:** `queue.py` (146 líneas) y las claves `queue.urgency_weight/demand_weight` son
  decorativas; la "urgencia" no afecta a nada y el agente solo elige host para el siguiente cloudlet
  del trace. Además `urgency = elapsed/time_remaining` es ≈0 con deadlines absolutos (SIM-01).
- **Fix:** o se aplica el orden (enviar los ids ordenados al gateway) o se borra el módulo y las
  claves. Decisión pendiente (§7).
- **Criterio:** el orden de despacho cambia al cambiar `queue.*`, o no existe el módulo.

### LOG-03 · No se registra nada de la evaluación ni del despliegue
- **Estado:** propuesto · **P1**
- **Evidencia:** `metrics.py` no tiene un `log_eval`; `_run_heldout_eval` devuelve las métricas y solo
  van al JSON/CSV. El único tag extra es `run/started = 1.0` (`metrics.py:15`), inútil.
- **Impacto:** para comparar dos corridas hay que abrir el CSV; TB no responde "¿mejoró el held-out?".
- **Fix:** `log_scalars` al final del run con `eval/*` y `perf/*` (latencia, throughput), y eliminar
  `run/started`.
- **Criterio:** en TB aparecen `eval/mean_reward`, `eval/makespan`, `perf/latency_mean_ms`.

---

## 5. P2 — deuda y limpieza

### REPO-01 · Entry points del paquete rotos
- **Estado:** propuesto · **P2**
- **Evidencia:** `pyproject.toml` → `train = "main:main"` (el wheel solo empaqueta `python-agent/d2ql`,
  `main` no queda instalado) y `build-deps = "scripts.build_deps:main"` (ese módulo **no existe**).
- **Fix:** borrar `build-deps`; para `train`, mover el orquestador a `d2ql.cli:main` o declarar
  `[tool.hatch.build.targets.wheel] only-include` correspondiente.
- **Criterio:** `uv run train --help` funciona tras `uv sync`.

### REPO-02 · `triton` se importa pero no se declara
- **Estado:** propuesto · **P2**
- **Evidencia:** `kernels.py:27` (`import triton`) en el top-level; no está en `dependencies` ni en un
  extra. El `try/except` de `precision.py:170-175` convierte el fallo en un *fallback* silencioso.
- **Fix:** extra opcional (`[project.optional-dependencies] triton = ["triton>=3.0"]`) o import
  perezoso dentro de `lowbit_matmul_triton`.
- **Criterio:** `D2QL_LOWBIT_BACKEND=triton` falla con un mensaje claro si falta la dependencia.

### REPO-03 · Suite de tests en rojo por un guard obsoleto
- **Estado:** **hecho** (2026-10-19) · **P2**
- **Evidencia:** `tests/test_h4_plan.py` afirmaba `seeds == [42, 43, 44]` y 138 corridas; el config
  real usa `[42, 41]` → 132 (que es lo que contiene el parquet). Antes: `1 failed, 41 passed`.
- **Fix aplicado:** el guard ahora refleja `[42, 41]` y `132`; `pytest` queda en **42 passed**.
- **Criterio:** `pytest` verde.

### REPO-04 · Claves de configuración que nadie lee
- **Estado:** **parcialmente hecho** (2026-10-19) · **P2**
- **Evidencia:** `n_edge_nodes`, `target_utilization`, `migration_penalty`, `weight_lr`, `w_*_floor`,
  `w_*_init` solo existían en `DEFAULT_CONFIG`/configs; `e_step_max_wh` tampoco se leía. En
  `docker-compose*.yml`, `DATASET_PATH`, `WORKLOAD_TIME_SCALE`, `WORKLOAD_MIPS`, `WINDOW_DURATION`,
  `MAX_CLOUDLETS_PER_WINDOW` e `INITIAL_WINDOW_START` no los lee ni Java (`GATEWAY_PORT` es lo único)
  ni Python (`JAVA_HOST`/`JAVA_PORT`).
- **Fix aplicado:** `DEFAULT_CONFIG` y los bloques `reward:` de H3/H4/H5 limpiados; se borraron los
  configs H1/H2.
- **Pendiente:** limpiar las variables de entorno muertas de `docker-compose.yml`/`h5.yml`.

### REPO-05 · `h4_capacity_plan` es código muerto
- **Estado:** propuesto · **P2**
- **Evidencia:** `precision.py:353-367` no tiene llamadores: el plan lo construye
  `main._build_native_plan`.
- **Fix:** borrarlo (o hacer que `main` lo use, pero hoy no soporta `device`).
- **Criterio:** `grep -r h4_capacity_plan` sin resultados.

### REPO-06 · Lint: 62 errores de ruff
- **Estado:** propuesto · **P2**
- **Evidencia:** `ruff check --no-cache .` → `Found 62 errors` (42 autofixables), casi todos `I001`
  (orden de imports en `tests/`) y `N` (nombres).
- **Fix:** `ruff check --fix .` y revisar los 20 restantes.
- **Criterio:** `ruff check` limpio y añadido al flujo de CI local.

### SIM-05 · `env.step()` devuelve reward 0 y `info["sim_time"]` es pre-step
- **Estado:** propuesto · **P2**
- **Evidencia:** `env.py:146` (`reward = 0.0`, el cálculo vive en `main.py`); `env.py:113` lee el reloj
  **antes** de `sim.step()` y lo publica en `info["sim_time"]`.
- **Impacto:** rompe la API de Gymnasium y cualquiera que use el env fuera de `main.py` no recibe
  señal; el tiempo reportado va un paso atrás.
- **Fix:** mover `RewardManager` al env (o documentar el contrato); leer el reloj post-step.
- **Criterio:** `env.step()` devuelve un reward no nulo y `info["sim_time"]` = reloj tras el paso.

---

## 6. P3 — ideas opcionales

### PERF-01 · Extender el eje de batch (256/512/1024)
- **Estado:** propuesto · **P3**
- **Contexto:** la ventaja de un kernel de bajo bit solo puede aparecer a batch grande; con
  `throughput_batch_sizes: [1, 8, 32, 128]` y `hidden ≤ 4096` no hay cruce con fp32 (ver Hecho · B1).
- **Criterio:** la curva `throughput_by_batch` incluye el punto de cruce, si existe.

### PERF-02 · El notebook mide "tamaño" con `capas × neuronas × bits`
- **Estado:** propuesto · **P3**
- **Evidencia:** `notebooks/results_analisis.ipynb` usa `n_layers * hidden_size * bits`, que ignora
  que los parámetros crecen con `hidden²` dentro de cada capa oculta.
- **Fix:** usar las columnas ya persistidas `params`, `flops` o `effective_capacity_bits`.
- **Criterio:** la métrica de tamaño usada en el análisis es una de las columnas del CSV.

---

## 7. Hecho (histórico unificado)

| Ítem | Qué se hizo | Dónde | Resultado / evidencia |
|---|---|---|---|
| A1 energía delta | El reward usa `E_t − E_{t−1}`, no el acumulado | `env.py`, `reward.py`, `main.py` | acotado en `[-1,0]`; test `test_energy_only_*` |
| A2 SLA sin doble conteo | `Set` de cloudlets ya contabilizados | `SimulationGateway.java` | 1 conteo por cloudlet (aunque el eje SLA siga en 0, SIM-01) |
| A3 "migración" | Plomería eliminada; se documentó que **no hay** migración | `java-sim`, `reward.py` | el agente es *scheduler* de colocación inicial (decisión, no bug) |
| B1 kernel real de bajo bit | `torch._int_mm` (padding M/K/N) + kernels propios en Triton + kernel ternario bit-packed (XOR+popcount) | `precision.py`, `kernels.py` | exacto vs referencia; **no acelera** a esta escala (`h≤128`: ~22× detrás de fp32); la ventaja es tamaño/memoria |
| B2 coste normalizado | `flops`, `effective_capacity_bits`, `packed_size_mb` en el CSV | `precision.py`, `results.py` | Pareto sobre ejes invariantes al ancho de bits |
| C1 evaluación held-out | Evaluación greedy sobre ventanas no vistas + métricas de negocio | `workload.py`, `main.py` | hecho, pero el hold-out es 10.7× más ligero (EVAL-01) |
| C2 reward normalizado | `avg_steps`, `norm_reward` | `main.py`, `results.py` | **cálculo inválido** → ver CSV-01 |
| C3 coste ≠ energía | Precio por watt distinto por host | `SimulationGateway.java` | coste descorrelacionado de la energía |
| D1 observación enriquecida | 12 dims: cpu×N, ram×N, cola, MI, PEs, deadline | `java-sim`, `env.py` | la feature de deadline es constante por SIM-01 |
| D2 VM→host | `VmAllocationPolicySimple` con mapeo fijo `vm i → host i` | `SimulationGateway.java` | cubierto por `SimulationGatewayRegressionTest` |
| D3 determinismo Java | Semilla en el gateway + mapeo fijo | `SimulationGateway.java` | `setSeed` por episodio; CloudSimPlus 8.5.7 no expone constructor con semilla (salvedad documentada) |
| D4 profundidad + escala de tiempo | `n_hidden_layers` vector `[1,2]` y `simulation_time_scale` (con `mi_scale` como alias) | `main.py`, `workload.py`, configs | 66 combinaciones × 2 semillas = 132 corridas |
| E1 analizador del Pareto | `analyze_pareto.py`: agregado por celda + frente no dominado + curva throughput-batch | `python-agent/analyze_pareto.py` | funciona; agrupa sin profundidad (EVAL-04) |
| E2 reward dentro del env | **No hecho** | — | ver SIM-05 |
| E3 presupuesto de cómputo | Early stopping por plateau | `early_stopping.py`, `main.py` | 105/132 corridas pararon por meseta; 27 agotaron el presupuesto |
| H4 early stopping + resultados | `EarlyStopper` + `result.json` + CSV agregado + benchmark de latencia/throughput | `early_stopping.py`, `results.py`, `agent.py` | 132 filas con latencia, throughput por batch, held-out |
| H6 reward energía + potencia convexa | `P = idle + (max−idle)·u^γ`, reward `-ΔE/E_step_max` | `java-sim`, `reward.py` | objetivo único, sin pesos que ajustar |
| H1 retirado | Config H1, pesos adaptativos (`w_*`, floors, `weight_lr`) y sus claves fuera de código y configs | `configs/`, `main.py`, `reward.py` | 2026-10-19: borrado `configs/h1_adaptive_baseline.yaml` |
| H2 retirado | PTQ (INT8 dinámico/FP16) eliminado: config + módulo standalone con `QNetwork` duplicada y API deprecada | `configs/h2_quantization.yaml`, `d2ql/quantization.py` | 2026-10-19: ambos borrados; H2 quedó superado por H4 |
| H5 alineado | `simulation_time_scale` igual a H4 | `configs/h5_heuristics.yaml` | 2026-10-19: 0.1 → 0.01 (EVAL-02) |
| Docs unificados | H4, H6, R1 y R2 absorbidos aquí | `docs/` | quedan `backlog.md` y `h5_heuristics.md` |

---

## 8. Descartado (con porqué)

| Ítem | Porqué | Sustituto |
|---|---|---|
| SLA como eje de comparación | Estructuralmente 0 (SIM-01); re-basar deadlines cambiaría obs y reward | energía / tamaño / latencia |
| Migración real de cloudlets | CloudSimPlus no la expone de forma simple aquí; el costo/beneficio no cierra para el objetivo actual | scheduling de colocación (nombre honesto en docs) |
| int8 como precisión estudiada | Sin VNNI en Zen 3 un GEMM int8 solo se emula: siempre más lento que fp32 | ternario (kernel bit-packed) en CPU; int8 queda solo como representación interna de `_int_mm` en CUDA |
| Triton como backend por defecto | Triton 3.2 `tl.dot` int8 corrompe con magnitudes ±127; 2-8 % sobre `int_mm` a costa de robustez | `int_mm` por defecto, Triton opt-in (`D2QL_LOWBIT_BACKEND=triton`) |
| PTQ (H2) | No captura el objetivo (entrenar desde cero a baja precisión) y su módulo estaba desfasado | H4 (entrenamiento nativo) |
| bf16 en CPU | Zen 3 sin AVX512-BF16 → emulación; el resultado sería un artefacto del hardware | bf16 solo en GPU |

---

## 9. Decisiones abiertas (bloquean ítems)

1. **¿Se retoma el eje SLA?** Si sí, priorizar SIM-01 (re-basar deadlines) y re-ejecutar el sweep.
2. **¿Cola de prioridad real o eliminar?** SIM-04: implementar despacho por prioridad en Java o borrar
   `queue.py` y las claves `queue.*`.
3. **¿Hold-out estratificado (EVAL-01) o se asume el corte temporal como "otra carga"?** Determina si
   las cifras de calidad actuales sirven o hay que re-correr.
4. **Hardware objetivo del despliegue** (edge/memoria vs servidor/velocidad): decide si el eje de
   comparación es `packed_size_mb` o latencia real, y si tiene sentido PERF-01.
