# Sugerencias de mejora — proyecto d2ql (CloudSim + DDQL load balancing)

## Estado de implementación (actualizado)

| ítem | Estado |
|------|--------|
| A1 — energía delta por paso | ✔ hecho (reward.py, env.py, main.py) |
| A2 — SLA contado una vez por cloudlet | ✔ hecho (SimulationGateway.java) |
| A3 — renombrar migración muerta (solo nombre) | ✔ hecho (penalty y plumbing eliminados; no se implementó migración real) |
| B1 — kernel real de bajo bit (opción 1) | ✔ hecho. Añadido también **throughput por batch** como métrica y **kernels propios (Triton)** testados contra `int_mm`. Ver Nota B1. |
| B2 — normalizar capacidad (FLOPs y capacidad efectiva) | ✔ hecho (precision.py, agent.py, results.py) |
| C1 — evaluación held-out + métricas de negocio | ✔ hecho (main.py, workload.py, config h4) |
| C2 — recompensa normalizada por pasos (`avg_steps`, `norm_reward`) | ✔ hecho (main.py, results.py) |
| C3 — coste descorrelacionado de energía | ✔ hecho (SimulationGateway.java) |
| Métrica adicional: **throughput por batch** | ✔ hecho (`agent.benchmark_throughput`, campos `throughput_pps`/`throughput_batch_size` en results; config `latency.throughput_batch_size`) |
| Experimentación de **kernels propios** (≤5 configs) | ✔ hecho (`d2ql/kernels.py` + `scripts/benchmark_kernels.py`) |

**Nota B1 (lectura honesta).** El kernel real int8 está implementado (vía `torch._int_mm`
default, con padding M/K/N) y es correcto (error máx ~0.002). Medido en esta GPU de consumo,
a batch=1 el camino int8 es más lento que fp32 (0.75 ms vs 0.10 ms); `torch._int_mm` no está
afinado para matrices pequeñas y el padding infla la carga. Implicación: la precisión rejilla
(4/8/ternary) NO reportará menor latencia que fp32 a muestra única; fp16 sí (nativo).

**Exploración de kernels propios (resultado).** Se escribió un kernel Triton int8 (acepta
cualquier `M`, sin padding de batch; BLOCK_K forzado a K por un bug de acumulación multi-K de
`tl.dot` int8 en Triton 3.2) y se barrió 5 configs de tile/warps contra `int_mm` y `fp32` en la
red de 3 capas (`scripts/benchmark_kernels.py`):

- El kernel Triton es **marginalmente más rápido que `int_mm`** en todo batch (~2-8%);
- pero ambos son **~6x más lentos que fp32** a cualquier batch en este hardware/escala
  (fp32 cuBLAS gana por amplio margen incluso a batch 512);
- además, Triton 3.2 `tl.dot` int8 corrompe con **valores int8 grandes** (pesos cuantizados
  reales ±127), por lo que `int_mm` sigue siendo el backend **robusto** por defecto.

Conclusión operativa: el throughput por batch ya se mide e integra (`throughput_pps`), pero en
este hardware la baja precisión no compensa; el backend por defecto es `int_mm` y se puede forzar
Triton experimental con `D2QL_LOWBIT_BACKEND=triton` (solo con activaciones pequeñas).

**Diseño final del experimento (sweep activo, `configs/h4_native_precision.yaml`).** Se fijó un
conjunto de **4 precisiones tipadas por device**, con exploración de capacidad para cada una y
reporte de latencia de entrenamiento, latencia de inferencia (single-sample + batch) y reward
held-out:

| Precisión | device | hidden_sizes (cap) |
|-----------|--------|--------------------|
| fp32 (`32`) | **cuda** | 256·512·1024·2048 |
| fp16 (`16`) | **cuda** | 256·512·1024·2048·4096 |
| fp32 (`32`) | **cpu** | 256·512·1024·2048 |
| int8 (`8`)  | **cpu** | 256·512·1024·2048·4096 |

18 runs en total; cada uno con tag `{precision}_{device}_h{hidden}`, entrenamiento nativo de baja
precisión (STE, no post-cuantización), device respetado por el agente (`agent.device`), y métricas
en el CSV: `wall_clock_s` (entrenamiento), `latency_*` + `throughput_pps` (inferencia) y `eval_*`
reward/SLA/makespan. int4/ternary quedan retirados del sweep (código conservado en `precision.py`).

**Contexto.** El proyecto usa CloudSimPlus (vía Py4J) para simular un centro de datos cloud y
una red DDQL (Double DQN) que aprende a hacer *load balancing*. La meta explorada es buscar el
**punto óptimo de combinación entre capacidad y precisión** del modelo (sweep H4) midiendo
calidad (reward) frente a coste/eficiencia (latencia, tamaño empaquetado).

Este documento recoge sugerencias concretas ordenadas por **severidad**: primero los bugs que hoy
ensucian (o invalidan) la métrica de comparación, luego el gap conceptual de la premisa
"precisión = velocidad", después la evaluación/comparabilidad, fidelidad de la simulación y
finalmente ingeniería/cierre.

---

## A. Bugs que sesgan la métrica de comparación

### A1. La recompensa de energía usa el valor ACUMULADO, no el delta por paso
- Archivos: `reward.py:42`, `env.py:112`, `main.py:258`
- **Problema.** `compute_step_reward()` espera el incremento de energía del paso
  `(E_t - E_{t-1}) / E_ref` (según el propio comentario), pero se le pasa
  `getTotalEnergyConsumed()`, que es el **total acumulado** desde el inicio del episodio.
- **Consecuencia.** `phi_energy = -cumulativo` genera un drift negativo sistemático que crece
  con cada paso. El reward deja de medir "qué tan bien balancea" y mide "cuánto pudo correr el
  simulador": ruido que corrompe la comparación entre runs de distinta duración.
- **Fix.** Llevar la energía previa en `env` (o en `main`) y pasar
  `energia_actual - energia_anterior`.

### A2. Las violaciones de SLA se cuentan múltiples veces por cloudlet
- Archivo: `SimulationGateway.java:277-295` (`updateEnergyAndSla`)
- **Problema.** Cada *step* itera **todos** los cloudlets y suma 1 si `finishTime > deadline`.
  Un cloudlet moroso se re-cuenta en cada step posterior (los cloudlets nunca se eliminan de la
  lista). El contador crece superlinealmente y desborda la recompensa.
- **Consecuencia.** Recompensa muy negativa y ruidosa, no comparable entre episodios; domina el
  reward total y enmascara el efecto real del scheduler.
- **Fix.** Llevar un `Set` de cloudlets ya contabilizados (contar una única vez), o contar solo en
  el instante del finish.

### A3. La "migración" es código muerto → el load balancing es solo scheduling inicial
- Archivos: `SimulationGateway.java` (`didMigrateLastStep`), `reward.py:61-63`
- **Problema.** `didMigrate()` siempre devuelve `false` (nadie lo pone a `true`), por lo que la
  penalización por migración nunca se aplica. El agente **no rebalancea tareas ya en ejecución**;
  es un *scheduler* de colocación inicial, no un balanceador adaptativo.
- **Fix.** Implementar migración real (mover una VM/cloudlet a otro host cuando detecta
  desbalance y el `didMigrate` se active), o al menos renombrar la acción y ajustar el
  planteamiento/documentación para que el término "load balancing" sea honesto.

---

## B. Gap conceptual: la precisión más baja NO corre más rápido en este stack

### B1. La comparación de "latencia" entre precisiones es prácticamente ficticia aquí
- Archivo: `precision.py:111-160` (`NativeBitLinear`)
- **Problema.** La capa redondea pesos y activaciones a una rejilla int4/int8/ternaria, pero el
  forward sigue siendo `F.linear` en **float32**. En GPU/CUDA ejecuta exactamente los mismos
  kernels dense f32 sin importar la "precisión".
- **Consecuencia.** El benchmark de inferencia (`agent.py:237-302`) medirá ~0 de speedup y no se
  puede dibujar un frente de Pareto de **latencia real**. `packed_size_mb` es teórico.
- **Opciones.**
  1. Usar kernels reales de bajo bit: **torchao** (int8/int4 weights con matmul int), **bitsandbytes**
     o kernels CUDA custom.
  2. Re-enmarcar la eficiencia como *tamaño empaquetado y viabilidad de despliegue edge/TinyML*,
     no como latencia de forward.

### B2. La exploración de capacidad NO está normalizada → la comparación es injusta
- Archivo: `precision.py:189-203` (`h4_capacity_plan`), `configs/h4_native_precision.yaml`
- **Problema.** `h4_capacity_plan` deja que las precisiones bajas escalen ancho (256→4096)
  mientras `32` queda clavado en 256. Con 2 capas, `ternary` h4096 tiene ~50M parámetros y
  MUCHÍSIMO más FLOPs que `32` h256 (~200K): **2–3 órdenes de magnitud** de coste de cómputo.
- **Consecuencia.** Al reportar `best_eval_reward` por celda estás comparando modelos de coste
  totalmente distinto; el "punto óptimo" encontrado puede ser un artefacto del diseño del grid.
- **Fix.** Normalizar el eje de capacidad por `n_params × bits` o por **FLOPs** reales, y dibujar
  el Pareto en ejes *(coste real de cómputo) vs (calidad)*.

---

## C. Evaluación y comparabilidad

### C1. No hay evaluación limpia ni split held-out
- Archivos: `main.py:348-373`, `early_stopping.py`
- **Problema.** El early stopping y `best_eval_reward` se basan en el reward de **entrenamiento**
  (además contaminado por A1/A2).
- **Fix.** (a) corregir la recompensa, (b) evaluar con `epsilon = 0` en un subconjunto de carga
  aparte (held-out), y (c) reportar **métricas de negocio**: tasa de violación SLA por cloudlet
  (sin doble conteo), makespan y utilidad media — no solo la suma de reward. El reward es un
  proxy; las métricas físicas son las comparables.

### C2. Reward no normalizado por número de pasos
- Archivo: `results.py` (`RunResult`)
- **Fix.** Añadir `steps/episodio` (y nº de cloudlets) al `RunResult` y normalizar el reward sum
  por pasos; de lo contrario las filas del CSV no son comparables entre sí.

### C3. "Coste" es redundante con energía
- Archivo: `SimulationGateway.java:242-245`
- **Problema.** `getOperationalCost() = energía × 0.048`: la tercera recompensa está 100%
  correlacionada con energía y no aporta información independiente a los 3 pesos adaptativos.
- **Fix.** Modelo de coste **diferenciado** (por tipo de VM / precio por host) para descorrelacionar
  `cost` y `energy`.

---

## D. Fidelidad de la simulación / estado del agente

### D1. La observación no deja que el agente "vea" el problema de balanceo
- Archivo: `env.py:24-29`, `SimulationGateway.java:297-307`
- **Problema.** Solo `cpu_util`, `ram_util` y `queue_depth` instantáneos; en el primer paso todo
  está a cero, así que la acción inicial es a ciegas.
- **Fix.** Enriquecer con: carga MIPS pendiente por VM, `eFinishTime` estimado por VM, y
  características del cloudlet entrante (`mi`, `num_pes`, `deadline`). Da señal real para
  cumplir deadlines.

### D2. La correspondencia VM→host es una suposición frágil
- Archivo: `SimulationGateway.java:201` (`vms.get(safeHost)`)
- **Problema.** Se asume que la VM `i` vive en el host `i`, pero esa asignación la decide el
  broker por defecto (a veces aleatoria). La acción "enviar al host h" puede no significar lo que
  se cree.
- **Fix.** Mapear explícitamente cada VM a su host al crearla.

### D3. Sin determinismo en el simulador Java
- Archivo: `agent.py:177-182`
- **Problema.** Se siembran `random`, `numpy` y `torch`, pero CloudSimPlus **no se siembra**; el
  scheduling interno del broker puede introducir variación entre runs que ensucie el Pareto.
- **Fix.** Sembrar/forzar determinismo en el lado Java (o fijar el scheduler del broker).

---

## E. Ingeniería / cierre

### E1. Falta el analizador del Pareto (la pieza que cierra la idea)
- Archivo: `results.py` acumula `h4_results.csv`, pero no hay script que lo lea y calcule el
  frente de Pareto.
- **Sugerencia.** Script `python-agent/analyze_pareto.py` que:
  1. Lea `outputs/results/h4_results.csv`.
  2. Calcule el **conjunto no-dominado** en ejes *(coste real normalizado, reward limpia)*.
  3. Emita una tabla/figura con el punto óptimo y los puntos del frente.

### E2. `env.step()` devuelve reward `0.0`
- Archivo: `env.py:132`
- **Problema.** La recompensa se computa fuera, en `main.py`; rompe la API Gymnasium y la
  reutilización del entorno.
- **Fix.** Mover el cálculo de recompensa dentro del `env` (o documentarlo/limpiarlo).

### E3. Presupuesto de cómputo
- **Nota.** Con los bugs A1/A2 y los puntos B1/B2 aún presentes, entrenar `ternary` h4096 es
  gastar GPU: no aporta latencia real y su reward está sesgado. Corregir primero A y B.

---

## Prioridades recomendadas

| Prioridad | Qué | Por qué |
|-----------|-----|---------|
| **P1** | Fix A1 (energía delta) y A2 (SLA sin doble conteo) | Sin esto el resultado del sweep es engañoso |
| **P2** | Decidir la premisa de precisión: edge (packed_size/FLOPs) o velocidad real (kernels low-bit / torchao) | Determina si B1/B2 tienen sentido |
| **P3** | Evaluación held-out + métricas de negocio (SLA rate, makespan) | Comparación honesta entre celdas del grid |
| **P4** | Script de análisis del Pareto sobre `h4_results.csv` con coste normalizado | Cierra el workflow y muestra el óptimo real |

---

## Resumen de fixes de alto impacto (P1)

1. **Energía por paso** — pasar `energia_actual - energia_anterior` al reward.
2. **SLA por cloudlet** — contabilizar cada cloudlet una sola vez (Set o contador en finish) en
   `updateEnergyAndSla`.
3. *(bono)* **Held-out + métricas de negocio** — reportar SLA rate y makespan reales en lugar de
   la suma de reward para decidir el punto óptimo de capacidad×precisión.