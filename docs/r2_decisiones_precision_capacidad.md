# Decisiones del proyecto — precisión × capacidad (H4)

> Documento de **decisiones aplicadas** (continúa a `docs/r1_recommendations.md`).
> Estado: aplicado en código y config; tests en verde (42 passed).

Estas son las decisiones tomadas sobre el proyecto y exactamente qué cambió.

---

## D1 — Throughput a varios batch + Pareto por FLOPs normalizados

**Decisión.** Dejar de reportar la velocidad como un único punto (latencia a batch=1 +
throughput a batch=64) y pasar a (a) medir la curva **throughput vs batch** completa, y (b)
analizar el compromiso precisión×capacidad sobre **ejes de coste normalizados** (FLOPs y
capacidad efectiva), no sobre latencia cruda.

**Por qué.** A batch=1 todo es overhead de lanzamiento (en los datos previos, fp32 h4 medía más
que h16: imposible si se midiera cómputo). Y la latencia cruda mezcla tres confounds a la vez
(bits, device, ancho); los FLOPs son invariantes al ancho de bits, así que normalizar por ellos
deja cualquier diferencia de calidad atribuible **solo** a la precisión.

**Qué cambió.**

| Archivo | Cambio |
|---|---|
| `python-agent/d2ql/agent.py` | Nuevo `DDQNAgent.benchmark_throughput_sweep(state_dim, batch_sizes, ...)`: corre `benchmark_throughput` para cada batch y devuelve `{"<batch>": {samples_per_sec, ms_per_batch}}`. |
| `python-agent/d2ql/results.py` | Nuevo campo/columna `throughput_by_batch` (JSON) en `RunResult`; se serializa junto a `extra`. Las columnas escalares `throughput_pps` / `throughput_batch_size` se conservan. |
| `python-agent/main.py` | Lee `latency.throughput_batch_sizes` (lista) y llama al sweep; la columna escalar toma el batch primario (`throughput_batch_size`) o, si no está en la lista, el mayor medido. |
| `configs/h4_native_precision.yaml` | `latency.throughput_batch_sizes: [1, 8, 32, 128]`. |
| `python-agent/analyze_pareto.py` | **Nuevo**: agrega por celda (media ± std sobre semillas) y calcula el conjunto no dominado en cuatro ejes de coste: `flops`, `effective_capacity_bits`, `packed_size_mb` y `latency_mean_ms`; además imprime la tabla throughput-vs-batch. |

**Cómo se usa.**

```bash
python python-agent/analyze_pareto.py --results outputs/results/h4_results.csv
python python-agent/analyze_pareto.py --quality norm_reward --out outputs/results/pareto.md
```

Un **cruce** de curvas de throughput a batch creciente es lo único que demuestra una ventaja real
de kernel; si ninguna cruza a fp32, la ventaja de la baja precisión es de tamaño, no de velocidad.

---

## D2 — Semillas múltiples (y SLA fuera de alcance)

**Decisión.** Replicar cada configuración con **≥3 semillas**. El eje de SLA **no se usará** en las
próximas pruebas, así que el arreglo de deadline/`mi_scale` del simulador queda **deliberadamente
fuera de alcance** (documentado como pendiente, no aplicado).

**Por qué.** Con una sola semilla y 10 episodios de evaluación, la dispersión del reward dentro de
una misma precisión (σ ≈ 3) es del mismo tamaño que el efecto que se quiere medir: cualquier
"óptimo" elegido así es ruido. El SLA no es un eje útil mientras esté estructuralmente en 0.

**Qué cambió.**

| Archivo | Cambio |
|---|---|
| `configs/h4_native_precision.yaml` | `experiment.seeds: [42, 43, 44]` (el `seed: 42` se conserva como fallback). |
| `python-agent/main.py` | Nuevo `_experiment_seeds(config)`; el plan H4 pasa a ser `seeds × (precisión, device, ancho)`; cada réplica fija `experiment.seed` (que alimenta al agente, al trace loader y al gateway Java vía `env.reset -> setSeed`) y usa tag `{precisión}_{device}_h{ancho}_l{capas}_s{semilla}` (carpeta de checkpoint distinta por réplica). El camino no-H4 también se replica por semilla. |

**Efecto en el plan.** 5 precisiones × 23 pares (ancho, profundidad) × 3 semillas = **138 corridas**
(la profundidad entra por D4).

---

## D3 — int8 fuera, ternario entra; bf16 se añade; kernel ternario bit-packed

**Decisión.**
1. **Retirar int8** del sweep de precisión: en este CPU (Zen 3, sin VNNI) un GEMM int8 no tiene
   camino acelerado, solo emulación int32, y siempre es más lento que fp32.
2. **Sustituirlo por ternario**: es la única precisión baja que puede batir a fp32 en este hardware,
   porque admite un kernel real de **dos pasadas binarias** (XOR + popcount).
3. **Implementar ese kernel** ternario bit-packed en CPU y conectarlo al camino de despliegue.
4. **Añadir bf16 en GPU** como el punto "mitad de tamaño, con rango de exponente de fp32".

**Qué cambió.**

| Archivo | Cambio |
|---|---|
| `python-agent/d2ql/kernels.py` | Nuevo `ternary_matmul_batched(x, weight, bias)`: descompone `w = s·(w_pos − w_neg)` y ejecuta dos pasadas del kernel binario (`dot(x,w_pos) − dot(x,w_neg) = 2·dot(x,w)`, de ahí el factor 0.5). Procesa por bloques de filas (`~2e7` elementos) para no materializar un temporal `[M,N,W]` gigante a batch grande. |
| `python-agent/d2ql/precision.py` | Nuevo `lowbit_ternary_matmul(...)`; `NativeBitLinear.forward` en modo `deploy` y en **CPU** con precisión `ternary` lo usa (en CUDA se mantiene el camino `int_mm`/Triton). `PRECISION_BITS["bf16"] = 16`, `compute_dtype("bf16") -> torch.bfloat16`, `parse_precision` acepta `"bf16"`. |
| `python-agent/d2ql/agent.py` | Soporte bf16: `amp_dtype` generalizado (`fp16` con `GradScaler`, `bf16` con autocast **sin** scaler por su rango de exponente); `_cast_fwd` usa `self.amp_dtype`. El `GradScaler` ahora solo se construye para fp16 (antes se creaba en cualquier run CUDA aunque fuera un no-op). |
| `configs/h4_native_precision.yaml` | Precisiones: `fp32-gpu`, `fp16-gpu`, `bf16-gpu`, `fp32-cpu`, `ternary-cpu`. `learning_rate_overrides`: `ternary: 0.0003`. `max_hidden_size`: `16: 4096` (aplica a fp16 y bf16), `ternary: 4096`. |
| `tests/test_precision.py` | Dos tests nuevos: exactitud del kernel ternario contra la referencia `sign(x) @ wᵀ` (con K=40 para ejercitar el padding de palabra) y verificación de que la capa ternaria en CPU usa el kernel empaquetado en modo deploy. |

**bf16 solo en GPU.** En CPU no se mide: Zen 3 carece de AVX512-BF16, así que sería emulado y el
resultado sería un artefacto de hardware, no de precisión.

**Nota de fidelidad.** El kernel ternario binariza las activaciones a ±1 (formulación BNN estándar).
Eso es una **aproximación de despliegue** del camino de entrenamiento STE (que usa activaciones de
8 bits), no un equivalente bit-exacto. Lo que se mide con él es latencia/throughput reales de un
kernel genuino, no su fidelidad al modelo entrenado.

---

## D4 — Profundidad como vector + escalado de tiempo de simulación

**Decisión.**
1. `n_hidden_layers` deja de ser un escalar en el sweep y pasa a ser un **vector**: H4 barre cada
   valor. Por ahora `[1, 2]`.
2. Se expone un **parámetro de escalado de tiempo de simulación** en el config H4, para poder
   modificar cuánto tiempo simulado dura la ejecución de cada cloudlet.

**Qué cambió.**

| Archivo | Cambio |
|---|---|
| `python-agent/main.py` | Nuevos `_resolve_layer_counts()` (acepta int o lista, deduplica y ordena) y `_build_native_plan()` (expande el sweep a `(precisión, device, ancho, profundidad)`); el tag de cada corrida pasa a `{prec}_{device}_h{ancho}_l{capas}_s{semilla}` y se loguea el tamaño del plan. |
| `python-agent/d2ql/workload.py` | `simulation_time_scale` (canónico) escala el MI estimado de cada cloudlet; como `exec_time ≈ mi/mips`, escala el tiempo simulado 1:1. Tiene **precedencia** sobre el antiguo `mi_scale`, que se mantiene como alias (y avisa por log si los dos están puestos y difieren). Sin ninguno de los dos, el default histórico es 1000.0. |
| `configs/h4_native_precision.yaml` | `capacity.n_hidden_layers: [1, 2]` y `workload.simulation_time_scale: 0.1` (mismo valor que antes tenía `mi_scale`). |
| `configs/h5_heuristics.yaml` | Solo comentario: h5 sigue con `mi_scale: 0.1`, que es el alias del mismo parámetro, así que sigue coincidiendo con H4. |
| `tests/test_h4_plan.py`, `tests/test_workload.py` | Nuevos: expansión del plan (profundidad y tope por precisión), parsing de semillas, y precedencia/escalado del tiempo de simulación. Incluye un guard sobre el config real de H4 (138 corridas). |

**Qué significa el 0.1.** Que cada cloudlet termina en <0.1 s de tiempo simulado — igual que antes.
Subirlo alarga el tiempo simulado y bajarlo lo acorta. Distinción importante: lo que escala es el
**tiempo de ejecución** (vía la cantidad de instrucciones); el `deadline` que viene del trace **no**
se escala, y esa asimetría es justo lo que mantiene el SLA en 0 (ver pendientes).

---

## Estado medido (honesto)

- **Tests:** 42 passed, incluidos los del kernel ternario (el kernel es **exacto** contra
  la referencia de signos).
- **Smoke end-to-end** (red pequeña, h=64, CPU, sin gateway):
  - `ternary-cpu`: latencia 0.47 ms vs `fp32-cpu` 0.024 ms a batch=1 (~20×).
  - Throughput vs batch — ternario `{1: 2112, 8: 15899, 32: 46973}` vs fp32-cpu
    `{1: 47029, 8: 265810, 32: 1073705}`: ternario escala con el batch, pero se mantiene **~22× por
    detrás** de fp32 en todo el rango.
  - Conclusión honesta: **la infraestructura de medición es real y el kernel es correcto, pero a
    esta escala (h≤128, batch≤128) no aparece ventaja de ternario en CPU.** El cruce, si existe,
    está a batch/ancho mayores — que es exactamente lo que el barrido de D1 ahora permite ver.
- `bf16` verificado como dtype (`bfloat16`) y a través del camino de agente.

---

## Lo que NO se hizo (a propósito)

- **No** se arregló el SLA (deadline absoluto del trace + `mi_scale=0.1`): queda pendiente para
  cuando se retome esa métrica.
- **No** se eliminó la maquinaria int8 interna (`quantize_int8`, `lowbit_real_matmul`, el kernel
  Triton). Sigue siendo la representación que usa `torch._int_mm` para las precisiones de rejilla en
  CUDA. Lo que se retiró es **int8 como precisión estudiada** en el sweep.
- **No** se midió bf16 en CPU (ver arriba).

---

## Coste del sweep y cómo correrlo

138 corridas (5 precisiones × 23 pares ancho×profundidad × 3 semillas). Con los tiempos observados
(~350–950 s por corrida en los anchos bajos, más en 2048/4096), el total es de **bastantes horas**
(>20 h); el early stopping recorta las corridas que estancan. Si se quiere recortar, reducir
`native_precision.capacity.hidden_sizes`, `capacity.n_hidden_layers` o las semillas en
`configs/h4_native_precision.yaml`.

```bash
# Sweep completo (requiere java-sim arriba)
docker compose run --rm python-agent python main.py --config configs/h4_native_precision.yaml

# Análisis (agregado por semilla + frentes de Pareto + curva throughput-vs-batch)
docker compose run --rm python-agent python analyze_pareto.py --results outputs/results/h4_results.csv
```

> Recordatorio: las imágenes hornean el código. Tras estos cambios hay que reconstruir
> (`docker compose up --build`) o usar `docker cp` + restart, o los contenedores correrán el código
> viejo.

---

## Pendientes

1. **SLA**: re-basar el deadline al episodio (relativo) y subir `mi_scale` para que la ejecución sea
   comparable al presupuesto; hoy `eval_sla_rate = 0` en todo.
2. **Escalar el eje batch** en el sweep (256/512/1024) si se busca el punto de cruce del kernel
   ternario.
3. Decidir el **hardware objetivo** del despliegue; de eso depende si la pregunta por la precisión
   baja es de memoria/edge (donde ternario y bf16 ganan) o de velocidad (donde no).
