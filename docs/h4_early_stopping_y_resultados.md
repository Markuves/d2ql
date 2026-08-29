# H4 — Early Stopping, Resultados Comparables y Optimización de Tiempo

Documento de cambios para el experimento **H4 (Native Bit-Width Training)**.

Este documento explica qué se modificó en el proyecto para que:

1. Cada combinación de **precisión × capacidad** del modelo deje de entrenar
   sola cuando la *reward* deja de mejorar (early stopping).
2. Se guarden los resultados de **reward y latencia** de cada combinación para
   poder compararlos después.
3. Se reduzca el tiempo de simulación/aprendizaje del modelo.

---

## 1. Resumen de los cambios

| Archivo | Qué se hizo |
| --- | --- |
| `python-agent/d2ql/early_stopping.py` | **Nuevo.** Lógica de early stopping basada en ventana móvil + paciencia. |
| `python-agent/d2ql/results.py` | **Nuevo.** Guardado de resultados por combinación (JSON + CSV agregado). |
| `python-agent/d2ql/agent.py` | Se agregó `benchmark_inference()` (latencia del modelo) y metadatos de parámetros. |
| `python-agent/d2ql/metrics.py` | Se quitó un `flush()` por step de TensorBoard (cuello de I/O). |
| `python-agent/main.py` | Se integró el early stopping, el benchmark de latencia y el guardado de resultados en el loop de entrenamiento. |
| `configs/h4_native_precision.yaml` | Se añadieron las secciones `early_stopping`, `latency` y `results`. |

Verificación ejecutada: `pytest` (17 passed) + smoke tests unitarios de
`EarlyStopper` y `benchmark_inference` + compilación y parseo del YAML H4.

---

## 2. Early stopping (parada temprana)

**Problema anterior:** el barrido H4 entrenaba *siempre* los 600 episodios
(`training.n_episodes`) por cada una de las 15 combinaciones precision×capacidad,
aunque la *reward* ya hubiera convergido. Eso desperdiciaba tiempo de
entrenamiento y de simulación.

**Solución:** `EarlyStopper` (`python-agent/d2ql/early_stopping.py`).

Cómo decide cuándo parar:

- En cada evaluación (cada `eval_every_n_episodes` episodios) se registra la
  *reward* acumulada de esa ventana de evaluación.
- El criterio de comparación es la **media móvil** de las últimas
  `window_size` evaluaciones (no un episodio aislado), para no cortar por ruido.
- Se guarda la mejor ventana vista (`best_window_reward`).
- Si pasan `patience` evaluaciones consecutivas **sin** superar esa mejor marca
  (por más de `min_delta`), se detiene el run y se pasa a la siguiente
  combinación.
- `min_evaluations` evita cortar durante la fase de calentamiento / ruido inicial.

Parámetros (en `configs/h4_native_precision.yaml`):

```yaml
early_stopping:
  enabled: true
  patience: 3            # evaluaciones a esperar sin mejora antes de parar
  window_size: 3         # evaluaciones promediadas en la métrica de comparación
  min_evaluations: 4     # nunca parar antes de esta cantidad de evaluaciones
  min_delta: 0.001       # mejora mínima para contar como "mejor"
```

Comportamiento en `main.py`:

- Al detectar el plateau se imprime `EARLY STOP at episode X/600: ...` y se hace
  `break` del loop de episodios.
- Se sigue guardando `checkpoint_final.pt` (el modelo ya entrenado).
- El run reporta `episodes_trained` real (p. ej. 210 en vez de 600) y
  `stopped_early: true`.

> Nota de diseño: el operador de comparación es **estricto** (`>` con
> `min_delta`). Si fuera `>=`, una *reward* plana (sin ruido) nunca acumularía
> paciencia y el entrenamiento no se detendría jamás. Eso se probó en el smoke
> test.

---

## 3. Resultados comparables por combinación

**Objetivo:** poder comparar, después de correr todo el barrido, la *reward* y la
*latencia* de cada par precision×capacidad.

### 3.1 Detalle por run — `result.json`

Junto al checkpoint de cada combinación se escribe:

```
outputs/checkpoints/h4/<precision>_h<hidden>/result.json
```

Contiene: `experiment_id`, `precision`, `bits`, `hidden_size`,
`n_hidden_layers`, `episodes_trained`, `stopped_early`, `best_episode`,
`best_mean_reward`, `best_eval_reward`, `best_eval_episode`, latencia
(`latency_mean_ms`, `latency_p50_ms`, `latency_p95_ms`, `latency_n_samples`),
`params`, `packed_size_mb`, `wall_clock_s`, `device`, `seed`, `notes`, `extra`.

### 3.2 Tabla agregada — `h4_results.csv`

Cada vez que termina un run se **añade una fila** al CSV agregado:

```
outputs/results/h4_results.csv
```

Columnas:

```
experiment_id, precision, bits, hidden_size, n_hidden_layers,
episodes_trained, stopped_early, best_episode, best_mean_reward,
best_eval_reward, best_eval_episode, latency_mean_ms, latency_p50_ms,
latency_p95_ms, latency_n_samples, params, packed_size_mb,
wall_clock_s, device, seed, notes, extra
```

El `extra` se guarda como texto JSON para mantener el CSV en 2 dimensiones.
La escritura está protegida con un lock para que runs en paralelo no
entrelacen líneas.

> Así comparas todas las pruebas abriendo un solo CSV (o con
> `pandas.read_csv("outputs/results/h4_results.csv")`), sin necesidad de
> revisar TensorBoard run por run.

---

## 4. Latencia del modelo (`benchmark_inference`)

**Por qué:** la latencia de "fin de episodio" en este proyecto está dominada por
el paso del simulador Java (Py4J), no por la red. Para comparar *公平* la
latencia entre precisiones y capacidades, medimos el **forward pass puro** del
Q-network.

`DDQNAgent.benchmark_inference(state_dim, n_samples=200, warmup=20)`:

- Pone la red en `eval()` y corre bajo `torch.no_grad()`.
- Hace `warmup` pasos para calentar kernels/gráficos antes de cronometrar.
- Toma `n_samples` tiempos de inferencia y reporta **mean / p50 / p95** en ms.
- En CUDA usa eventos con `torch.cuda.synchronize()` para medir los lanzamientos
  asíncronos de verdad; en CPU usa `time.perf_counter()`.

Estos valores se guardan tanto en `result.json` como en `h4_results.csv`.

---

## 5. Optimización del tiempo de entrenamiento

1. **Early stopping** — las combinaciones que convergen temprano terminan en
   pocas decenas de episodios en lugar de 600. Es el ahorro más grande.
2. **Se quitó el flush por step en TensorBoard** — `metrics.log_training`
   hacía `writer.flush()` en *cada* step de entrenamiento (decenas de miles de
   escrituras a disco por run). Ese era el principal cuello de I/O. TensorBoard
   ya hace flush por su propia cadencia, así que solo se hace flush al cerrar.
3. **`select_action` ya usaba `torch.no_grad()`** en el forward — se revisó y
   no había fuga de grafo/rendimiento ahí, así que no se tocó.

---

## 6. Cómo correr el experimento H4

```bash
docker compose run --rm python-agent uv run python main.py --config configs/h4_native_precision.yaml
```

Al terminar, revisar resultados:

```bash
# Tabla comparativa de todas las combinaciones
cat outputs/results/h4_results.csv

# Detalle de una combinación
cat outputs/checkpoints/h4/32_h256/result.json
```

---

## 7. Verificación

- `pytest tests/` → **17 passed**.
- Smoke test (en venv local):
  - `EarlyStopper` detiene en plateau (ventana 1 y 2), no detiene mientras
    mejora, respeta `min_evaluations`, y `min_delta` bloquea ruido.
  - `benchmark_inference` devuelve `mean_ms`, `p50_ms`, `p95_ms`, `n_samples`.
- Compilación de todos los módulos tocados + parseo del YAML H4 (plan de 15
  combinaciones se construye igual que antes).

**Limitación conocida:** no se ejecutó un entrenamiento H4 end-to-end real
porque el loop necesita el gateway Py4J de `java-sim` (Docker), que no está
activo en el entorno de desarrollo. La lógica está verificada a nivel de unidad
y de cableado; el flujo completo contra el simulador Java debe lanzarse con el
comando de la sección 6.
