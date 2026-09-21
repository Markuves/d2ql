# H5 — Comparación contra heurísticas tradicionales de balanceo

H5 evalúa **cuatro heurísticas clásicas de balanceo de carga** en el **mismo
entorno CloudSim** y sobre la **misma carga held-out** que el agente DDQN de H4,
recogiendo **las mismas métricas de negocio**, para comparar la política
aprendida contra baselines tradicionales.

## Las 4 heurísticas

Todas mapean la observación del entorno
`obs = [cpu_util × N, ram_util × N, queue_depth]` a un índice de host, sin
aprendizaje:

| Heurística       | Política                                                        |
|------------------|-----------------------------------------------------------------|
| `round_robin`    | Asigna los cloudlets a los hosts en rotación fija (ciega a carga) |
| `random`         | Host uniformemente aleatorio                                    |
| `least_loaded`   | Host con **menor utilización de CPU** (Least-Loaded)             |
| `least_combined` | Host que minimiza **cpu_util + ram_util** (Least-Combined)       |

Registro extensible: `python-agent/d2ql/heuristics.py` (`HEURISTICS`). Añadir una
heurística es una subclase con `name` + `select_action` y registrarla.

## Métricas recogidas (comparables con H4)

Se escriben en `outputs/results/h5_results.csv` (una fila por heurística):

- `eval_mean_reward` — reward medio sobre los episodios held-out
- `eval_makespan` — makespan medio
- `eval_sla_violations`, `eval_sla_rate` — violaciones de SLA (y tasa por cloudlet)
- `mean_energy`, `mean_cost` — energía y coste medios por episodio
- `decision_latency_mean_ms`, `_p50_ms`, `_p95_ms` — latencia por decisión
  (análoga a la latencia de inferencia del modelo en H4)
- `wall_clock_s`, `seed`, `notes`, `extra`

Las columnas espejan las de `h4_results.csv` (`eval_mean_reward`,
`eval_makespan`, `eval_sla_violations`, `eval_sla_rate`) para comparación directa.

### Nota sobre el reward

Las heurísticas usan los **pesos de reward iniciales** (fijos, no adaptativos,
`w_sla=0.4, w_energy=0.3, w_cost=0.3`), mientras el eval de H4 usa los pesos ya
adaptados por el entrenamiento. Por eso la comparación **más justa** son las
**métricas de negocio** (makespan/SLA/energía/coste), que son independientes de
los pesos. El reward también se reporta, pero con esa salvedad.

## Configuración

`configs/h5_heuristics.yaml`. Debe **espejar** los ajustes de H4
(`episode_length`, `holdout_frac`, `mi_scale`, `seed`, `n_episodes` de
evaluación) para que las heurísticas y el agente se evalúen sobre las **mismas
ventanas held-out**.

## Cómo ejecutar

Requiere el gateway `java-sim` arriba. **Un solo cliente a la vez** (el gateway
es una simulación con estado único; no corras H4 y H5 contra el mismo gateway
simultáneamente).

```bash
# 1. Levanta el gateway
docker compose up -d java-sim

# 2. Ejecuta H5 (una fila por heurística en outputs/results/h5_results.csv)
docker compose run --rm python-agent python main.py --config configs/h5_heuristics.yaml
```

`main.py` despacha H5 automáticamente cuando `heuristics.enabled: true` en el
config (`run_heuristics`), sin entrenar.

## Test

`tests/test_heuristics.py` — verifica el registro, la rotación de Round-Robin, la
selección least-loaded/least-combined, el determinismo del random por semilla y
que todos devuelvan un índice de host válido.
