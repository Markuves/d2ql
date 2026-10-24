# Snapshots de resultados

`data_h4.parquet` y `data.parquet` son **snapshots v1** del sweep H4 (132 filas),
generados con el código de `c4063eb`:

| Característica | v1 (estos parquet) | v2 (código actual) |
|---|---|---|
| Observación | 12 dims (incluye `time_to_deadline`, constante) | 11 dims |
| Orden de despacho | orden del trace | LPT por trabajo/PE |
| Semillas | `[41, 42]` | `[42, 41]` |
| Columnas SLA | `eval_sla_violations`, `eval_sla_rate` (siempre 0) | eliminadas |
| `best_mean_reward` | suma de la ventana de 50 episodios | media por episodio |
| Hold-out | cola cronológica (10.7× más ligera) | estratificado por MI |
| `extra` | 4 metadatos | + `schema_version`, `git_sha`, `config_sha256`, `run_timestamp_utc`, `obs_dim`, `dispatch_order`, `eval_split` |

Consecuencia: **no se pueden mezclar** filas v1 y v2 en un mismo análisis ni comparar sus
`norm_reward`/`best_mean_reward`. `analyze_pareto.py` etiqueta cada fila con `schema_version` y avisa
si hay mezcla. Para regenerar un snapshot v2 hay que re-correr el sweep (ver `docs/backlog.md`,
EVAL-06).
