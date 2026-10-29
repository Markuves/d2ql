# Guía de Ejecución Distribuida en Google Colab (d2ql - H4 Sweep)

Esta rama (`run_exp_colab`) añade la infraestructura para particionar y ejecutar las **66 combinaciones** del experimento H4 en **2 instancias paralelas de Google Colab** (33 combinaciones por máquina).

---

## Métodos de Ejecución

Tienes dos opciones:

### Opción A: Desde Terminal Local con `google-colab-cli` (Recomendado / Desatendido)

Si tienes instalada la CLI de Google Colab (`uv tool install google-colab-cli` o `pip install google-colab-cli`), puedes disparar los dos workers directamente en background desde dos terminales locales o en segundo plano:

#### Terminal 1 (Worker 1 - Combinaciones 0 a 32):
```bash
colab run --gpu T4 -- "git clone -b run_exp_colab <URL_DEL_REPO> d2ql && cd d2ql && bash scripts/colab_runner.sh --shard-id 0 --num-shards 2"
```

#### Terminal 2 (Worker 2 - Combinaciones 33 a 65):
```bash
colab run --gpu T4 -- "git clone -b run_exp_colab <URL_DEL_REPO> d2ql && cd d2ql && bash scripts/colab_runner.sh --shard-id 1 --num-shards 2"
```

---

### Opción B: Mediante Interfaz Web de Colab (2 Pestañas en Navegador)

1. Abre dos pestañas en Google Colab con entorno de ejecución GPU (T4 o A100).
2. Sube o abre [`notebooks/colab_run_experiments.ipynb`](notebooks/colab_run_experiments.ipynb).
3. En la **Pestaña 1**, ejecuta la celda de **Worker 1**:
   ```bash
   !bash scripts/colab_runner.sh --shard-id 0 --num-shards 2
   ```
4. En la **Pestaña 2**, ejecuta la celda de **Worker 2**:
   ```bash
   !bash scripts/colab_runner.sh --shard-id 1 --num-shards 2
   ```

---

## Parámetros de Sharding agregados a `d2ql.train`

El orquestador (`python-agent/main.py`) ahora soporta particionado flexible:
- `--shard-id <ID>`: Índice del shard (0-indexed).
- `--num-shards <N>`: Número total de particiones (ej. 2 para dividir en mitades, 3 para tercios, etc.).
- `--slice <START:END>`: Rango explícito de combinaciones (ej. `--slice 0:33` o `--slice 33:66`).

---

## ¿Qué hace automáticamente `scripts/colab_runner.sh`?
1. Detecta e instala OpenJDK 21 y Maven.
2. Compila el JAR de simulación si aún no existe (`java-sim-1.0.0.jar`).
3. Levanta el Gateway Py4J en background en el puerto 25333 y realiza un health-check hasta que esté respondiendo.
4. Descarga y prepara el dataset de Azure (`data/workload.csv.gz`) si no está presente.
5. Instala las dependencias de Python con `uv`.
6. Configura `JAVA_HOST=localhost` y arranca la simulación con el shard asignado.
