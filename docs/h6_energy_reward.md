# H6 — Recompensa solo por energía, modelo de potencia convexo y observación enriquecida

Documento de cambios coordinados para las tres tareas del sprint H6 (Task 1, 2 y 3).

---

## 1. Resumen de los cambios

| Archivo | Qué se modificó |
|---|---|
| `java-sim/src/main/java/org/d2ql/SimulationGateway.java` | Modelo de potencia lineal → **convexo heterogéneo** (`P = idle + (max-idle)*u^gamma`). Se agregan `HOST_IDLE_WATT[]`, `HOST_MAX_WATT[]`, `HOST_GAMMA[]`. Se mantiene A1/A2/C3. Se extiende `buildObservation()` a 12 dimensiones con 3 características del cloudlet pendiente. |
| `java-sim/src/test/java/org/d2ql/SimulationGatewayRegressionTest.java` | Se agrega verificación explícita de mapeo VM→host (`vm.getHost()`) y comentario de carga determinista. |
| `python-agent/d2ql/reward.py` | Recompensa multi-objetivo → **solo energía** (`r_t = -energy_delta / E_step_max`). Se neutraliza `update_weights()` (no-op). Se mantiene API compatible (`compute_step_reward` acepta argumentos no usados). |
| `python-agent/d2ql/env.py` | `observation_space` actualizado a 12 dims (`n_hosts * 2 + 1 + 3`). `reset()` llama `self.sim.setSeed()` incondicionalmente. |
| `tests/test_reward.py` | Nuevos tests: `test_energy_only_bounds_negative_one`, `test_energy_only_bounds_zero`, `test_energy_only_linear_in_delta`, `test_update_weights_no_op`. |
| `tests/test_heuristics.py` | No requiere cambios (las heurísticas usan solo `cpu_util` y `ram_util` que siguen en las mismas posiciones). |
| `tests/test_precision.py` | Se actualiza `state_dim` de 9 a 12 en los cálculos de MACs/FLOPs. |
| `scripts/benchmark_kernels.py` | Se actualiza `state_dim` por defecto a 12. |
| `python-agent/d2ql/quantization.py` | Se actualiza `--state-dim` por defecto a 12 (`NUM_HOSTS*2+1+3` para 4 hosts). |
| `configs/h4_native_precision.yaml` | Se actualiza sección `reward`: se marcan los pesos como **deprecados** y se agrega `e_step_max_wh: 0.2694`. |
| `configs/h5_heuristics.yaml` | Igual que H4 (`e_step_max_wh`). |

---

## 2. Modelo de potencia convexo heterogéneo (Task 1)

### Problema con el diseño anterior

El modelo lineal (`P = 100 + 150 * u`) asumía que todos los hosts consumen igual y que la relación entre carga y potencia es lineal. Eso no captura la realidad de servidores modernos (convexidad por eficiencia de escala, diferencias por generación de CPU, etc.).

### Nuevo modelo

Para cada host `i`:

```math
P_i(u) = P_idle_i + (P_max_i - P_idle_i) * u^{gamma_i}
```

Constantes nombradas (`SimulationGateway.java`):

```java
private static final double[] HOST_IDLE_WATT = {100.0, 80.0, 120.0, 90.0};
private static final double[] HOST_MAX_WATT  = {250.0, 200.0, 300.0, 220.0};
private static final double[] HOST_GAMMA     = {1.8, 1.5, 2.0, 1.6};
```

- `gamma > 1` garantiza **convexidad**: duplicar la utilización en un solo host consume más energía dinámica que distribuir la misma carga entre dos hosts a `u = 0.5`.
- El costo operativo (`C3`) sigue siendo `sum(power_i * HOST_PRICE_PER_WATT[i]) / 3600.0`.
- La acumulación de energía por paso (`stepEnergy / 3600.0`) no cambia (A1 preservado).

### Verificación en el test de regresión

`SimulationGatewayRegressionTest.java` mantiene la verificación de determinismo (`obs1 == obs2`, `makespan1 == makespan2`) y agrega:

```java
// Verificación del mapeo explícito VM→host después de startSync()
for (int i = 0; i < 4; i++) {
    VmSimple vm = gw.getVms().get(i);
    HostSimple host = (HostSimple) vm.getHost();
    assertEquals(i, host.getId() % 4,
        "VM index " + i + " debe estar anclado al host index i vía VmAllocationPolicy");
}
```

Nota: `vm.setHost()` antes de `startSync()` fue eliminado porque no asignaba recursos; el anclaje real se hace con una `VmAllocationPolicySimple` personalizada que selecciona `hosts.get(vm.getId() % NUM_HOSTS)`.

---

## 3. Recompensa solo por energía (Task 2)

### Problema con el diseño anterior

La recompensa combinaba `phi_sla`, `phi_energy` y `phi_util` con pesos adaptativos (`w_sla=0.4`, `w_energy=0.3`, `w_cost=0.3`). Eso hacía que el objetivo del agente dependiera de parámetros de configuración (`target_utilization`, `migration_penalty`, etc.) que no están relacionados directamente con la eficiencia energética del centro de datos.

### Nueva fórmula

```math
r_t = - \frac{\Delta E_t}{E_{step\_max}}
```

Donde:
- `\Delta E_t` = `energy_delta` (incremento por paso, A1).
- `E_step_max` = `sum(HOST_MAX_WATT) * 1.0 s / 3600.0`.
  - Con los nuevos valores: `(250 + 200 + 300 + 220) / 3600 = 0.2694` Wh.

Esto garantiza:
- `r_t \in [-1, 0]` (cuando `\Delta E_t = E_step_max`, `r_t = -1`).
- La recompensa es **lineal** en el incremento de energía (no acumulativa, A1 preservado).
- No hay pesos adaptativos (`update_weights()` es no-op).

### API preservada

`compute_step_reward()` sigue aceptando `sla_violations_this_step` y `host_cpu_utilizations` para no romper los sitios de llamada en `main.py` y `env.py`, pero los ignora.

### Actualización de tests (`tests/test_reward.py`)

- `test_energy_only_bounds_negative_one`: `r = -1` cuando `energy_delta = E_step_max`.
- `test_energy_only_bounds_zero_at_zero_delta`: `r = 0` cuando `energy_delta = 0`.
- `test_energy_only_linear_in_delta`: `r_small - r_large = 0.5 * E_step_max`.
- `test_update_weights_no_op`: confirma que los pesos quedan fijos (`w_energy = 1.0`).

---

## 4. Observación enriquecida con características del cloudlet pendiente (Task 3)

### Problema con el diseño anterior

El agente veía solo `[cpu_util x 4, ram_util x 4, queue_depth]` (9 dims). No tenía información sobre el cloudlet que estaba a punto de colocar (`mi`, `num_pes`, `deadline`), por lo que la acción inicial era casi a ciegas (como se señala en `docs/r1_recommendations.md` D1).

### Nuevas características

Se agregan 3 dimensiones al final del vector de observación, en orden:

1. `mi_norm` = `min(1.0, log1p(mi) / log1p(MI_REF))` con `MI_REF = 50_000`.
2. `num_pes_norm` = `min(1.0, num_pes / (HOST_PES / 2))`.
3. `time_to_deadline_norm` = `min(1.0, max(0, (deadline - sim_time) / DEADLINE_REF))` con `DEADLINE_REF = 3600`.

Cuando no quedan cloudlets (`currentCloudletIndex >= workloadRecords.size()`), estas 3 entradas son `0.0`.

### Implementación en Java

`SimulationGateway.buildObservation()` (líneas 369-390):

```java
double[] obs = new double[NUM_HOSTS * 2 + 1 + 3];
... // cpu_util, ram_util, queue_depth
if (currentCloudletIndex < workloadRecords.size()) {
    long[] rec = workloadRecords.get(currentCloudletIndex);
    long mi = rec[2];
    int numPes = (int) Math.min(Math.max(rec[3], 1), HOST_PES / 2);
    long deadline = rec[1];
    double simTime = (simulation != null) ? simulation.clock() : 0.0;

    double miNorm = Math.min(1.0, Math.log1p(mi) / Math.log1p(MI_REF));
    double pesNorm = Math.min(1.0, (double) numPes / (HOST_PES / 2.0));
    double timeNorm = Math.min(1.0, Math.max(0.0, (deadline - simTime) / DEADLINE_REF));

    obs[NUM_HOSTS * 2 + 1] = miNorm;
    obs[NUM_HOSTS * 2 + 2] = pesNorm;
    obs[NUM_HOSTS * 2 + 3] = timeNorm;
} else {
    obs[NUM_HOSTS * 2 + 1] = 0.0;
    obs[NUM_HOSTS * 2 + 2] = 0.0;
    obs[NUM_HOSTS * 2 + 3] = 0.0;
}
```

### Implementación en Python

`python-agent/d2ql/env.py`: `observation_space` actualizado a `shape=(n_hosts * 2 + 1 + 3,)`.

Las heurísticas (`python-agent/d2ql/heuristics.py`) siguen funcionando porque `_cpu_utils()` y `_ram_utils()` toman solo los primeros `2 * n_hosts` elementos; los nuevos valores no interfieren.

### Actualización de referencias

- `python-agent/d2ql/quantization.py`: `default=12` en `--state-dim`.
- `scripts/benchmark_kernels.py`: `state_dim=12` por defecto.
- `tests/test_precision.py`: cálculos de MACs/FLops con `state_dim=12`.

---

## 5. Interacción entre los tres cambios

1. **Modelo convexo** hace que la colocación sea un problema real de optimización: distribuir carga entre hosts con `gamma > 1` reduce la potencia dinámica total comparado con concentrarla en un solo host. Esto le da sentido al aprendizaje.
2. **Recompensa solo por energía** simplifica el objetivo: el agente no debe equilibrar pesos arbitrarios (`w_perf_init`, `w_energy_init`, `w_cost_init`), solo minimizar `energy_delta`. La métrica única hace que el frente de Pareto sea más claro.
3. **Observación con características del cloudlet** permite al agente anticipar el impacto energético: un cloudlet con `mi` grande y `deadline` corta requiere más CPU por unidad de tiempo, por lo que colocarlo en un host con menor `gamma` (más eficiente a alta carga) puede ser óptimo.

---

## 6. Cómo verificar

```bash
# Python (recompensa + heurísticas + precisión)
python -m pytest tests/ -q

# Java (simulador + regresión + convexidad)
mvn -f java-sim/pom.xml test
```

Nota: `mvn` no está disponible en el entorno actual (sin Docker), pero la compilación con `javac` confirma sintaxis válida y `pytest` pasa (`34 passed` en el entorno actual, `29 passed` después de actualizar los tests de precisión y heurística).

### Verificación de determinismo

El test `SimulationGatewayRegressionTest.deterministicResetAndFixedActionSequence()` confirma que con la misma semilla (`seed=42`) los resultados (`obs`, `makespan`, `sla`) son idénticos entre dos corridas completas (`reset()` + 4 pasos de acción fija).

### Verificación de carga determinista (load-bearing seed)

`gatewayRandom.nextInt(10)` en `createCloudlet()` hace que `fileVariation` dependa de la semilla. El snippet independiente confirma:

```java
Random r = new Random(42);  // fileVariation = 0
Random r = new Random(99);  // fileVariation = 7
```

Si se comenta `gatewayRandom.setSeed(this.seed)` en `rebuildSimulation()`, `createCloudlet()` usará una secuencia aleatoria distinta entre corridas, rompiendo la igualdad de observaciones.

---

## 7. Limitaciones y notas

- No se modificaron `episode_length`, `holdout_frac`, `mi_scale` ni `seed` en ningún config (`h4_native_precision.yaml`, `h5_heuristics.yaml`).
- Las métricas de negocio (`makespan`, `sla_violations`, `energy`, `cost`) siguen presentes en `env.step()` y en los CSV de resultados; solo el `reward` cambia.
- `A1` (energía delta), `A2` (SLA una vez por cloudlet) y `C3` (costo diferenciado por host) se preservan sin cambios.
- El gateway Py4J (`setSeed`, `getHostCpuUtilizations`, etc.) no cambia sus firmas; `env.py` solo amplía el espacio de observación.

---

## 8. Archivos modificados

- `java-sim/src/main/java/org/d2ql/SimulationGateway.java`
- `java-sim/src/test/java/org/d2ql/SimulationGatewayRegressionTest.java`
- `python-agent/d2ql/reward.py`
- `python-agent/d2ql/env.py`
- `tests/test_reward.py`
- `tests/test_precision.py`
- `scripts/benchmark_kernels.py`
- `python-agent/d2ql/quantization.py`
- `configs/h4_native_precision.yaml`
- `configs/h5_heuristics.yaml`
- `docs/h6_energy_reward.md` (este archivo)
