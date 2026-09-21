package org.d2ql;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SimulationGatewayRegressionTest {

    @Test
    public void deterministicResetAndFixedActionSequence() {
        SimulationGateway gw = new SimulationGateway();

        // Push a small fixed workload
        gw.clearWorkload();
        gw.addWorkloadRow(0, 10, 5000, 1);
        gw.addWorkloadRow(0, 10, 5000, 1);

        int seed = 42;

        // First run
        gw.setSeed(seed);
        double[] obs1 = gw.reset();
        // Fixed round-robin actions 0,1,2,3 repeated for 4 steps
        int[] actions = {0, 1, 2, 3};
        for (int a : actions) {
            gw.step(a);
        }
        double makespan1 = gw.getMakespan();
        double sla1 = gw.getSlaViolationCount();

        // Second run with same seed
        gw.setSeed(seed);
        double[] obs2 = gw.reset();
        for (int a : actions) {
            gw.step(a);
        }
        double makespan2 = gw.getMakespan();
        double sla2 = gw.getSlaViolationCount();

        assertArrayEquals(obs1, obs2, 1e-6, "Same seed should produce identical observations");
        assertEquals(makespan1, makespan2, 1e-6, "Same seed should produce identical makespan");
        assertEquals(sla1, sla2, "Same seed should produce identical SLA violation count");

        // Verify explicit VM-to-host mapping is pinned
        // (indirect: deterministic metrics confirm no broker randomness)
        assertTrue(makespan1 >= 0, "Makespan must be non-negative");
    }
}
