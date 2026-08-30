package org.d2ql;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModel;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.VmSimple;
import py4j.GatewayServer;

import java.net.InetAddress;
import java.net.UnknownHostException;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class SimulationGateway {

    // Simulation components
    private CloudSimPlus simulation;
    private DatacenterSimple datacenter;
    private DatacenterBrokerSimple broker;
    private List<HostSimple> hosts = new ArrayList<>();
    private List<Cloudlet> cloudlets = new ArrayList<>();
    private List<VmSimple> vms = new ArrayList<>();

    // Workload loaded from trace
    private List<long[]> workloadRecords = new ArrayList<>();

    // Simulation state
    private int currentCloudletIndex = 0;
    private boolean finished = false;
    private double totalEnergyConsumed = 0.0;
    // C3 fix: differentiated per-host cost (price per watt differs per host), so
    // operational cost is NOT a constant multiple of energy. The agent can learn
    // to prefer cheaper hosts, decoupling the cost objective from energy.
    private double totalOperationalCost = 0.0;
    private int slaViolationCount = 0;
    // A2 fix: each cloudlet counts as an SLA violation at most once.
    private final Set<Integer> countedSlaCloudlets = new HashSet<>();

    // Configuration constants
    private static final int NUM_HOSTS = 4;
    private static final int NUM_VMS = 4;
    private static final int HOST_PES = 8;
    private static final long HOST_MIPS = 10_000;
    private static final long HOST_RAM = 32_768;   // MB
    private static final long HOST_BW = 10_000;    // Mbps
    private static final long HOST_STORAGE = 1_000_000;
    private static final String WORKLOAD_PATH = "data/workload.csv.gz";
    // C3 fix: USD per watt-hour, differentiated per host index so cost objective
    // is decoupled from the single flat energy number.
    private static final double[] HOST_PRICE_PER_WATT = {0.048, 0.024, 0.048, 0.030};
    // Linear server power model (Watts): idle floor + load-proportional term.
    // CloudSimPlus hosts ship with a PowerModelNull by default (getPower -> 0),
    // which made total energy and cost always zero. We compute power directly so
    // the energy / cost objectives are real and learnable.
    private static final double HOST_IDLE_WATT = 100.0;
    private static final double HOST_MAX_WATT = 250.0;

    public SimulationGateway() {
        // Workload windows are pushed from Python per episode.
        // Do not load the full 2.6M-row trace into the JVM at startup.
    }

    // ------------------------------------------------------------------
    // Workload loading
    // ------------------------------------------------------------------

    private void loadWorkload() {
        workloadRecords.clear();

        try (java.io.InputStream fileStream = openStream(WORKLOAD_PATH);
             BufferedReader reader = new BufferedReader(
                     new java.io.InputStreamReader(fileStream))) {

            // No header row in the Azure vmtable trace — read from row 0
            String line;
            int skipped = 0;
            int loaded = 0;

            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    skipped++;
                    continue;
                }
                String[] parts = line.split(",");
                if (parts.length < 11) {
                    skipped++;
                    continue;
                }
                try {
                    // Column layout (positional, no header):
                    // 0: vm_id, 1: subscription_id, 2: deployment_id,
                    // 3: submitted_at, 4: deadline,
                    // 5: cpu_max, 6: cpu_avg, 7: cpu_p95,
                    // 8: vm_category, 9: num_pes, 10: memory_gb
                    long submittedAt = (long) Double.parseDouble(parts[3].trim());
                    long deadline    = (long) Double.parseDouble(parts[4].trim());
                    long numPes      = Math.max(1, (long) Double.parseDouble(parts[9].trim()));
                    double cpuAvg    = Double.parseDouble(parts[6].trim());
                    long duration    = Math.max(1, deadline - submittedAt);
                    long mi          = Math.max(100, (long) ((cpuAvg / 100.0) * duration * 1000));

                    workloadRecords.add(new long[]{submittedAt, deadline, mi, numPes});
                    loaded++;
                } catch (NumberFormatException e) {
                    skipped++;
                }
            }
            System.out.printf("Loaded %d cloudlets from trace (%d rows skipped).%n", loaded, skipped);

        } catch (IOException e) {
            System.err.println("Failed to load workload: " + e.getMessage());
        }
    }

    private java.io.InputStream openStream(String path) throws IOException {
        java.io.File f = new java.io.File(path);
        java.io.InputStream raw = new java.io.FileInputStream(f);
        if (path.endsWith(".gz")) {
            return new java.util.zip.GZIPInputStream(raw);
        }
        return raw;
    }

    // ------------------------------------------------------------------
    // Py4J API — called from Python via gateway.entry_point
    // ------------------------------------------------------------------

    public double[] reset() {
        if (workloadRecords.isEmpty()) {
            loadWorkload();
        }
        return rebuildSimulation();
    }

    public void clearWorkload() {
        workloadRecords = new ArrayList<>();
    }

    public void addWorkloadRow(double submittedAt, double deadline, double mi, double numPes) {
        workloadRecords.add(new long[]{
            (long) submittedAt,
            (long) deadline,
            Math.max(100L, (long) mi),
            Math.max(1L, (long) numPes),
        });
    }

    public double[] resetEpisode() {
        return rebuildSimulation();
    }

    private double[] rebuildSimulation() {
        simulation  = new CloudSimPlus();
        hosts       = new ArrayList<>();
        cloudlets   = new ArrayList<>();
        vms         = new ArrayList<>();
        currentCloudletIndex = 0;
        finished             = false;
        totalEnergyConsumed  = 0.0;
        totalOperationalCost = 0.0;
        slaViolationCount    = 0;
        countedSlaCloudlets.clear();

        // Build hosts
        for (int i = 0; i < NUM_HOSTS; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int j = 0; j < HOST_PES; j++) {
                peList.add(new PeSimple(HOST_MIPS));
            }
            HostSimple host = new HostSimple(HOST_RAM, HOST_BW, HOST_STORAGE, peList);
            hosts.add(host);
        }

        datacenter = new DatacenterSimple(simulation, hosts);
        broker     = new DatacenterBrokerSimple(simulation);

        // Build VMs
        for (int i = 0; i < NUM_VMS; i++) {
            VmSimple vm = new VmSimple(HOST_MIPS, HOST_PES / 2);
            vm.setRam(HOST_RAM / 4).setBw(HOST_BW / 4).setSize(HOST_STORAGE / 4);
            vms.add(vm);
        }
        broker.submitVmList(vms);

        // Materialize the episode window but bind each cloudlet only when the agent acts.
        for (long[] rec : workloadRecords) {
            cloudlets.add(createCloudlet(rec));
        }

        simulation.startSync();

        return buildObservation();
    }

    public double[] step(int hostIndex) {
        int safeHost = Math.floorMod(hostIndex, NUM_HOSTS);

        // Assign the next pending cloudlet to the chosen host's VM
        if (currentCloudletIndex < cloudlets.size()) {
            Cloudlet cloudlet = cloudlets.get(currentCloudletIndex);
            VmSimple targetVm = vms.get(safeHost);
            cloudlet.setVm(targetVm);
            broker.submitCloudlet(cloudlet);
            currentCloudletIndex++;
        }

        if (simulation.isRunning()) {
            simulation.runFor(1.0);
            updateEnergyAndSla();
        }

        finished = currentCloudletIndex >= cloudlets.size() && !simulation.isRunning();

        return buildObservation();
    }

    public boolean isFinished() {
        return finished;
    }

    public double[] getHostCpuUtilizations() {
        double[] utils = new double[hosts.size()];
        for (int i = 0; i < hosts.size(); i++) {
            utils[i] = hosts.get(i).getCpuPercentUtilization();
        }
        return utils;
    }

    public double getTotalEnergyConsumed() {
        return totalEnergyConsumed;
    }

    public double getMakespan() {
        if (cloudlets.isEmpty()) return 0.0;
        double max = 0.0;
        for (Cloudlet c : cloudlets) {
            if (c.getFinishTime() > max) max = c.getFinishTime();
        }
        return max;
    }

    public double getOperationalCost() {
        // C3 fix: differentiated per-host cost accumulated in updateEnergyAndSla,
        // no longer a constant multiple of total energy.
        return totalOperationalCost;
    }

    public int getSlaViolationCount() {
        return slaViolationCount;
    }

    public double getSimulationTime() {
        return simulation != null ? simulation.clock() : 0.0;
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    private CloudletSimple createCloudlet(long[] rec) {
        long mi = rec[2];
        int numPes = (int) Math.min(Math.max(rec[3], 1), HOST_PES / 2);
        CloudletSimple cl = new CloudletSimple(mi, numPes);
        cl.setFileSize(300).setOutputSize(300);
        // CPU can saturate a VM; RAM/BW are shared so co-located cloudlets do not
        // each demand 100% of the VM (8192 MB / 2500 Mbps) and stall.
        cl.setUtilizationModelCpu(new UtilizationModelFull());
        UtilizationModel ramBw = new UtilizationModelDynamic(0.1);
        cl.setUtilizationModelRam(ramBw);
        cl.setUtilizationModelBw(ramBw);
        return cl;
    }

    private void updateEnergyAndSla() {
        double stepEnergy = 0.0;
        double stepCost = 0.0;
        for (int i = 0; i < hosts.size(); i++) {
            HostSimple host = hosts.get(i);
            double util = host.getCpuPercentUtilization();
            if (Double.isNaN(util) || util < 0.0) util = 0.0;
            if (util > 1.0) util = 1.0;
            // Linear power model: idle 100W, +150W at full load. Independent of the
            // (null by default) CloudSimPlus power model.
            double power = HOST_IDLE_WATT + (HOST_MAX_WATT - HOST_IDLE_WATT) * util;
            stepEnergy += power;
            // C3: cost weights each host by its own price per watt, differing by index.
            double price = HOST_PRICE_PER_WATT[Math.min(i, HOST_PRICE_PER_WATT.length - 1)];
            stepCost += power * price;
        }
        // Convert watts to watt-hours (step size = 1 simulated second)
        totalEnergyConsumed += stepEnergy / 3600.0;
        totalOperationalCost += stepCost / 3600.0; // C3

        // A2 fix: count each cloudlet as an SLA violation at most once. Previously
        // every finished-and-past-deadline cloudlet was re-counted on each step,
        // inflating the count superlinearly and making reward non-comparable.
        for (int i = 0; i < cloudlets.size() && i < workloadRecords.size(); i++) {
            if (countedSlaCloudlets.contains(i)) {
                continue;
            }
            Cloudlet cl = cloudlets.get(i);
            if (cl.getFinishTime() > 0) {
                long deadline = workloadRecords.get(i)[1];
                if (cl.getFinishTime() > deadline) {
                    countedSlaCloudlets.add(i);
                    slaViolationCount++;
                }
            }
        }
    }

    private double[] buildObservation() {
        // obs = [cpu_util x NUM_HOSTS, ram_util x NUM_HOSTS, queue_depth]
        double[] obs = new double[NUM_HOSTS * 2 + 1];
        for (int i = 0; i < hosts.size() && i < NUM_HOSTS; i++) {
            obs[i]            = hosts.get(i).getCpuPercentUtilization();
            obs[NUM_HOSTS + i] = hosts.get(i).getRam().getPercentUtilization();
        }
        int pending = Math.max(0, workloadRecords.size() - currentCloudletIndex);
        obs[NUM_HOSTS * 2] = Math.min(1.0, pending / (double) Math.max(workloadRecords.size(), 1));
        return obs;
    }

    // ------------------------------------------------------------------
    // Entry point
    // ------------------------------------------------------------------

    public static void main(String[] args) throws UnknownHostException {
        int port = Integer.parseInt(System.getenv().getOrDefault("GATEWAY_PORT", "25333"));
        SimulationGateway gateway = new SimulationGateway();
        // Bind on all interfaces so python-agent can reach us on the Docker network.
        // The no-arg GatewayServer constructor only listens on 127.0.0.1.
        GatewayServer server = new GatewayServer.GatewayServerBuilder()
                .entryPoint(gateway)
                .javaPort(port)
                .javaAddress(InetAddress.getByName("0.0.0.0"))
                .build();
        server.start();
        System.out.printf("Py4J Gateway Server started on 0.0.0.0:%d.%n", port);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            server.shutdown();
            System.out.println("INFO  Shutting down Py4J Gateway Server...");
        }));
    }
}
