package d2ql;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.VmSimple;
import py4j.GatewayServer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

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
    private int slaViolationCount = 0;
    private boolean migratedLastStep = false;

    // Configuration constants
    private static final int NUM_HOSTS = 4;
    private static final int NUM_VMS = 4;
    private static final int HOST_PES = 8;
    private static final long HOST_MIPS = 10_000;
    private static final long HOST_RAM = 32_768;   // MB
    private static final long HOST_BW = 10_000;    // Mbps
    private static final long HOST_STORAGE = 1_000_000;
    private static final String WORKLOAD_PATH = "data/workload.csv.gz";

    public SimulationGateway() {
        loadWorkload();
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
        simulation  = new CloudSimPlus();
        hosts       = new ArrayList<>();
        cloudlets   = new ArrayList<>();
        vms         = new ArrayList<>();
        currentCloudletIndex = 0;
        finished             = false;
        totalEnergyConsumed  = 0.0;
        slaViolationCount    = 0;
        migratedLastStep     = false;

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

        // Submit first batch of cloudlets
        submitNextCloudlets();

        simulation.startSync();

        return buildObservation();
    }

    public double[] step(int hostIndex) {
        migratedLastStep = false;

        int safeHost = Math.floorMod(hostIndex, NUM_HOSTS);

        // Assign the next pending cloudlet to the chosen host's VM
        if (currentCloudletIndex < cloudlets.size()) {
            Cloudlet cloudlet = cloudlets.get(currentCloudletIndex);
            VmSimple targetVm = vms.get(safeHost);
            cloudlet.setVm(targetVm);
            currentCloudletIndex++;
        }

        // Advance simulation by one step
        if (!simulation.isRunning()) {
            finished = true;
        } else {
            simulation.runFor(1.0);
            updateEnergyAndSla();
            submitNextCloudlets();

            if (!simulation.isRunning()) {
                finished = true;
            }
        }

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
        // Simple cost model: $0.048 per host per simulated hour
        return totalEnergyConsumed * 0.048;
    }

    public int getSlaViolationCount() {
        return slaViolationCount;
    }

    public boolean didMigrateLastStep() {
        return migratedLastStep;
    }

    public double getSimulationTime() {
        return simulation != null ? simulation.clock() : 0.0;
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    private void submitNextCloudlets() {
        int batchSize = NUM_VMS;
        List<Cloudlet> batch = new ArrayList<>();

        for (int i = 0; i < batchSize && currentCloudletIndex + i < workloadRecords.size(); i++) {
            long[] rec = workloadRecords.get(currentCloudletIndex + i);
            long mi     = rec[2];
            long numPes = rec[3];

            CloudletSimple cl = new CloudletSimple(mi, (int) numPes,
                    new UtilizationModelFull());
            cl.setFileSize(300).setOutputSize(300);
            cloudlets.add(cl);
            batch.add(cl);
        }

        if (!batch.isEmpty()) {
            broker.submitCloudletList(batch);
        }
    }

    private void updateEnergyAndSla() {
        double stepEnergy = 0.0;
        for (HostSimple host : hosts) {
            stepEnergy += host.getPowerModel().getPower(host.getCpuPercentUtilization());
        }
        // Convert watts to watt-hours (step size = 1 simulated second)
        totalEnergyConsumed += stepEnergy / 3600.0;

        // Count cloudlets that exceeded their original deadline
        for (int i = 0; i < cloudlets.size() && i < workloadRecords.size(); i++) {
            Cloudlet cl = cloudlets.get(i);
            if (cl.getFinishTime() > 0) {
                long deadline = workloadRecords.get(i)[1];
                if (cl.getFinishTime() > deadline) {
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

    public static void main(String[] args) {
        SimulationGateway gateway = new SimulationGateway();
        GatewayServer server = new GatewayServer(gateway, 25333);
        server.start();
        System.out.println("Py4J Gateway Server started on port 25333.");

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            server.shutdown();
            System.out.println("INFO  Shutting down Py4J Gateway Server...");
        }));
    }
}
