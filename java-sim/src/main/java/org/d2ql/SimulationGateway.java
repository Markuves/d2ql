package org.d2ql;

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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import py4j.GatewayServer;

import java.net.InetAddress;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

public class SimulationGateway {

    private static final Logger logger = LoggerFactory.getLogger(SimulationGateway.class);

    // Simulation state held at instance level, reset() rebuilds it
    private CloudSimPlus simulation;
    private DatacenterBrokerSimple broker;
    private int loadedCloudlets;
    private int normalizedPesCloudlets;
    private boolean simulationFinished;
    private int windowIndex = -1;
    private double windowStart;
    private double windowEnd;
    private List<Cloudlet> windowCloudlets = new ArrayList<>();
    private List<VmSimple> vms = new ArrayList<>();
    private int lastAction = -1;
    private double lastMakespan;
    private double lastEnergy;
    private double lastCost;
    private int lastSlaViolations;

    private static final int NUM_HOSTS = 4;
    private static final int HOST_PES = 8;
    private static final long HOST_RAM = 16_384;  // MB
    private static final long HOST_BW = 10_000;   // Mbps
    private static final long HOST_STORAGE = 1_000_000; // MB
    private static final int VM_PES = 2;
    private static final long VM_MIPS = 1000;
    private static final int NUM_VMS = 4;
    private static final double WORKLOAD_TIME_SCALE = Double.parseDouble(
        System.getenv().getOrDefault("WORKLOAD_TIME_SCALE", "0.001")
    );
    private static final double WORKLOAD_MIPS = Double.parseDouble(
        System.getenv().getOrDefault("WORKLOAD_MIPS", "1000")
    );
    private static final double WINDOW_DURATION = Double.parseDouble(
        System.getenv().getOrDefault("WINDOW_DURATION", "60000")
    );
    private static final int MAX_CLOUDLETS_PER_WINDOW = Integer.parseInt(
        System.getenv().getOrDefault("MAX_CLOUDLETS_PER_WINDOW", "10000")
    );
    private static final double INITIAL_WINDOW_START = Double.parseDouble(
        System.getenv().getOrDefault("INITIAL_WINDOW_START", "0")
    );

    public SimulationGateway() {
        // Instance is the Py4j entry point; initialization deferred to reset()
    }

    // reset() rebuilds the full simulation for each RL episode.
    // CloudSimPlus simulations cannot be restarted in place, so a fresh
    // CloudSimPlus instance and datacenter are constructed on every call.
    public void reset() {
        logger.info("Resetting simulation environment for new episode.");
        simulation = new CloudSimPlus();
        simulationFinished = false;
        loadedCloudlets = 0;
        normalizedPesCloudlets = 0;
        windowCloudlets = new ArrayList<>();
        vms = new ArrayList<>();
        lastAction = -1;
        lastMakespan = 0.0;
        lastEnergy = 0.0;
        lastCost = 0.0;
        lastSlaViolations = 0;
        windowIndex++;
        windowStart = INITIAL_WINDOW_START + windowIndex * WINDOW_DURATION;
        windowEnd = windowStart + WINDOW_DURATION;

        // Build hosts
        List<HostSimple> hosts = new ArrayList<>();
        for (int i = 0; i < NUM_HOSTS; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int j = 0; j < HOST_PES; j++) {
                peList.add(new PeSimple(VM_MIPS));
            }
            hosts.add(new HostSimple(HOST_RAM, HOST_BW, HOST_STORAGE, peList));
        }

        // Build datacenter and broker
        new DatacenterSimple(simulation, hosts);
        broker = new DatacenterBrokerSimple(simulation);

        for (int i = 0; i < NUM_VMS; i++) {
            vms.add(new VmSimple(VM_MIPS, VM_PES));
        }
        broker.submitVmList(vms);

        String datasetPath = System.getenv().getOrDefault(
            "DATASET_PATH", "/app/data/workload.csv.gz"
        );
        loadWorkload(Paths.get(datasetPath), vms);

        logger.info(
            "Simulation reset complete. {} hosts, {} VMs and {} Cloudlets ready.",
            NUM_HOSTS, NUM_VMS, loadedCloudlets
        );
        logger.info(
            "Active workload window {}: [{}..{}), max Cloudlets {}",
            windowIndex, windowStart, windowEnd, MAX_CLOUDLETS_PER_WINDOW
        );
    }

    public double[] step(int action) {
        if (simulation == null) {
            throw new IllegalStateException("Simulation not initialized. Call reset() first.");
        }
        if (!simulationFinished) {
            int selectedVm = Math.floorMod(action, vms.size());
            lastAction = selectedVm;
            for (Cloudlet cloudlet : windowCloudlets) {
                cloudlet.setVm(vms.get(selectedVm));
            }
            logger.info(
                "Action {} selected VM {} for {} Cloudlets in window {}",
                action, selectedVm, windowCloudlets.size(), windowIndex
            );
            simulation.start();
            simulationFinished = true;
            calculateMetrics();
        }
        return getObservation();
    }

    private void calculateMetrics() {
        lastMakespan = 0.0;
        for (Cloudlet cloudlet : windowCloudlets) {
            lastMakespan = Math.max(lastMakespan, cloudlet.getFinishTime());
        }
        lastEnergy = lastMakespan * 0.001;
        lastCost = lastMakespan * windowCloudlets.size() / 1000.0;
        lastSlaViolations = 0;
    }

    private void loadWorkload(Path datasetPath, List<VmSimple> vms) {
        if (!Files.isRegularFile(datasetPath)) {
            throw new IllegalStateException("Dataset not found: " + datasetPath);
        }

        try (
            InputStream fileInput = Files.newInputStream(datasetPath);
            GZIPInputStream gzipInput = new GZIPInputStream(fileInput);
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(gzipInput, StandardCharsets.UTF_8)
            )
        ) {
            String line;
            int rowNumber = 0;
            List<Cloudlet> cloudlets = new ArrayList<>();

            while ((line = reader.readLine()) != null) {
                rowNumber++;
                if (line.isBlank()) {
                    continue;
                }

                String[] columns = line.split(",", -1);
                if (columns.length < 11) {
                    throw new IllegalArgumentException(
                        "Dataset row " + rowNumber + " has " + columns.length
                            + " columns; expected at least 11"
                    );
                }

                double startTime = parseDouble(columns[3], rowNumber, "start_time");
                if (startTime < windowStart || startTime >= windowEnd) {
                    continue;
                }
                if (loadedCloudlets >= MAX_CLOUDLETS_PER_WINDOW) {
                    continue;
                }
                double endTime = parseDouble(columns[4], rowNumber, "end_time");
                double cpuUtilization = parseDouble(columns[5], rowNumber, "cpu_utilization");
                int pes = parsePes(columns[9], rowNumber);
                double duration = Math.max(0.001, endTime - startTime);
                long length = Math.max(
                    1L,
                    Math.round(duration * WORKLOAD_TIME_SCALE * WORKLOAD_MIPS
                        * Math.max(0.01, cpuUtilization))
                );

                CloudletSimple cloudlet = new CloudletSimple(length, pes);
                cloudlet.setSubmissionDelay(
                    Math.max(0.0, (startTime - windowStart) * WORKLOAD_TIME_SCALE)
                );
                windowCloudlets.add(cloudlet);
                cloudlets.add(cloudlet);
                loadedCloudlets++;
            }

            broker.submitCloudletList(cloudlets);
            logger.info(
                "Loaded {} Cloudlets from {} ({} PEs values normalized to VM capacity)",
                loadedCloudlets, datasetPath, normalizedPesCloudlets
            );
        } catch (IOException | NumberFormatException exception) {
            throw new IllegalStateException("Could not load dataset: " + datasetPath, exception);
        }
    }

    private double parseDouble(String value, int rowNumber, String columnName) {
        try {
            return Double.parseDouble(value.trim());
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(
                "Invalid " + columnName + " at dataset row " + rowNumber + ": " + value,
                exception
            );
        }
    }

    private int parsePes(String value, int rowNumber) {
        String normalizedValue = value.trim();
        int requestedPes;

        try {
            if (normalizedValue.startsWith(">")) {
                requestedPes = Integer.parseInt(normalizedValue.substring(1).trim()) + 1;
            } else {
                requestedPes = (int) Math.round(Double.parseDouble(normalizedValue));
            }
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException(
                "Invalid pes at dataset row " + rowNumber + ": " + value,
                exception
            );
        }

        if (requestedPes < 1) {
            throw new IllegalArgumentException(
                "PEs must be positive at dataset row " + rowNumber + ": " + value
            );
        }

        int effectivePes = Math.min(requestedPes, VM_PES);
        if (effectivePes != requestedPes) {
            normalizedPesCloudlets++;
        }
        return effectivePes;
    }

    // getObservation() returns a primitive double[] for efficient
    // Py4j transfer. Java objects and collections are significantly
    // slower across the bridge under RL training throughput.
    public double[] getObservation() {
        if (broker == null) {
            return new double[]{0.0, 0.0, 0.0};
        }
        double finishedCloudlets = broker.getCloudletFinishedList().size();
        double createdVms = broker.getVmCreatedList().size();
        double pendingCloudlets = Math.max(0.0, loadedCloudlets - finishedCloudlets);
        return new double[]{finishedCloudlets, createdVms, pendingCloudlets};
    }

    public int getLastAction() {
        return lastAction;
    }

    public double getMakespan() {
        return lastMakespan;
    }

    public double getEnergy() {
        return lastEnergy;
    }

    public double getCost() {
        return lastCost;
    }

    public int getSlaViolations() {
        return lastSlaViolations;
    }

    // isDone() signals episode termination to the Gymnasium wrapper
    public boolean isDone() {
        return simulationFinished;
    }

    public int getWindowIndex() {
        return windowIndex;
    }

    public double getWindowStart() {
        return windowStart;
    }

    public double getWindowEnd() {
        return windowEnd;
    }

    public int getLoadedCloudlets() {
        return loadedCloudlets;
    }

    public static void main(String[] args) {
        // Port configurable via environment variable, not hardcoded
        int port = Integer.parseInt(System.getenv().getOrDefault("GATEWAY_PORT", "25333"));

        try {
            InetAddress bindAddress = InetAddress.getByName("0.0.0.0");
            SimulationGateway app = new SimulationGateway();

            GatewayServer server = new GatewayServer.GatewayServerBuilder(app)
                    .javaAddress(bindAddress)
                    .javaPort(port)
                    .build();

            // JVM shutdown hook for clean server teardown
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                logger.info("Shutting down Py4J Gateway Server...");
                server.shutdown();
            }));

            logger.info("Starting Py4J Gateway Server on 0.0.0.0:{}...", port);
            server.start();

        } catch (Exception e) {
            //  Descriptive error message before exit, not just a stack trace
            logger.error("Fatal error starting Gateway Server on port {}: {}", port, e.getMessage(), e);
            System.exit(1);
        }
    }
}