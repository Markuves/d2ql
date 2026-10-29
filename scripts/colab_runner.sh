#!/bin/bash
# ==============================================================================
# d2ql - Automated Runner for Google Colab
# Usage:
#   bash scripts/colab_runner.sh --shard-id 0 --num-shards 2
#   bash scripts/colab_runner.sh --shard-id 1 --num-shards 2
#   bash scripts/colab_runner.sh --slice 0:33
# ==============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "================================================================="
echo " Starting d2ql Colab Environment Setup"
echo " Repo directory: $REPO_DIR"
echo " Arguments: $@"
echo "================================================================="

# 1. Install Java 21 & Maven for CloudSimPlus Gateway
echo "[1/5] Checking Java & Maven..."
PACKAGES_TO_INSTALL=""
if ! command -v java >/dev/null 2>&1 || ! java -version 2>&1 | grep -q "21"; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL openjdk-21-jdk"
fi
if ! command -v mvn >/dev/null 2>&1; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL maven"
fi
if ! command -v nc >/dev/null 2>&1; then
    PACKAGES_TO_INSTALL="$PACKAGES_TO_INSTALL netcat-openbsd"
fi

if [ -n "$PACKAGES_TO_INSTALL" ]; then
    echo "Installing missing packages:$PACKAGES_TO_INSTALL..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq $PACKAGES_TO_INSTALL
fi

# 2. Compile Java Sim JAR if not compiled
JAR_PATH="$REPO_DIR/java-sim/target/java-sim-1.0.0.jar"
if [ ! -f "$JAR_PATH" ]; then
    echo "[2/5] Building java-sim-1.0.0.jar with Maven..."
    (cd "$REPO_DIR/java-sim" && mvn clean package -DskipTests -q)
    echo "JAR built successfully: $JAR_PATH"
else
    echo "[2/5] Java simulation JAR already exists: $JAR_PATH"
fi

# 3. Launch Java Simulation Gateway in Background (Py4J server on port 25333)
echo "[3/5] Starting Java Gateway service on port 25333..."
# Kill any previous instance if running
pkill -f "java-sim-1.0.0.jar" || true

# Run Java Gateway in background
nohup java -XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0 -jar "$JAR_PATH" > /tmp/java_gateway.log 2>&1 &
JAVA_PID=$!
echo "Java Gateway started with PID $JAVA_PID"

# Wait for Py4J port 25333 to be listening
echo "Waiting for Gateway port 25333 to respond..."
for i in {1..30}; do
    if nc -z localhost 25333 2>/dev/null; then
        echo "Java Gateway is UP and listening on port 25333!"
        break
    fi
    if ! kill -0 $JAVA_PID 2>/dev/null; then
        echo "ERROR: Java Gateway failed to start! Logs:"
        cat /tmp/java_gateway.log
        exit 1
    fi
    sleep 1
done

# 4. Prepare Dataset (Azure VM trace)
DATA_DIR="$REPO_DIR/data"
mkdir -p "$DATA_DIR"
TRACE_FILE="$DATA_DIR/workload.csv.gz"

if [ ! -f "$TRACE_FILE" ]; then
    echo "[4/5] Trace not found. Downloading Azure dataset..."
    python3 scripts/download_azure.py
    
    # Check if download_azure.py saved it under another name
    if [ -f "$DATA_DIR/trace_data_vmtable_vmtable.csv.gz" ]; then
        mv "$DATA_DIR/trace_data_vmtable_vmtable.csv.gz" "$TRACE_FILE"
    fi
    echo "Trace ready at: $TRACE_FILE"
else
    echo "[4/5] Dataset already exists at $TRACE_FILE"
fi

# 5. Setup Python Dependencies & Run Experiment
echo "[5/5] Setting up Python dependencies with uv..."
if ! command -v uv >/dev/null 2>&1; then
    pip install -q uv
fi

uv pip install -q -e .

echo "================================================================="
echo " Launching Training Sweep with arguments: $@"
echo "================================================================="

# Tell d2ql where the Java Gateway is running
export JAVA_HOST="localhost"
export JAVA_PORT="25333"

# Run training
python3 python-agent/main.py --config configs/h4_native_precision.yaml "$@"

echo "================================================================="
echo " Training finished successfully!"
echo " Results stored in outputs/"
echo "================================================================="
