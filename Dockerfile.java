# Stage 1: Build
# -DskipTests keeps the build fast and explicit until Java tests are written
FROM maven:3.9.6-eclipse-temurin-21 AS builder
WORKDIR /build

# Copy pom.xml first to cache dependency downloads as a separate layer 
# This means Maven dependencies are only re-downloaded when pom.xml changes,
# not on every source code change
COPY java-sim/pom.xml .
RUN mvn dependency:go-offline -q

# Copy source and build the fat JAR
COPY java-sim/src ./src
RUN mvn clean package -DskipTests -q

# -----------------------------------------------------------------------

# Stage 2: Runtime
# Use JRE instead of full JDK — javac is not needed at runtime [1][26]
# eclipse-temurin:21-jre is significantly smaller than the full JDK image
FROM eclipse-temurin:21-jre
WORKDIR /app

#  create and use a dedicated non-root user 
# Running as root inside a container is a security risk even with namespacing 
RUN groupadd -r d2ql && useradd -r -g d2ql -s /bin/false d2ql

# Copy the shaded JAR from the builder stage 
COPY --from=builder /build/target/java-sim-1.0.0.jar ./gateway.jar

# Ensure the non-root user owns the application file 
RUN chown d2ql:d2ql /app/gateway.jar

# Drop privileges before the process starts 
USER d2ql

# Expose the Py4j GatewayServer port
EXPOSE 25333

#  JVM flags tuned for container execution 
#
# -XX:+UseContainerSupport        — ensures JVM reads cgroup limits, not host RAM [21]
#                                   Required: without this the JVM may see host memory
#                                   and allocate far more heap than the container allows [21]
#
# -XX:InitialRAMPercentage=50.0   — start the heap at 50% of container memory [26]
#                                   avoids a slow ramp-up phase during JIT warm-up
#
# -XX:MaxRAMPercentage=75.0       — cap the heap at 75% of container memory [16][17][18][20]
#                                   the remaining 25% covers metaspace, thread stacks,
#                                   code cache, and JVM native overhead [21][26]
#
# -XX:+UseG1GC                    — G1 is the recommended GC for container workloads [26]
#                                   provides predictable pause times for the RL loop
#
# -XX:+ExitOnOutOfMemoryError     — crash cleanly on OOM instead of hanging in a
#                                   degraded state; lets Docker restart policy take over [2][21][22]
#
# --enable-native-access=ALL-UNNAMED — required by CloudSimPlus on Java 21
ENTRYPOINT ["java", \
  "-XX:+UseContainerSupport", \
  "-XX:InitialRAMPercentage=50.0", \
  "-XX:MaxRAMPercentage=75.0", \
  "-XX:+UseG1GC", \
  "-XX:+ExitOnOutOfMemoryError", \
  "--enable-native-access=ALL-UNNAMED", \
  "-jar", "gateway.jar"]