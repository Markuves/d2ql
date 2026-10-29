import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, executable="/bin/bash")
    if res.returncode != 0:
        print(f"Command failed with code {res.returncode}", file=sys.stderr, flush=True)
        sys.exit(res.returncode)

def main():
    args = sys.argv[1:]
    arg_str = " ".join(args)
    print(f"Colab worker starting with arguments: {arg_str}", flush=True)

    # 1. Clonar el repositorio y checkout a run_exp_colab
    if not os.path.exists("d2ql"):
        run_cmd("git clone -b run_exp_colab https://github.com/Markuves/d2ql.git d2ql")
    else:
        run_cmd("cd d2ql && git fetch origin run_exp_colab && git checkout run_exp_colab && git pull origin run_exp_colab")

    # 2. Diagnosticar archivos en el repo clonado
    run_cmd("cd d2ql && ls -la scripts/")

    # 3. Asegurar que colab_runner.sh tenga terminaciones de línea Unix (LF)
    run_cmd("cd d2ql && sed -i 's/\\r$//' scripts/colab_runner.sh && chmod +x scripts/colab_runner.sh")

    # 4. Ejecutar el colab_runner.sh con bash explícito y capturar salida
    run_cmd(f"cd d2ql && bash -x scripts/colab_runner.sh {arg_str}")

if __name__ == "__main__":
    main()
