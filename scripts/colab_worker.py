import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Usar Popen para transmitir stdout y stderr en tiempo real
    process = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        print(f"\n[ERROR] Command failed with code {return_code}", file=sys.stderr, flush=True)
        sys.exit(return_code)

def main():
    args = sys.argv[1:]
    arg_str = " ".join(args)
    print(f"Colab worker starting with arguments: {arg_str}", flush=True)

    # 1. Clonar o actualizar el repositorio
    if not os.path.exists("d2ql"):
        run_cmd("git clone -b run_exp_colab https://github.com/Markuves/d2ql.git d2ql")
    else:
        run_cmd("cd d2ql && git fetch origin run_exp_colab && git checkout run_exp_colab && git pull origin run_exp_colab")

    # 2. Diagnóstico de entorno
    run_cmd("which bash && which sh && which git && which python3")

    # 3. Limpieza de saltos de línea CRLF en todo scripts/
    run_cmd("cd d2ql && sed -i 's/\\r$//' scripts/*.sh scripts/*.py && chmod +x scripts/*.sh")

    # 4. Ejecución del script
    run_cmd(f"cd d2ql && bash -x scripts/colab_runner.sh {arg_str}")

if __name__ == "__main__":
    main()
