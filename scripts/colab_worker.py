import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
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
    # 1. Leer argumentos desde CLI o desde variables de entorno
    args = sys.argv[1:]
    shard_id = os.environ.get("SHARD_ID")
    num_shards = os.environ.get("NUM_SHARDS")
    slice_arg = os.environ.get("SLICE")

    if not args:
        if slice_arg:
            args = [f"--slice {slice_arg}"]
        elif shard_id is not None and num_shards is not None:
            args = [f"--shard-id {shard_id}", f"--num-shards {num_shards}"]

    arg_str = " ".join(args)
    print(f"Colab worker starting with resolved arguments: '{arg_str}'", flush=True)

    # 2. Clonar o actualizar el repositorio en Colab
    if not os.path.exists("d2ql"):
        run_cmd("git clone -b run_exp_colab https://github.com/Markuves/d2ql.git d2ql")
    else:
        run_cmd("cd d2ql && git fetch origin run_exp_colab && git checkout run_exp_colab && git pull origin run_exp_colab")

    # 3. Limpiar CRLF si existiera
    run_cmd("cd d2ql && sed -i 's/\\r$//' scripts/*.sh scripts/*.py && chmod +x scripts/*.sh")

    # 4. Ejecutar el colab_runner.sh
    run_cmd(f"cd d2ql && bash -x scripts/colab_runner.sh {arg_str}")

if __name__ == "__main__":
    main()
