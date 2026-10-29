import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"\n>>> Running: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"Command failed with code {res.returncode}", file=sys.stderr, flush=True)
        sys.exit(res.returncode)

def main():
    args = sys.argv[1:]
    arg_str = " ".join(args)
    print(f"Colab worker starting with arguments: {arg_str}")

    # 1. Clonar el repositorio y checkout a run_exp_colab
    if not os.path.exists("d2ql"):
        run_cmd("git clone -b run_exp_colab https://github.com/Markuves/d2ql.git d2ql")
    else:
        run_cmd("git -C d2ql fetch && git -C d2ql checkout run_exp_colab && git -C d2ql pull")

    # 2. Ejecutar el colab_runner.sh dentro de d2ql
    run_cmd(f"cd d2ql && chmod +x scripts/colab_runner.sh && bash scripts/colab_runner.sh {arg_str}")

if __name__ == "__main__":
    main()
