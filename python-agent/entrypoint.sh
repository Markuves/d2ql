#!/bin/sh
# libcuda.so MUST come from the host driver (/usr/local/nvidia), not from
# pip's nvidia-* wheels. Putting those dirs first makes nvidia-smi work
# (it uses the driver) while torch.cuda.is_available() fails.
driver_libs="/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu"
torch_libs=""
for d in /app/.venv/lib/python3.*/site-packages/nvidia/*/lib \
         /app/.venv/lib/python3.*/site-packages/nvidia/*/lib64; do
  [ -d "$d" ] || continue
  if [ -e "$d/libcuda.so" ] || [ -e "$d/libcuda.so.1" ]; then
    continue
  fi
  torch_libs="${d}:${torch_libs}"
done
export LD_LIBRARY_PATH="${driver_libs}:${torch_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-}"
ls -l /dev/nvidia* /dev/nvidia-uvm /dev/nvidia-caps 2>&1 | head -30

# Keep NVIDIA image setup (banner, ldconfig). Replacing it was hiding the GPU.
if [ -x /opt/nvidia/nvidia_entrypoint.sh ]; then
  exec /opt/nvidia/nvidia_entrypoint.sh "$@"
fi
exec "$@"
