#!/usr/bin/env bash
set -euo pipefail

# Directorio para locks de GPUs
LOCK_DIR="/tmp/gpu_locks_$$"
mkdir -p "$LOCK_DIR"

# Función para limpiar locks al salir
cleanup() {
  rm -rf "$LOCK_DIR"
}
trap cleanup EXIT

# Función para intentar obtener lock de una GPU
try_lock_gpu() {
  local gpu_id=$1
  local lockfile="$LOCK_DIR/gpu_${gpu_id}.lock"
  
  if (set -C; echo $$ > "$lockfile") 2>/dev/null; then
    echo "[DEBUG] Lock obtenido para GPU${gpu_id}"
    return 0
  else
    echo "[DEBUG] GPU${gpu_id} ya está siendo usada por otro job en paralelo"
    return 1
  fi
}

# Función para liberar lock de una GPU
release_gpu_lock() {
  local gpu_id=$1
  local lockfile="$LOCK_DIR/gpu_${gpu_id}.lock"
  rm -f "$lockfile"
  echo "[DEBUG] Lock liberado para GPU${gpu_id}"
}

# Función para verificar si una GPU está siendo usada por ricoiban
check_gpu_usage() {
  local gpu_id=$1
  echo "[DEBUG] Verificando GPU${gpu_id}..."
  
  # Obtener procesos en la GPU
  local gpu_processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits -i ${gpu_id})
  echo "[DEBUG] Procesos en GPU${gpu_id}: ${gpu_processes}"
  
  if [ -z "$gpu_processes" ]; then
    echo "[DEBUG] GPU${gpu_id} está libre (sin procesos)"
    return 1  # GPU está libre
  fi
  
  if echo "$gpu_processes" | while IFS=, read -r uuid pid; do
    if [ -n "$pid" ]; then
      local user=$(ps -o user= -p $pid 2>/dev/null | tr -d ' ')
      echo "[DEBUG] PID $pid pertenece al usuario: '$user'"
      if [ "$user" = "ricoiban" ]; then
        echo "[DEBUG] Encontrado proceso de ricoiban (PID: $pid) en GPU${gpu_id}"
        exit 0
      fi
    fi
  done; then
    echo "[DEBUG] GPU${gpu_id} está siendo usada por ricoiban"
    return 0  # GPU está siendo usada por ricoiban
  else
    echo "[DEBUG] GPU${gpu_id} está libre o usada por otro usuario"
    return 1  # GPU está libre o usada por otro usuario
  fi
}

# Función para lanzar un job en GPU primaria y, si falla, reintentar en GPU3 o GPU secundaria
run_with_failover() {
  local primary_gpu=$1
  local fallback_gpu=$2
  local third_gpu=$3
  local cmd="$4"
  local used_gpu=""

  while true; do
    # Verificar si la GPU primaria está siendo usada por ricoiban y si podemos obtener el lock
    if check_gpu_usage ${primary_gpu}; then
      echo "[$(date +'%Y-%m-%d %H:%M:%S')] GPU${primary_gpu} está siendo usada por ricoiban, intentando GPU${third_gpu}..."
    elif try_lock_gpu ${primary_gpu}; then
      echo "[$(date +'%Y-%m-%d %H:%M:%S')] Lanzando en GPU${primary_gpu}: ${cmd}"
      used_gpu=${primary_gpu}
      CUDA_VISIBLE_DEVICES=${primary_gpu} bash -c "${cmd}"
      local status=$?
      release_gpu_lock ${primary_gpu}

      if [ $status -eq 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job completado exitosamente en GPU${primary_gpu}"
        break
      fi
    fi

    # Verificar si GPU3 está siendo usada por ricoiban y si podemos obtener el lock
    if check_gpu_usage ${third_gpu}; then
      echo "[$(date +'%Y-%m-%d %H:%M:%S')] GPU${third_gpu} está siendo usada por ricoiban, intentando GPU${fallback_gpu}..."
    elif try_lock_gpu ${third_gpu}; then
      echo "[$(date +'%Y-%m-%d %H:%M:%S')] Lanzando en GPU${third_gpu}: ${cmd}"
      used_gpu=${third_gpu}
      CUDA_VISIBLE_DEVICES=${third_gpu} bash -c "${cmd}"
      status=$?
      release_gpu_lock ${third_gpu}

      if [ $status -eq 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job completado exitosamente en GPU${third_gpu}"
        break
      fi
    fi

    # Verificar si la GPU de fallback está siendo usada por ricoiban y si podemos obtener el lock
    if check_gpu_usage ${fallback_gpu}; then
      echo "[$(date +'%Y-%m-%d %H:%M:%S')] Todas las GPUs están siendo usadas por ricoiban, esperando 30 segundos..."
    elif try_lock_gpu ${fallback_gpu}; then
      echo "[$(date +'%Y-%m-%d %H:%M:%S')] Lanzando en GPU${fallback_gpu}: ${cmd}"
      used_gpu=${fallback_gpu}
      CUDA_VISIBLE_DEVICES=${fallback_gpu} bash -c "${cmd}"
      status=$?
      release_gpu_lock ${fallback_gpu}

      if [ $status -eq 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job completado exitosamente en GPU${fallback_gpu}"
        break
      fi
    else
      echo "[$(date +'%Y-%m-%d %H:%M:%S')] Todas las GPUs están ocupadas (ricoiban o jobs paralelos), esperando 30 segundos..."
    fi

    sleep 30
  done
}

# 2. Tell bitsandbytes to use the CUDA-12.5 binary
export BNB_CUDA_VERSION=125          # for this shell only
# or add the same line to ~/.bashrc for permanence

# 3. (only if you have a system CUDA toolkit in /usr/local/cuda-12.5)
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/cuda-12.5/lib64

# 4. Optimize HuggingFace cache for faster I/O
# Check if we have space in /tmp for faster cache
TMP_SPACE=$(df /tmp --output=avail -B 1G | tail -n1)
if [ "$TMP_SPACE" -gt 50 ]; then
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] Using /tmp for HF cache (faster I/O)"
  export HF_HOME="/tmp/hf_cache_ricoiban_$$"
  export TRANSFORMERS_CACHE="/tmp/hf_cache_ricoiban_$$/transformers"
  export HF_DATASETS_CACHE="/tmp/hf_cache_ricoiban_$$/datasets"
  mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
else
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] Using home directory for HF cache"
fi

# 5. Optimize PyTorch for faster training
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUDA_LAUNCH_BLOCKING=0

# -------------------
# Fase 1: entrenamientos iniciales en paralelo
# -------------------
# Job1: pretraining OpenQA en GPU2 (dependencia para fase 2)
run_with_failover 3 5 2 "python main_pre-training.py \
  --dataset RikoteMaster/OpenQA_merged \
  --output_name model_openqa \
  --model_name Qwen/Qwen3-0.6B" &
pid_job1=$!

# Job2: pretraining MCQA en GPU5
run_with_failover 5 2 3 "python main_pre-training.py \
  --dataset jonlecumberri/MNLP_M2_mcqa_dataset_processed \
  --output_name model_mcqa \
  --model_name Qwen/Qwen3-0.6B" &

# Esperar a que ambos jobs terminen
wait ${pid_job1}
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job1 (model_openqa) completado"
wait
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job2 (model_mcqa) completado"
sleep 10  # dar tiempo a liberar memoria

# -------------------
# Fase 2: tareas que dependen de model_openqa
# -------------------
# Job3: pretraining MCQA sobre model_openqa
run_with_failover 2 5 3 "python main_pre-training.py \
  --dataset jonlecumberri/MNLP_M2_mcqa_dataset_processed \
  --output_name model_openqa_mcqa \
  --model_name RikoteMaster/model_openqa"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job3 (model_openqa_mcqa) completado"
sleep 10  # dar tiempo a liberar memoria

# Job4: fine-tuning LoRA sobre model_openqa
run_with_failover 5 2 3 "python main_lora.py \
  --model_name RikoteMaster/model_openqa \
  --output_name model_openqa_lora_mcqa"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job4 (model_openqa_lora_mcqa) completado"
sleep 10  # dar tiempo a liberar memoria

# Job5: fine-tuning LoRA base
run_with_failover 2 5 3 "python main_lora.py \
  --model_name Qwen/Qwen3-0.6B \
  --output_name model_base_lora"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job5 (model_base_lora) completado"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Todas las tareas finalizadas"

# Clean up temporary cache if we used /tmp
if [ -n "$HF_HOME" ] && [[ "$HF_HOME" == /tmp/* ]]; then
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] Limpiando caché temporal en /tmp..."
  rm -rf "$HF_HOME"
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] Caché temporal limpiada"
fi
