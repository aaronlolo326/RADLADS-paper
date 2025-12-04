#!/bin/bash
export USERNAME=$(whoami)
# ---------- config ----------
# export BASE="${1:-/work/${USERNAME}/data/TopK}"
export BASE="${1:-/work/${USERNAME}}/radlads"

MODEL_NAME="L28-D3584-qwerky7_qwen2-4" # GatedDeltaNet-1B7-ckpt36k"
export MODEL_DIR="${2:-${BASE}/out/hf/${3:-${MODEL_NAME}}}"

# a name you pick up for the experiment, it used to distinguish output directory name
export MAIN_PROCESS_PORT=29503

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #1,3,5 #0,1,2,3,4,5,6,7 #0,1,3,4,5,6,7

MODEL=${MODEL_DIR}


logs_dir="logs/logs_1201"
mkdir ${logs_dir}

# All possible config
#CONFIG_TO_RUN=("longbench_128" "niah_all_128" "harness_all") #

is_true() {
    case "${1:-}" in
        1|true|TRUE|True|yes|YES|Yes|y|Y|on|ON) return 0 ;;
        0|false|FALSE|False|no|NO|No|n|N|off|OFF) return 1 ;;
        *) return 1 ;;
    esac
}

# returns 0 if results_*.json exists under the dir, else 1
has_results_json() {
    local dir="$1"
    [[ -d "$dir" ]] || return 1
    find "$dir" -type f -name 'results_*.json' -print -quit | grep -q .
}

# sync_modeling() {
#     local local run_flag="${1:-true}"
#     if is_true "$run_flag"; then
#         echo "Executing sync_modeling..."
#     python utils/sync_modeling.py ${MODEL_DIR} ${CODE_DIR} ${SYNC_MODEL_NAME} ${CONFIG_PATH}
#     else
#         echo "Skipping sync_modeling"
#     fi
# }

# Return 0 if $1 is in the remaining arguments; else 1.
in_list() {
    local needle=$1; shift
    local e
    for e in "$@"; do
        [[ "$e" == "$needle" ]] && return 0
    done
    return 1
}

# for mode in "${MODES[@]}"; do
CONFIG_PATH="${6:-configs/config_gated_deltanet_1.3b_${MODEL_CONFIG_NAME}_${mode}.json}"
RUN_SYNC_MODELING="true"


# Comma-separated input can come from $6 or RUNCONFIG; fallback default:
CONFIG_CSV="${7:-${RUNCONFIG:-longbench_128,niah_all_128,harness_all}}"

# Split on commas -> array, then trim whitespace and drop empties
IFS=',' read -r -a _raw_items <<< "$CONFIG_CSV"
CONFIG_TO_RUN=()
for x in "${_raw_items[@]}"; do
    # trim leading whitespace
    x="${x#"${x%%[![:space:]]*}"}"
    # trim trailing whitespace
    x="${x%"${x##*[![:space:]]}"}"
    [[ -n "$x" ]] && CONFIG_TO_RUN+=("$x")
done

# Print each item on its own line (your “converted to a list” view)
printf '%s\n' "${CONFIG_TO_RUN[@]}"

########################################
# EDIT ME — SYNC MODELING
########################################
# Function: run sync_modeling conditionally based on boolean arg
# Usage:
#   sync_modeling true
#   sync_modeling "$RUN_BLOCK"

# if is_true "$RUN_SYNC_MODELING"; then
#     echo "Executing sync_modeling..."
#     python utils/sync_modeling.py ${MODEL_DIR} ${CODE_DIR} ${SYNC_MODEL_NAME} ${CONFIG_PATH} --verbose
# fi
# ----------------------------



# --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa \
# --batch_size auto \
TASK="harness_6tasks_padded"
export OUTPUT_DIR="${MODEL_DIR}/evals/${TASK}_${mode}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch \
    --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL_DIR},trust_remote_code=True,dtype=bfloat16 \
    --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa \
    --device cuda \
    --trust_remote_code \
    --batch_size ${bs} \
    --output_path $OUTPUT_DIR \
    --log_samples \
    --trust_remote_code

    
# TASK="longbench_narrativeqa_16"
# if in_list "${TASK}" "${CONFIG_TO_RUN[@]}"; then
# export OUTPUT_DIR="${MODEL_DIR}/evals/${TASK}"
# #  if has_results_json "$OUTPUT_DIR"; then
# #    echo "[$TASK] results already present in $OUTPUT_DIR — skipping."
# #  else
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
# --model hf \
# --model_args pretrained=${MODEL_DIR},attn_implementation="flash_attention_2",trust_remote_code=True,dtype=bfloat16,max_length=32768 \
# --tasks longbench_narrativeqa \
# --device cuda \
# --trust_remote_code \
# --num_fewshot 0 \
# --batch_size auto \
# --output_path $OUTPUT_DIR \
# --log_samples \
# --gen_kwargs '{"max_new_tokens": 16}' \
# --seed 1234
# #  fi
# fi

# TASK="longbench_16"
# if in_list "${TASK}" "${CONFIG_TO_RUN[@]}"; then
# TASK="longbench_16"
# export OUTPUT_DIR="${MODEL_DIR}/evals/${TASK}"
# if has_results_json "$OUTPUT_DIR"; then
#     echo "[$TASK] results already present in $OUTPUT_DIR — skipping."
# else
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
# --model hf \
# --model_args pretrained=${MODEL_DIR},attn_implementation="flash_attention_2",trust_remote_code=True,dtype=bfloat16,max_length=32768 \
# --tasks longbench \
# --device cuda \
# --num_fewshot 0 \
# --batch_size auto \
# --output_path $OUTPUT_DIR \
# --log_samples \
# --gen_kwargs '{"max_new_tokens": 16}' \
# --seed 1234
# fi
# fi

# TASK="longbench_128"
# if in_list "${TASK}" "${CONFIG_TO_RUN[@]}"; then

# export OUTPUT_DIR="${MODEL_DIR}/evals/${TASK}"
# if has_results_json "$OUTPUT_DIR"; then
#     echo "[$TASK] results already present in $OUTPUT_DIR — skipping."
# else
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
# --model hf \
# --model_args pretrained=${MODEL_DIR},attn_implementation="flash_attention_2",trust_remote_code=True,dtype=bfloat16,max_length=32768 \
# --tasks longbench \
# --trust_remote_code \
# --device cuda \
# --num_fewshot 0 \
# --batch_size 4 \
# --output_path $OUTPUT_DIR \
# --log_samples \
# --gen_kwargs '{"max_new_tokens": 128}' \
# --seed 1234
# fi
# fi

# TASK="niah_single_1_16"
# if in_list "${TASK}" "${CONFIG_TO_RUN[@]}"; then

# export OUTPUT_DIR="${MODEL_DIR}/evals/${TASK}"
# if has_results_json "$OUTPUT_DIR"; then
#     echo "[$TASK] results already present in $OUTPUT_DIR — skipping."
# else
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
# --model hf \
# --model_args pretrained=${MODEL_DIR},attn_implementation="flash_attention_2",trust_remote_code=True,dtype=bfloat16,max_length=32768 \
# --tasks niah_single_1 \
# --metadata='{"max_seq_lengths":[4096,8192,16384,32768]}' \
# --device cuda \
# --num_fewshot 0 \
# --batch_size auto \
# --output_path $OUTPUT_DIR \
# --log_samples \
# --gen_kwargs '{"max_new_tokens": 16}' \
# --seed 1234
# fi
# fi

# TASK="niah_multikey_3_128"
# if in_list "${TASK}" "${CONFIG_TO_RUN[@]}"; then
# export OUTPUT_DIR="${MODEL_DIR}/evals/${TASK}"
# if has_results_json "$OUTPUT_DIR"; then
#     echo "[$TASK] results already present in $OUTPUT_DIR — skipping."
# else
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
# --model hf \
# --model_args pretrained=${MODEL_DIR},attn_implementation="flash_attention_2",trust_remote_code=True,dtype=bfloat16,max_length=32768 \
# --tasks niah_multikey_3 \
# --metadata='{"max_seq_lengths":[4096,8192,16384,32768]}' \
# --device cuda \
# --trust_remote_code \
# --num_fewshot 0 \
# --batch_size auto \
# --output_path $OUTPUT_DIR \
# --log_samples \
# --gen_kwargs '{"max_new_tokens": 128}' \
# --seed 1234
# fi
# fi


# TASK="niah_all_128"
# if in_list "${TASK}" "${CONFIG_TO_RUN[@]}"; then
# export OUTPUT_DIR="${MODEL_DIR}/evals/${TASK}"
# if has_results_json "$OUTPUT_DIR"; then
#     echo "[$TASK] results already present in $OUTPUT_DIR — skipping."
# else
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
# --model hf \
# --model_args pretrained=${MODEL_DIR},attn_implementation="flash_attention_2",trust_remote_code=True,dtype=bfloat16,max_length=32768 \
# --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3 \
# --metadata='{"max_seq_lengths":[4096,8192,16384,32768]}' \
# --device cuda \
# --trust_remote_code \
# --batch_size 4 \
# --output_path $OUTPUT_DIR \
# --log_samples \
# --gen_kwargs '{"max_new_tokens": 128}' \
# --seed 1234
# fi
# fi


# TASK="harness_all"
# if in_list "${TASK}" "${CONFIG_TO_RUN[@]}"; then
#     export OUTPUT_DIR="${MODEL_DIR}/evals/${TASK}"
#     if has_results_json "$OUTPUT_DIR"; then
#         echo "[$TASK] results already present in $OUTPUT_DIR — skipping."
#     else
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch \
#             --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
#             --model hf \
#             --model_args pretrained=${MODEL_DIR},trust_remote_code=True,dtype=bfloat16,max_length=32768 \
#             --tasks arc_easy,arc_challenge,gsm8k,mathqa,piqa,race,winogrande,mmlu \
#             --device cuda \
#             --trust_remote_code \
#             --batch_size 8 \
#             --output_path $OUTPUT_DIR \
#             --log_samples \
#             --trust_remote_code
#     fi
# fi

# TASK="arc_challenge"
# if in_list "${TASK}" "${CONFIG_TO_RUN[@]}"; then
# export OUTPUT_DIR="${MODEL_DIR}/evals/${TASK}"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
# --model hf \
# --model_args pretrained=${MODEL_DIR},attn_implementation="flash_attention_2",trust_remote_code=True,dtype=bfloat16,max_length=32768 \
# --tasks arc_challenge \
# --device cuda \
# --trust_remote_code \
# --batch_size 1 \
# --output_path $OUTPUT_DIR \
# --log_samples \
# --limit 100 \
# --trust_remote_code
# fi



# if in_list "${TASK}" "${CONFIG_TO_RUN[@]}"; then
#     export OUTPUT_DIR="${MODEL_DIR}/evals/${TASK}"
#     # if has_results_json "$OUTPUT_DIR"; then
#     #     echo "[$TASK] results already present in $OUTPUT_DIR — skipping."
#     # else
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch \
#         --main_process_port ${MAIN_PROCESS_PORT} -m lm_eval \
#         --model hf \
#         --model_args pretrained=${MODEL_DIR},attn_implementation="flash_attention_2",trust_remote_code=True,dtype=bfloat16 \
#         --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa \
#         --device cuda \
#         --trust_remote_code \
#         --batch_size auto \
#         --output_path $OUTPUT_DIR \
#         --log_samples \
#         --trust_remote_code
#     fi
# fi



# niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3


# TODO: vllm gives lower results then hugginface, stay with Hugginface for now
#VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} lm_eval --model vllm \
#    --model_args pretrained=${MODEL_DIR},dtype=bfloat16,gpu_memory_utilization=0.95,data_parallel_size=8,max_length=32768 \
#    --tasks longbench_narrativeqa \
#    --batch_size auto \
#    --output_path $OUTPUT_DIR \
#    --log_samples \
#    --seed 1234 \
#    --gen_kwargs '{"max_gen_toks": 16}'

# TODO: test on harness
# ,max_model_len=32768
#
# --limit 10 \
#
#hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa
# mmlu,race,pubmedqa