DISABLE_VERSION_CHECK=1 python api.py \
 --model_name_or_path /data03/irlab_share/Qwen2.5-0.5B-Instruct/ \
 --template qwen \
 --infer_backend vllm \
 --vllm_gpu_util 0.2 \
 --vllm_enforce_eager true \
 --trust_remote_code true
