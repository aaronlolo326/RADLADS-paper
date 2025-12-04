# # from topk2
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers==4.53.3 accelerate==1.7.0 datasets==3.6.0 liger-kernel==0.5.10 gpustat 
pip install flash-attn==2.7.4.post1 --no-build-isolation # added --no-build-isolation

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install lm-eval[longbench]
pip install lm-eval[ruler]

pip install python-Levenshtein
pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention --no-deps

# for radlads
# pip install lightning torch flash-linear-attention triton deepspeed wandb ninja --upgrade
pip install lightning triton deepspeed wandb ninja