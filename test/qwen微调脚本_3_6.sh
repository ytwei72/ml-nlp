
# 指令监督微调SFT脚本
# 注意model_name_or_path 基础模型路径要正确
# 注意output_dir 要自定义
# 注意dataset 要改成自己的数据集
# 重点关注learning_rate、num_train_epochs等超参设置
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path  /home/ai1/LLaMA-Factory/Qwen-1_8B-Chat \
    --finetuning_type lora \
    --template default \
    --dataset_dir data \
    --dataset medical \
    --cutoff_len 1024 \
    --learning_rate 1e-03 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --output_dir ysl/Qwen-1.8B/lora/train_medical_sz \
    --fp16 True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target c_attn \
    --plot_loss True


# 查看中间模型运行效果
# model_name_or_path 基础模型路径要正确
# adapter_name_or_path 中间模型路径
python src/cli_demo.py \
    --model_name_or_path /home/ai1/LLaMA-Factory/Qwen-1_8B-Chat \
    --adapter_name_or_path ysl/Qwen-1.8B/lora/train_medical_sz \
    --template default \
    --finetuning_type lora

# 模型导出
python src/export_model.py \
    --model_name_or_path /home/ai1/LLaMA-Factory/Qwen-1_8B-Chat \
    --adapter_name_or_path ysl/Qwen-1.8B/lora/train_medical_sz \
    --template default \
    --finetuning_type lora \
    --export_dir ysl/output_model \
    --export_size 2 \
    --export_legacy_format False

# 奖励模型训练
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path /home/ai1/LLaMA-Factory/Qwen-1_8B-Chat \
    --adapter_name_or_path ysl/Qwen-1.8B/lora/train_medical_sz \
    --create_new_adapter \
    --dataset comparison_gpt4_zh \
    --template default \
    --finetuning_type lora \
    --lora_target c_attn \
    --output_dir ysl/Qwen-1.8B/lora/train_medical_rm \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-6 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16

#运行奖励模型训练后效果
python src/cli_demo.py \
    --model_name_or_path /home/ai1/LLaMA-Factory/Qwen-1_8B-Chat \
    --adapter_name_or_path ysl/Qwen-1.8B/lora/train_medical_rm \
    --template default \
    --finetuning_type lora


# PPO训练
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage ppo \
    --do_train \
    --model_name_or_path /home/ai1/LLaMA-Factory/Qwen-1_8B-Chat \
    --adapter_name_or_path ysl/Qwen-1.8B/lora/train_medical_sz \
    --create_new_adapter \
    --dataset alpaca_gpt4_zh \
    --template default \
    --finetuning_type lora \
    --lora_target c_attn \
    --reward_model ysl/Qwen-1.8B/lora/train_medical_rm \
    --output_dir ysl/Qwen-1.8B/lora/train_medical_ppo1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1000000 \
    --lr_scheduler_type cosine \
    --top_k 0 \
    --top_p 0.9 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate 1e-5 \
    --num_train_epochs 100.0 \
    --plot_loss \
    --fp16


# DPO训练
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path /home/ai1/LLaMA-Factory/Qwen-1_8B-Chat \
    --adapter_name_or_path ysl/Qwen-1.8B/lora/train_medical_sz \
    --create_new_adapter \
    --dataset comparison_gpt4_zh \
    --template default \
    --finetuning_type lora \
    --lora_target c_attn \
    --output_dir  ysl/Qwen-1.8B/lora/train_medical_dpo3 \
    --per_device_train_batch_size 2000000 \
    --gradient_accumulation_steps 40000000000 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate 1000000000000000000000 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16


