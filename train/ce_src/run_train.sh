torchrun \
  --nproc_per_node=1 \
  train.py \
  --output_dir "/workspace/result/modernbert_reranker" \
  --model_name_or_path "BAAI/bge-reranker-large" \
  --train_data "CocoRoF/msmacro_triplet_ko" \
  --data_dir "triplet_ko" \
  --hf_model_token "" \
  --hf_data_token "" \
  --hub_token "" \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --dataloader_drop_last \
  --warmup_ratio 0.0 \
  --train_group_size 2 \
  --max_len 512 \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --save_steps 3000 \
  --save_total_limit 1 \
  --report_to "mlflow" \
  --push_to_hub True \
  --hub_model_id "CocoRoF/KoModernBERT-base-test" \
  --hub_strategy "checkpoint" \
  --max_grad_norm 1 \
  --ddp_find_unused_parameters False \
  --flash_attn False
  # --do_train True \
  # --do_eval True \