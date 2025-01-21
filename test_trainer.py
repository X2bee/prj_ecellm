import os
import mlflow
import torch
from torch.utils.data import IterableDataset, get_worker_info
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, get_scheduler, TrainingArguments, ModernBertForMaskedLM
from huggingface_hub import HfApi, login
from src.optimizer import StableAdamW
from src.sequence_packer import GreedyBestFitSequencePacker
from src.data_packer import PackedSequenceIterableDataset
import torch.distributed as dist

# if dist.is_initialized():
#     rank = dist.get_rank()
# else:
#     print("No is_initialized.")

api = HfApi()
with open('./api_key/HGF_TOKEN.txt', 'r') as hgf:
    login(token=hgf.read())
repo_name = "KoModernBERT-base-ckp00"
repo_id = f"x2bee/{repo_name}"
dataset_repo = "x2bee/newspaper_data"
api.create_repo(repo_id=repo_id, exist_ok=True)

# Set up MLflow
remote_server_uri = "https://polar-mlflow.x2bee.com/"
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment(repo_name)

hgf_path = "x2bee/KoModernBERT-base-ckp00"
tokenizer = AutoTokenizer.from_pretrained(hgf_path)
model = ModernBertForMaskedLM.from_pretrained(hgf_path)
model.to('cuda')

dataset = load_dataset(dataset_repo, data_dir="modu_bf2018_chunk_00")
dataset = dataset['train']

test_size = 0.01
test_split_seed = 42

split_dataset = dataset.train_test_split(test_size=test_size, seed=test_split_seed)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

def encode(examples):
    return tokenizer(examples['sentence'])

# 텍스트를 토큰화
tokenized_train_dataset = train_dataset.map(encode, batched=True, num_proc=os.cpu_count())
tokenized_test_dataset = test_dataset.map(encode, batched=True, num_proc=os.cpu_count())

packer = GreedyBestFitSequencePacker(
    src_iterable=1,      # 어차피 .src_iterable는 나중에 바꿀 것이므로 None
    src_batch_size=8,      # 실제로는 PackedSequenceIterableDataset에서 batchify 때 사용
    src_max_seq_len=1024,    # 혹은 1024 등
    out_batch_size=8,       # 한번에 몇 개 pseq를 만들 건지
    out_pseq_len=2048,      # micro_batch_size * max_seq_len 같은 식으로 계산
    buffer_size=256,        # 내부 버퍼 크기
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1,
    mask_token_id=50284,      # 예: [MASK] == 103 (BERT)
    ignore_token_id=-100,
    mask_prob=0.25,
    seed=42,
    suppress_masking=False,
    batch_size_warmup_min_size=None,   # 필요 시 설정
    batch_size_warmup_tokens=None,     # 필요 시 설정
    world_size=1,                      # 멀티 GPU 시에는 실제 world_size
)

train_iterable_dataset = PackedSequenceIterableDataset(
    hf_dataset=tokenized_train_dataset,
    packer=packer,
    src_batch_size=8,
)

test_iterable_dataset = PackedSequenceIterableDataset(
    hf_dataset=tokenized_test_dataset,
    packer=packer,
    src_batch_size=8,
)

# DataCollator 생성 (토크나이저와 모델에 맞는 마스크 생성)
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

# if rank == 0:
#     report = "mlflow"
# else:
#     report = "none"

# Trainer 설정
training_args = TrainingArguments(
    output_dir="/workspace/result/modern_bert",
    overwrite_output_dir=True,
    save_strategy="steps",
    save_steps=4000,
    eval_strategy="steps",
    eval_steps=4000,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="/workspace/result/logs",
    logging_steps=100,
    warmup_ratio=0.33,
    optim="adamw_torch",
    report_to="none",
    do_train=True,
    do_eval=True,
    save_only_model=False,
    dataloader_num_workers=(os.cpu_count() // 2),
    push_to_hub=True,
    hub_model_id=repo_id,
    hub_strategy="every_save",
)

optimizer = StableAdamW(
    params=model.parameters(),
    lr=1e-4, 
    betas=(0.9,0.999), 
    eps=1e-7, 
    weight_decay=0.01
)

num_update_steps_per_epoch = len(train_dataset) // training_args.per_device_train_batch_size
max_train_steps = num_update_steps_per_epoch * training_args.num_train_epochs
warmup_steps = int(max_train_steps * 0.1)

scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=max_train_steps,
)

def dummy_data_collator(features):
    # packer가 이미 batch dict를 완성했으므로 그대로 반환
    return features[0] 

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=dummy_data_collator,
    optimizers=(optimizer,scheduler),
    train_dataset=train_iterable_dataset,
    eval_dataset=test_iterable_dataset,
)

# if rank == 0:
#     with mlflow.start_run(run_name="SentenceTransformer_Training") as run:
#         mlflow.log_param("model_name", model.__class__.__name__)
#         mlflow.log_param("learning_rate", training_args.learning_rate)
#         mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
#         mlflow.log_param("gradient_accumulation", training_args.gradient_accumulation_steps)
#         mlflow.log_param("num_epochs", training_args.num_train_epochs)
#         mlflow.log_param("warmup_ratio", training_args.warmup_ratio)
#         mlflow.log_param("train_size", len(train_dataset))
#         mlflow.log_param("eval_size", len(test_dataset))
#         mlflow.log_param("seed", 42)

#         trainer.train()
        
# # 학습 시작
# else:
#     trainer.train()
    
# if rank == 0:
#     trainer.push_to_hub()

# dist.destroy_process_group()

trainer.train()