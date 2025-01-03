import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from huggingface_hub import HfApi, login
import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()

api = HfApi()
with open('./api_key/HGF_TOKEN.txt', 'r') as hgf:
    login(token=hgf.read())
repo_name = "ModernBert_MLM_kotoken_v03"
username = api.whoami()["name"]
repo_id = f"x2bee/{repo_name}"
dataset_repo = "x2bee/Korean_wiki_corpus_all"
api.create_repo(repo_id=repo_id, exist_ok=True)

hgf_path = "x2bee/ModernBert_MLM_kotoken_v02"
tokenizer = AutoTokenizer.from_pretrained(hgf_path, subfolder="last-checkpoint")
model = AutoModelForMaskedLM.from_pretrained(hgf_path, subfolder="last-checkpoint")
model.to('cuda')

model.resize_token_embeddings(len(tokenizer))
dataset = load_dataset(dataset_repo)
dataset = dataset['train']

test_size = 0.1
test_split_seed = 42

split_dataset = dataset.train_test_split(test_size=test_size, seed=test_split_seed)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

print(f"Vocab Size: {len(tokenizer.get_vocab())}")
embedding_size = model.get_input_embeddings().weight.size(0)
print(f"Embedding size: {embedding_size}")
# print(model)

def encode(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# 텍스트를 토큰화
tokenized_train_dataset = train_dataset.map(encode, batched=True, num_proc=os.cpu_count())
tokenized_test_dataset = test_dataset.map(encode, batched=True, num_proc=os.cpu_count())

# DataCollator 생성 (토크나이저와 모델에 맞는 마스크 생성)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

# Trainer 설정
training_args = TrainingArguments(
    output_dir="/workspace/result/modern_bert",
    overwrite_output_dir=True,
    ddp_find_unused_parameters=True,
    save_strategy="steps",
    save_steps=10000,
    eval_strategy="steps",
    eval_steps=10000,
    learning_rate=1e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="/workspace/result/logs",
    logging_steps=250,
    warmup_steps=15000,
    gradient_accumulation_steps=4,
    load_best_model_at_end=True,
    optim="adamw_torch",
    report_to="none",
    do_train=True,
    do_eval=True,
    dataloader_num_workers=(os.cpu_count() // 2),
    push_to_hub=True,
    hub_model_id=repo_id,
    hub_strategy="every_save",
)

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    processing_class=tokenizer,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
)

# 학습 시작
trainer.train()
trainer.push_to_hub()
