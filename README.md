---
base_model: answerdotai/ModernBERT-base
---

# ModernBert_MLM_kotoken_v03
This model is a fine-tuned version of [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on [x2bee/Korean_wiki_corpus](https://huggingface.co/datasets/x2bee/Korean_wiki_corpus) and [x2bee/Korean_namuwiki_corpus](https://huggingface.co/datasets/x2bee/Korean_namuwiki_corpus). <br>
It achieves the following results on the evaluation set:
- Loss: 1.5698

### Example Use.

```bash
git clone https://github.com/X2bee/prj_ecellm.git
```

```python
import os
os.chdir("/workspace")
from models.bert_mlm import ModernBertMLM

test_model = ModernBertMLM(model_id="x2bee/ModernBert_MLM_kotoken_v03")

text = "30일 전남 무안국제[MASK] 활주로에 전날 발생한 제주항공 [MASK] 당시 기체가 [MASK]착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 [MASK]과 승무원 181명 중 179명이 숨지고 [MASK]는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다."
result = test_model.modern_bert_convert_with_multiple_masks(text, top_k=5)
result
```

### Output
```
Predicted: 공항 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 [MASK] 당시 기체가 [MASK]착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 [MASK]과 승무원 181명 중 179명이 숨지고 [MASK]는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 사고 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 [MASK]착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 [MASK]과 승무원 181명 중 179명이 숨지고 [MASK]는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 비상 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 [MASK]과 승무원 181명 중 179명이 숨지고 [MASK]는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 승객 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 [MASK]는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 잔해 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 훼손 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 사고 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 발생 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 주장 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 내부 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 내부에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: ESP | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 내부에 설치된 ESP(착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 사고 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 내부에 설치된 ESP(착륙 유도 안전시설)가 사고를 키웠다는 [MASK]이 나오고 있다.
Predicted: 주장 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 내부에 설치된 ESP(착륙 유도 안전시설)가 사고를 키웠다는 주장이 나오고 있다.

```

### Compare with Real Value
```
Answer: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 대참사 당시 기체가 동체착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 항공기는 형체를 알아볼 수 없이 파손됐다. 피해 규모와 사고 원인 등에 대해 다양한 의문점이 제기되고 있는 가운데 활주로에 설치된 로컬라이저(착륙 유도 안전시설)가 피해를 키웠다는 지적이 나오고 있다.
Output: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 내부에 설치된 ESP(착륙 유도 안전시설)가 사고를 키웠다는 주장이 나오고 있다.

```

# plateer_classifier_ModernBERT_v01
This model is a fine-tuned version of [x2bee/ModernBert_MLM_kotoken_v01](https://huggingface.co/x2bee/ModernBert_MLM_kotoken_v01) on [x2bee/plateer_category_data](https://huggingface.co/datasets/x2bee/plateer_category_data). <br>
It achieves the following results on the evaluation set:
- Loss: 0.3379

### Example Use.
```python
import os
os.chdir("/workspace")
from models.bert_cls import plateer_classifier;

result = plateer_classifier("겨울 등산에서 사용할 옷")[0]
print(result)

############# result
-----------Category-----------
{'label': 2, 'label_decode': '기능성의류', 'score': 0.9214227795600891}
{'label': 8, 'label_decode': '스포츠', 'score': 0.07054771482944489}
{'label': 15, 'label_decode': '패션/의류/잡화', 'score': 0.0036312134470790625}
```

# Fine-Tune Example Code (Sentence-Transformer)

```python
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator, EmbeddingSimilarityEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, CosineSimilarityLoss
from sentence_transformers.training_args import BatchSamplers
from huggingface_hub import HfApi, login

api = HfApi()
with open('./api_key/HGF_TOKEN.txt', 'r') as hgf:
    login(token=hgf.read())
repo_name = "ModernBert_STS_sentence_bert_v01"
username = api.whoami()["name"]
repo_id = f"x2bee/{repo_name}"
dataset_repo = "x2bee/Korean_STS_all"
api.create_repo(repo_id=repo_id, exist_ok=True)

# 기존 MLM 방식으로 Korean Data를 학습한 모델을 Load.
# PreTraining된 Model을 로드하여 SentenceTransformer Class로 불러옴.
hgf_path = "x2bee/ModernBert_MLM_kotoken_v03"
model = SentenceTransformer(hgf_path, device="cuda")

# Data의 경우 STS 데이터를 사용. STS는 Sentence, Pair, Label(Score)로 된 데이터로, 두 문장의 유사도를 점수로 표현한 데이터임.
# 이러한 데이터에 맞추어 Loss Function을 제대로 지정해 줄 필요가 있음.
dataset = load_dataset(dataset_repo)
dataset = dataset["train"]

# CosineSimilarityLoss Class가 0~1로 된 Label을 요구하기 때문에 변경.
# 0~5점으로 표현된 데이터라, 단순하게 5로 나눠버림.
def normalize_label(example):
    example["label"] = (example["label"] / 5)
    return example
dataset = dataset.map(normalize_label)

# 데이터 가져와서 split.
dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

# Loss 정의의
loss = CosineSimilarityLoss(model)

# Training Argument. 다른 것들과 비슷하게.
args = SentenceTransformerTrainingArguments(
        output_dir=f"/workspace/result/{repo_name}",
        num_train_epochs=3,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        learning_rate=2e-5,
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        save_total_limit=2,
        logging_steps=500,
        push_to_hub=True,
        hub_model_id=repo_id,
        hub_strategy="every_save",
        do_train=True,
        do_eval=True,
    )

# 주어진 Function에 맞추어 Evaluator 설정.
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence"],
    sentences2=eval_dataset["pair"],
    scores=eval_dataset["label"],
    name="sts_dev",
)

# 시작 전 검증해보기/
dev_evaluator(model)

# Trainer 설정.
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)

# 학습.
trainer.train()

dev_evaluator(model)

trainer.push_to_hub()
```


# Fine-Tune Example Code (Classification)
```python
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from huggingface_hub import HfApi, login
import torch.distributed as dist

if dist.is_initialized():
    dist.destroy_process_group()
    
api = HfApi()
with open('./api_key/HGF_TOKEN.txt', 'r') as hgf:
    login(token=hgf.read())
repo_name = "plateer_classifier_ModernBERT_v02"
username = api.whoami()["name"]
repo_id = f"x2bee/{repo_name}"
api.create_repo(repo_id=repo_id, exist_ok=True)

# Label의 갯수에 따라 AutoModel로 Load.
model_id = "x2bee/ModernBert_MLM_kotoken_v03"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=17)
model.to("cuda")

# 주어진 Dataset에 따라 설정.
# 여기서는 총 500만개만 사용.
def is_valid_str(example):
    return isinstance(example['goods_nm'], str)
selected_columns = ['goods_nm', 'label']
dataset_name = "x2bee/plateer_category_data"
dataset = load_dataset(dataset_name, data_dir="data")
dataset = dataset['train'].select_columns(selected_columns)
dataset = dataset.shuffle().select(range(5000000))
dataset = dataset.filter(is_valid_str)

test_size = 0.1
test_split_seed = 42

split_dataset = dataset.train_test_split(test_size=test_size, seed=test_split_seed)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# 학습 전 주어진 Tokenizer로 Input을 토큰화.
def encode(examples):
    return tokenizer(examples['goods_nm'], padding="max_length", truncation=True, max_length=512)

# 텍스트 토큰 작업.
tokenized_train_dataset = train_dataset.map(encode, batched=True, num_proc=os.cpu_count())
tokenized_test_dataset = test_dataset.map(encode, batched=True, num_proc=os.cpu_count())

# DataCollator 생성 (토크나이저와 모델에 맞는 마스크 생성)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer 설정
training_args = TrainingArguments(
    output_dir="/workspace/result/modern_bert",
    overwrite_output_dir=True,
    ddp_find_unused_parameters=True,
    save_strategy="steps",
    save_steps=5000,
    eval_strategy="steps",
    eval_steps=5000,
    learning_rate=2e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="/workspace/result/logs",
    logging_steps=250,
    warmup_steps=10000,
    gradient_accumulation_steps=4,
    load_best_model_at_end=True,
    optim="adamw_torch",
    report_to="none",
    do_train=True,
    do_eval=True,
    dataloader_num_workers=(os.cpu_count() // 2),
    push_to_hub=True,
    hub_model_id=repo_id,
    hub_strategy="checkpoint",
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
api.create_repo(repo_id=repo_id, exist_ok=True)
trainer.push_to_hub()
```

