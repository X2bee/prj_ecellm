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
dataset_repo = "x2bee/Korean_namuwiki_corpus"
api.create_repo(repo_id=repo_id, exist_ok=True)

hgf_path = "x2bee/ModernBert_MLM_kotoken_v02"
model = SentenceTransformer(hgf_path, device="cuda")

dataset = load_dataset("x2bee/Korean_STS_all")
dataset = dataset["train"]

# label 값을 0~1로 정규화
def normalize_label(example):
    example["label"] = (example["label"] / 5)
    return example
dataset = dataset.map(normalize_label)

dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

loss = CosineSimilarityLoss(model)

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

dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence"],
    sentences2=eval_dataset["pair"],
    scores=eval_dataset["label"],
    name="sts_dev",
)

dev_evaluator(model)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

dev_evaluator(model)

trainer.push_to_hub()