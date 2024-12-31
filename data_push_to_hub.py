import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from huggingface_hub import HfApi, login

api = HfApi()
with open('./api_key/HGF_TOKEN.txt', 'r') as hgf:
    login(token=hgf.read())
repo_name = "test_ModernBert_MLM"
username = api.whoami()["name"]
repo_id = f"x2bee/{repo_name}"
api.create_repo(repo_id=repo_id, exist_ok=True)

hgf_path = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained("/workspace/result/new_tokenizer_2")
tokenizer.push_to_hub("x2bee/test_ModernBert_MLM")

dataset = load_dataset('text', data_files={'train': '/workspace/data/corpus_data/korean_train_corpus.txt'})
dataset = dataset['train']
dataset.push_to_hub("x2bee/Korean_wiki_corpus")

api.create_repo(repo_id=repo_id, exist_ok=True)