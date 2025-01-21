import os
import mlflow
import mlflow.sklearn
from Korpora import Korpora
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import (
    models,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import TripletEvaluator, EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, CosineSimilarityLoss, CoSENTLoss
from sentence_transformers.training_args import BatchSamplers
from huggingface_hub import HfApi, login

api = HfApi()
with open('./api_key/HGF_TOKEN.txt', 'r') as hgf:
    login(token=hgf.read())
repo_name = "KoModernBERT_SBERT_compare_mlmlv2"

# Set up MLflow
remote_server_uri = "https://polar-mlflow.x2bee.com/"
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment(repo_name)

username = api.whoami()["name"]
repo_id = f"x2bee/{repo_name}"
dataset_repo = "x2bee/Korean_NLI_dataset"
api.create_repo(repo_id=repo_id, exist_ok=True)

hgf_path = "x2bee/ModernBert_MLM_kotoken_v02"
transformer = models.Transformer(hgf_path, max_seq_length=512)
pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="weightedmean")
dence = models.Dense(in_features=transformer.get_word_embedding_dimension(), out_features=768)
# normalize = models.Normalize()

# model = SentenceTransformer(hgf_path, device="cuda")
model = SentenceTransformer(modules=[transformer, pooling, dence], device="cuda")
trainable_params = [(name, param.requires_grad) for name, param in model.named_parameters()]
# print("############# trainable_params ###############")
# print(trainable_params)

dataset = load_dataset(dataset_repo, data_dir="snli_pair-score")
# dataset = load_dataset(dataset_repo)
dataset = dataset["train"]

dataset_dict = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

loss = CosineSimilarityLoss(model)

args = SentenceTransformerTrainingArguments(
        output_dir=f"/workspace/result/{repo_name}",
        dataloader_drop_last=True,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        warmup_ratio=0.3,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        learning_rate=1e-5,
        save_strategy="epoch",
        # save_steps=2500,
        eval_strategy="epoch",
        # eval_steps=2500,
        logging_steps=100,
        push_to_hub=True,
        hub_model_id=repo_id,
        hub_strategy="every_save",
        optim="adamw_torch",
        do_train=True,
        do_eval=True,
    )


corpus = Korpora.load("korsts", root_dir="/workspace/Korpora")
labels = [(float(label)/5) for label in corpus.test.labels]

dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=corpus.test.texts,
    sentences2=corpus.test.pairs,
    scores=labels,
    similarity_fn_names=['cosine', 'euclidean', 'manhattan', 'dot'],
    name="sts_dev",
)

start_result = dev_evaluator(model)
print(start_result)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

trainer.push_to_hub(repo_id)