import os
import mlflow
import mlflow.sklearn
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import (
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
repo_name = "ModernBert_NLI_sentence_bert_v04"

# Set up MLflow
remote_server_uri = "https://polar-mlflow.x2bee.com/"
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment(repo_name)

username = api.whoami()["name"]
repo_id = f"x2bee/{repo_name}"
dataset_repo = "x2bee/Korean_NLI_dataset"
api.create_repo(repo_id=repo_id, exist_ok=True)

hgf_path = "x2bee/ModernBert_MLM_kotoken_v03"
model = SentenceTransformer(hgf_path, device="cuda")

dataset = load_dataset(dataset_repo, data_dir="pair-score")
dataset = dataset["train"]

dataset_dict = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

loss = CoSENTLoss(model)

args = SentenceTransformerTrainingArguments(
        output_dir=f"/workspace/result/{repo_name}",
        dataloader_drop_last=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        warmup_ratio=0.3,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        learning_rate=4e-5,
        save_strategy="steps",
        save_steps=2500,
        eval_strategy="steps",
        eval_steps=2500,
        logging_steps=250,
        push_to_hub=True,
        hub_model_id=repo_id,
        hub_strategy="every_save",
        optim="adamw_torch",
        do_train=True,
        do_eval=True,
    )

dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
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

with mlflow.start_run(run_name="SentenceTransformer_Training") as run:
    mlflow.log_param("model_name", model.__class__.__name__)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("batch_size", args.per_device_train_batch_size)
    mlflow.log_param("gradient_accumulation", args.gradient_accumulation_steps)
    mlflow.log_param("num_epochs", args.num_train_epochs)
    mlflow.log_param("warmup_ratio", args.warmup_ratio)
    mlflow.log_param("train_size", len(train_dataset))
    mlflow.log_param("eval_size", len(eval_dataset))
    mlflow.log_param("seed", 42)
    mlflow.log_param("loss_function", loss.__class__.__name__)

    start_time = datetime.now()
    print(f"Training started at: {start_time}")
    trainer.train()
    end_time = datetime.now()
    print(f"Training ended at: {end_time}")
    trainer.push_to_hub(repo_id)

    # 학습 후 평가
    dev_evaluator(model)

    # 모델 평가 결과 로깅
    eval_score = dev_evaluator(model)

    # 모델 저장 및 로깅
    model_save_path = f"/workspace/result/{repo_name}_mlflow"
    model.save(model_save_path)
    mlflow.log_artifacts(model_save_path, artifact_path="model_files")

    # MLflow에 푸쉬 정보 추가
    mlflow.log_param("hub_repo_id", repo_id)
    
    # MLflow에 관련된 추가 정보 로깅
    mlflow.set_tag("framework", "sentence-transformers")
    mlflow.set_tag("experiment", "NLI sentence transformer training")
    mlflow.log_artifact(args.output_dir, artifact_path="training_output")

print("Training completed and logged in MLflow.")

trainer.push_to_hub(repo_id)