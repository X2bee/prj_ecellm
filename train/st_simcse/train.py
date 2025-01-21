import logging
import os
import torch
import torch.distributed as dist
import mlflow

from huggingface_hub import HfApi, login
from transformers import HfArgumentParser, TrainingArguments, set_seed, get_scheduler
from sentence_transformers import (
    models,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CosineSimilarityLoss, MultipleNegativesRankingLoss, TripletLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import BatchSamplers

from arguments import ModelArguments, DataArguments
from data import STSrainDataLoader
from optimizer import StableAdamW

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if dist.is_initialized():
        rank = dist.get_rank()
        # print(f"rank = {rank}")
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://"
        )

        rank = dist.get_rank()
        # print(f"rank = {rank}")
        
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
        
    if model_args.hf_model_token:
        login(token=model_args.hf_model_token)

    if training_args.hub_model_id:
        api = HfApi()
        api.create_repo(repo_id=training_args.hub_model_id, exist_ok=True)
        
    if "mlflow" in training_args.report_to:
        remote_server_uri = "https://polar-mlflow.x2bee.com/"
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(training_args.hub_model_id)
        print(f"Report to {training_args.report_to}. Log MLFlow")
        
    if model_args.flash_attn:
        automodel_args = {
            "attn_implementation": "flash_attention_2", 
            "torch_dtype": torch.float16
        }
    else:
        automodel_args = {}
        
    transformer = models.Transformer(model_args.model_name_or_path, max_seq_length=512, model_args=automodel_args)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(), 
        pooling_mode="mean",
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )
    dence = models.Dense(in_features=transformer.get_word_embedding_dimension(), out_features=768)
    model = SentenceTransformer(modules=[transformer, pooling, dence], device="cuda")
    loss = TripletLoss(model)
    data_loader = STSrainDataLoader(data_args)
    train_dataset, eval_dataset = data_loader.prepare_samples()
    
    optimizer_parameter = {
        "lr": training_args.learning_rate, 
        "betas": (0.9,0.999),
        "eps": 1e-7,
        "weight_decay": 0.01,
    }
    
    num_update_steps_per_epoch = len(train_dataset) // training_args.per_device_train_batch_size
    max_train_steps = num_update_steps_per_epoch * training_args.num_train_epochs
    warmup_steps = int(max_train_steps * training_args.warmup_ratio)
    
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset['text'],
        sentences2=eval_dataset['pair'],
        scores=eval_dataset['label'],
        similarity_fn_names=['cosine', 'euclidean', 'manhattan', 'dot'],
        name="sts_dev",
    )

    model.fit(
        train_objectives=[(train_dataset, loss)],
        epochs=int(training_args.num_train_epochs),
        evaluator=dev_evaluator,
        evaluation_steps=model_args.eval_step,
        warmup_steps=warmup_steps,
        output_path=training_args.output_dir,
        optimizer_class=StableAdamW,
        optimizer_params=optimizer_parameter
    )
        
if __name__ == "__main__":
    main()
