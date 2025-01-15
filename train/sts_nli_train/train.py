import logging
import os
import torch
import mlflow

from huggingface_hub import HfApi, login
from transformers import HfArgumentParser, TrainingArguments, set_seed, get_scheduler
from sentence_transformers import (
    models,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CosineSimilarityLoss
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
        
    if training_args.report_to == "mlflow":
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
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="weightedmean")
    dence = models.Dense(in_features=transformer.get_word_embedding_dimension(), out_features=768)
    # normalize = models.Normalize()
    
    model = SentenceTransformer(modules=[transformer, pooling, dence], device="cuda")
    loss = CosineSimilarityLoss(model)
    data_loader = STSrainDataLoader(data_args)
    train_dataset, eval_dataset = data_loader.prepare_samples()
    
    optimizer = StableAdamW(
        params=model.parameters(),
        lr=training_args.learning_rate, 
        betas=(0.9,0.999), 
        eps=1e-7, 
        weight_decay=0.01)
    
    num_update_steps_per_epoch = len(train_dataset) // training_args.per_device_train_batch_size
    max_train_steps = num_update_steps_per_epoch * training_args.num_train_epochs
    warmup_steps = int(max_train_steps * training_args.warmup_ratio)
    
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset['text'],
        sentences2=eval_dataset['pair'],
        scores=eval_dataset['label'],
        similarity_fn_names=['cosine', 'euclidean', 'manhattan', 'dot'],
        name="sts_dev",
    )
        
    if training_args.warmup_steps != None:
        warmup_steps = training_args.warmup_steps
        
    args = SentenceTransformerTrainingArguments(
        output_dir=training_args.output_dir,
        overwrite_output_dir=training_args.overwrite_output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        dataloader_drop_last=training_args.dataloader_drop_last,
        warmup_ratio=training_args.warmup_ratio,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        learning_rate=training_args.learning_rate,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        eval_strategy=training_args.eval_strategy,
        eval_steps=training_args.eval_steps,
        logging_steps=training_args.logging_steps,
        push_to_hub=training_args.push_to_hub,
        hub_model_id=training_args.hub_model_id,
        hub_strategy=training_args.hub_strategy,
        report_to=training_args.report_to,
        max_grad_norm=training_args.max_grad_norm,
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        optimizers=(optimizer, scheduler),
        evaluator=dev_evaluator,
    )
    
    trainer.train()
    trainer.push_to_hub(training_args.hub_model_id)

if __name__ == "__main__":
    main()
