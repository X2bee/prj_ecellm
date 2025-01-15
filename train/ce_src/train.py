import logging
import os
import mlflow
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer, TrainingArguments
from transformers import HfArgumentParser, set_seed, get_scheduler

from arguments import ModelArguments, DataArguments
from data import TrainDatasetForCE, GroupCollator
from modeling import CrossEncoder
from trainer import CETrainer
from src.optimizer import StableAdamW

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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    if training_args.report_to == "mlflow":
        remote_server_uri = "https://polar-mlflow.x2bee.com/"
        mlflow.set_tracking_uri(remote_server_uri)
        mlflow.set_experiment(training_args.hub_model_id)
    
    set_seed(training_args.seed)

    num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
    )
    _model_class = CrossEncoder

    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    model.to('cuda')
    
    optimizer = StableAdamW(
        params=model.parameters(),
        lr=training_args.learning_rate, 
        betas=(0.9,0.999), 
        eps=1e-7, 
        weight_decay=0.01)
    
    train_dataset = TrainDatasetForCE(data_args, tokenizer=tokenizer)
    
    num_update_steps_per_epoch = len(train_dataset) // training_args.per_device_train_batch_size
    max_train_steps = num_update_steps_per_epoch * training_args.num_train_epochs
    warmup_steps = int(max_train_steps * 0.0)
    
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    _trainer_class = CETrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer,scheduler),
        data_collator=GroupCollator(tokenizer),
        processing_class=tokenizer
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
