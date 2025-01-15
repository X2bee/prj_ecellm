import logging
import os
import torch
from pathlib import Path

from transformers import TrainingArguments
from transformers import HfArgumentParser, set_seed
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator

from arguments import ModelArguments, DataArguments
from data import CETrainDataLoader
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
    
    set_seed(training_args.seed)

    num_labels = 1
    
    if model_args.flash_attn:
        automodel_args = {
            "attn_implementation": "flash_attention_2", 
            "torch_dtype": torch.float16
        }
    else:
        automodel_args = {}

    model = CrossEncoder(
        model_args.model_name_or_path, 
        num_labels=num_labels, 
        max_length=model_args.max_length,
        automodel_args=automodel_args
    )
    
    optimizer_parameter = {
        "lr": training_args.learning_rate, 
        "betas": (0.9,0.999),
        "eps": 1e-7,
        "weight_decay": 0.01,
    }
    
    data_loader  = CETrainDataLoader(data_args)
    train_dataloader, val_samples = data_loader.prepare_samples(type="split", train_batch_size=training_args.per_device_train_batch_size, split_ratio=0.01)
    
    evaluator = CERerankingEvaluator(val_samples, name="train-eval")
    
    num_update_steps_per_epoch = len(train_dataloader) // training_args.per_device_train_batch_size
    max_train_steps = num_update_steps_per_epoch * training_args.num_train_epochs
    warmup_steps = int(max_train_steps * training_args.warmup_ratio)
    
    if training_args.warmup_steps != None:
        warmup_steps = training_args.warmup_steps
    
    model.fit(
        train_dataloader=train_dataloader,
        epochs=int(training_args.num_train_epochs),
        evaluator=evaluator,
        evaluation_steps=model_args.eval_step,
        warmup_steps=warmup_steps,
        output_path=training_args.output_dir,
        use_amp=True,
        optimizer_class=StableAdamW,
        optimizer_params=optimizer_parameter
    )


    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    model.save(training_args.output_dir)

if __name__ == "__main__":
    main()
