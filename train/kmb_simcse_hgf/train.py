import logging
import os
import mlflow
import torch
import torch.distributed as dist

from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer, get_scheduler, TrainingArguments, ModernBertForMaskedLM, ModernBertModel
from transformers import HfArgumentParser, set_seed
from data import NLIDataLoader, STSDataLoader, SimCSEDataCollator
from arguments import ModelArguments, DataArguments
from optimizer import StableAdamW
from model import ModernBERTSimCSE

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://"
        )

        rank = dist.get_rank()

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
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.float16
    else:
        attn_implementation = None
        torch_dtype = None
        
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, subfolder=model_args.model_subfolder)
    model = ModernBERTSimCSE.from_pretrained(
        model_args.model_name_or_path,
        subfolder=model_args.model_subfolder,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,  
    )
    
    # model = ModernBERTSimCSE(ModernBertModel.from_pretrained(model_args.model_name_or_path, subfolder=model_args.model_subfolder, attn_implementation=attn_implementation, torch_dtype=torch_dtype))
    model.to('cuda')
    
    nli_loader = NLIDataLoader(data_args, tokenizer=tokenizer)
    sts_loader = STSDataLoader(data_args, tokenizer=tokenizer)
    
    optimizer = StableAdamW(
        params=model.parameters(),
        lr=training_args.learning_rate,
        betas=(0.9,0.999), 
        eps=1e-7, 
        weight_decay=training_args.weight_decay
    )
    
    num_update_steps_per_epoch = len(nli_loader) // training_args.per_device_train_batch_size
    max_train_steps = num_update_steps_per_epoch * training_args.num_train_epochs
    warmup_steps = int(max_train_steps * training_args.warmup_ratio)
    
    if training_args.warmup_steps != None:
        warmup_steps = training_args.warmup_steps
    
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
        
    training_args.dataloader_num_workers = (os.cpu_count() // 4)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        optimizers=(optimizer,scheduler),
        train_dataset=nli_loader,
        eval_dataset=sts_loader,
        compute_metrics=model.compute_metrics,
        data_collator=SimCSEDataCollator(tokenizer=tokenizer)
    )
    
    if rank == 0:
        print(trainer.optimizer)

    try:
        trainer.train(resume_from_checkpoint=model_args.is_resume)
    except Exception as e:
        print(f"Error: {e}")
        print("Train Error")
        dist.destroy_process_group()
        exit(1)

    if rank == 0:
        trainer.push_to_hub(commit_message=f"{model_args.model_commit_msg} Done")

    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()