import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    hf_model_token: str = field(
        default=None, metadata={"help": "HuggingFace Model Login Token"}
    )
    max_length: int = field(
        default=512, metadata={"help": "Model Max Length"}
    )
    eval_step: int = field(
        default=500, metadata={"help": "Evaluation Step"}
    )
    flash_attn: bool = field(
        default=True, metadata={"help": "Use Flash Attention 2"}
    )


@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Repository Name of HuggingFace Dataset"}
    )
    data_dir: str = field(
        default='', metadata={"help": "Repository Subfolder of HuggingFace Dataset, Default is = '' "}
    )
    hf_data_token: str = field(
        default=None, metadata={"help": "HuggingFace Data Login Token"}
    )
    train_group_size: int = field(default=8)
    max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
