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
    config_name: str = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
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
        default=False, metadata={"help": "Use Flash Attention 2"}
    )


@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Repository Name of HuggingFace Dataset"}
    )
    train_data_dir: str = field(
        default='', metadata={"help": "Repository Subfolder of HuggingFace Dataset, Default is = '' "}
    )
    train_data_split: str = field(
        default='train', metadata={"help": "Train Data Split Method"}
    )
    test_data: str = field(
        default=None, metadata={"help": "Repository Name of HuggingFace Dataset. If test_data path is set, train-test split ratio is not working"}
    )
    test_data_dir: str = field(
        default=None, metadata={"help": "Repository Subfolder of HuggingFace Dataset"}
    )
    test_data_split: str = field(
        default='train', metadata={"help": "Test Data Split Method"}
    )
    train_test_split_ratio: float = field(
        default=0.05, metadata={"help": "Test Data Split Ratio"}
    )
    hf_data_token: str = field(
        default=None, metadata={"help": "HuggingFace Data Login Token"}
    )
    max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
