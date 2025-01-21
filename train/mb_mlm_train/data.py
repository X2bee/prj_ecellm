import os
import datasets
from huggingface_hub import HfApi, login

from torch.utils.data import Dataset
from arguments import DataArguments

class MLMDataLoader(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer,
            column_name: str
    ):
        """
        Args:
            train_data (str): HuggingFace dataset 경로 (예: "CocoRoF/msmacro_triplet_ko").
            train_group_size (int): 1개의 positive 샘플에 대해 포함할 negative 샘플 개수.
            data_dir (str, optional): 데이터셋 저장 경로.
            hf_data_token: 접근 권한이 필요한 경우 hf_token
        """
        try:
            login(token=args.hf_data_token)
        except:
            print("Fail to login hgf")
        
        items = args.train_data_dir.split(',')
        
        try:
            dataset = [datasets.load_dataset(path=args.train_data, data_dir=dir_name, split=args.train_data_split) for dir_name in items]
            self.dataset = datasets.concatenate_datasets(dataset)
        except:
            self.dataset = datasets.load_dataset(path=args.train_data, data_dir=args.data_dir, split=args.train_data_split)
            
        self.args = args
        self.tokenizer = tokenizer
        self.total_len = len(self.dataset)
        self.column_name = column_name

    def __len__(self):
        return self.total_len
    
    def is_valid_input(self, example):
        try:
            text = example[self.column_name]
            if not isinstance(text, str):
                return False

            return True
        except Exception:
            return False
        
    def encode(self, examples):
        return self.tokenizer(examples[self.column_name], padding="max_length", truncation=True, max_length=self.args.max_len)

    def prepare_samples(self):
        dataset = self.dataset.filter(self.is_valid_input)
        split_dataset = dataset.train_test_split(test_size=self.args.train_test_split_ratio, seed=42)
        
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        
        tokenized_train_dataset = train_dataset.map(self.encode, batched=True, num_proc=os.cpu_count())
        tokenized_test_dataset = test_dataset.map(self.encode, batched=True, num_proc=os.cpu_count())

        return tokenized_train_dataset, tokenized_test_dataset

        
