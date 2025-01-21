import math
import os
import random
import datasets

from typing import List
from huggingface_hub import HfApi, login

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from transformers import BatchEncoding
from arguments import DataArguments

class STSrainDataLoader():
    def __init__(
            self,
            args: DataArguments,
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
        
        self.train_dataset = datasets.load_dataset(path=args.train_data, data_dir=args.train_data_dir, split=args.train_data_split)
        
        if args.test_data is not None:
            self.test_dataset = datasets.load_dataset(path=args.test_data, data_dir=args.test_data_dir, split=args.test_data_split)
            
        else:
            splitdata = self.train_dataset.train_test_split(test_size=args.train_test_split_ratio, seed=42)
            
            self.train_dataset = splitdata['train']
            self.test_dataset = splitdata['test']
            
        self.args = args
        self.total_len = len(self.train_dataset)

    def __len__(self):
        return self.total_len
    
    def prepare_samples(self):
        if self.args == None and self.train_dataset == None and self.args.train_test_split_ratio == 0:
            return self.train_dataset
        else:
            return self.train_dataset, self.test_dataset

        
