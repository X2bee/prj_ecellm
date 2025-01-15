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

class CETrainDataLoader(Dataset):
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
        
        self.dataset = datasets.load_dataset(path=args.train_data, data_dir=args.data_dir, split='train')
        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        query = self.dataset[item]["query"]
        positive = random.choice(self.dataset[item]["positive"])
        
        if len(self.dataset[item]["negative"]) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]["negative"]))
            negatives = random.sample(self.dataset[item]["negative"] * num, self.args.train_group_size - 1)
        else:
            negatives = random.sample(self.dataset[item]["negative"], self.args.train_group_size - 1)

        
        samples = [
            InputExample(texts=[query, positive], label=1.0)
        ]
        samples.extend(
            [InputExample(texts=[query, neg], label=0.0) for neg in negatives]
        )

        return samples
    
    def prepare_samples(self, type:str="train_only", train_batch_size:int=1, split_ratio:float=0.1) -> List[InputExample]:
        """
        전체 데이터를 `list[InputExample]` 형태로 변환하여 반환.

        Returns:
            train_samples (List[InputExample])
        """
        if type=="train_only":
            split_ratio = 0.0
            type_list = ['train']
        
        elif type=="split":
            type_list = ['train', 'test']
        
        split_dataset = self.dataset.train_test_split(test_size=split_ratio, seed=42)
        
        train_samples = []
        val_samples = {}

        for type_name in type_list:
            for item in split_dataset[type_name]:
                query = item["query"]
                positive = random.choice(item["positive"])
                if len(item["negative"]) < self.args.train_group_size - 1:
                    num = math.ceil(
                        (self.args.train_group_size - 1) / len(item["negative"])
                    )
                    negatives = random.sample(
                        item["negative"] * num, self.args.train_group_size - 1
                    )
                else:
                    negatives = random.sample(
                        item["negative"], self.args.train_group_size - 1
                    )

                # InputExample 생성
                examples = [
                    InputExample(texts=[query, positive], label=1.0)
                ]
                examples.extend(
                    [InputExample(texts=[query, neg], label=0.0) for neg in negatives]
                )
                
                if type_name=="train":
                    train_samples.extend(examples)
                elif type_name == "test":
                    if query not in val_samples:
                        val_samples[query] = {"query": query, "positive": set(), "negative": set()}
                    val_samples[query]["positive"].add(positive)
                    val_samples[query]["negative"].update(negatives)
                    
        for query in val_samples:
            val_samples[query]["positive"] = list(val_samples[query]["positive"])
            val_samples[query]["negative"] = list(val_samples[query]["negative"])
        
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
        
        if type=="split":
            return train_dataloader, val_samples
        else:
            return train_dataloader, None

        
