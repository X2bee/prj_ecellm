import os
import torch
import datasets
from huggingface_hub import HfApi, login
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset
from arguments import DataArguments

class NLIDataLoader(Dataset):
    def __init__(self, 
                 args: DataArguments, 
                 tokenizer: PreTrainedTokenizer):
        """
        Initialize the NLI data loader.

        Args:
        """
        try:
            login(token=args.hf_data_token)
        except:
            print("Fail to login hgf")
        
        self.dataset = datasets.load_dataset(path=args.train_data, data_dir=args.train_data_dir, split=args.train_data_split)
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_len
        
        if args.data_filtering:
            self.dataset = self.dataset.filter(self.is_valid_input)
        
        self.dataset = self.dataset.map(self.preprocess, batched=True, num_proc=os.cpu_count())

    def preprocess(self, examples):
        """
        Preprocess the NLI dataset by tokenizing anchor, positive, and negative samples.

        Args:
            examples (dict): A dictionary containing raw dataset examples.

        Returns:
            dict: Tokenized inputs for anchor, positive, and negative samples.
        """
        anchor = self.tokenizer(
            examples["anchor"],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        positive = self.tokenizer(
            examples["positive"],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        negative = self.tokenizer(
            examples["negative"],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "anchor_input_ids": anchor["input_ids"],
            "anchor_attention_mask": anchor["attention_mask"],
            "positive_input_ids": positive["input_ids"],
            "positive_attention_mask": positive["attention_mask"],
            "negative_input_ids": negative["input_ids"],
            "negative_attention_mask": negative["attention_mask"],
        }

    def prepare(self):
        """
        Tokenize the dataset using the preprocess function.
        """
        self.dataset = self.dataset.map(self.preprocess, batched=True)
        
    def is_valid_input(self, example):
        """
        Check if the input data contains valid strings for processing.

        Args:
            example (dict): A single example from the dataset.

        Returns:
            bool: True if the input is valid, False otherwise.
        """
        try:
            return all(isinstance(example[key], str) for key in ["anchor", "positive", "negative"])
        except KeyError:
            return False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class STSDataLoader(Dataset):
    def __init__(self, 
                 args: DataArguments, 
                 tokenizer: PreTrainedTokenizer
                ):
        """
        """
        try:
            login(token=args.hf_data_token)
        except:
            print("Fail to login hgf")
            
        self.dataset = datasets.load_dataset(path=args.sts_data, data_dir=args.sts_data_dir, split=args.sts_data_split)
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_len
        
        if args.data_filtering:
            self.dataset = self.dataset.filter(self.is_valid_input)
        
        self.dataset = self.dataset.map(self.preprocess, batched=True, num_proc=os.cpu_count())
        
    def preprocess(self, examples):
        """
        Preprocess the STS dataset by tokenizing sentence pairs.

        Args:
            examples (dict): A dictionary containing raw dataset examples.

        Returns:
            dict: Tokenized inputs for sentence pairs and corresponding labels.
        """
        sentence_1 = self.tokenizer(
            examples["text"],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        sentence_2 = self.tokenizer(
            examples["pair"],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "sentence_1_input_ids": sentence_1["input_ids"],
            "sentence_1_attention_mask": sentence_1["attention_mask"],
            "sentence_2_input_ids": sentence_2["input_ids"],
            "sentence_2_attention_mask": sentence_2["attention_mask"],
            "labels": examples["label"],
        }

    def prepare(self):
        """
        Tokenize the dataset using the preprocess function.
        """
        self.dataset = self.dataset.map(self.preprocess, batched=True)
        
    def is_valid_input(self, example):
        """
        """
        try:
            return all(isinstance(example[key], str) for key in ["text", "pair"])
        except KeyError:
            return False
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class SimCSEDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        """
        Custom collator to handle both NLI and STS inputs.
        """
        # Check the type of input data
        
        if "anchor_input_ids" in features[0]:
            # NLI 데이터 처리
            batch = {
                "anchor_input_ids": [f["anchor_input_ids"] for f in features],
                "anchor_attention_mask": [f["anchor_attention_mask"] for f in features],
                "positive_input_ids": [f["positive_input_ids"] for f in features],
                "positive_attention_mask": [f["positive_attention_mask"] for f in features],
                "negative_input_ids": [f["negative_input_ids"] for f in features],
                "negative_attention_mask": [f["negative_attention_mask"] for f in features],
            }
        elif "sentence_1_input_ids" in features[0]:
            # STS 데이터 처리
            batch = {
                "sentence_1_input_ids": [f["sentence_1_input_ids"] for f in features],
                "sentence_1_attention_mask": [f["sentence_1_attention_mask"] for f in features],
                "sentence_2_input_ids": [f["sentence_2_input_ids"] for f in features],
                "sentence_2_attention_mask": [f["sentence_2_attention_mask"] for f in features],
                "labels": [f["labels"] for f in features],
            }
        else:
            raise ValueError("Features do not match NLI or STS format.")

        # Convert lists to tensors
                
        batch = {key: torch.tensor(value, dtype=torch.long) if "labels" not in key else torch.tensor(value, dtype=torch.float) for key, value in batch.items()}
        
        return batch

# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    max_seq_length = 128

    # NLI DataLoader
    nli_loader = NLIDataLoader("snli", tokenizer, max_seq_length)
    nli_loader.prepare()

    # STS DataLoader
    sts_loader = STSDataLoader("sts", tokenizer, max_seq_length)
    sts_loader.prepare()

    print("NLI sample:", nli_loader[0])
    print("STS sample:", sts_loader[0])
