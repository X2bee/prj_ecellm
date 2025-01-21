from torch.utils.data import IterableDataset

def batchify(iterable, batch_size):
    """
    iterable에서 하나씩 꺼내어, batch_size 크기만큼 묶어서 
    list[list[int]] 형태를 yield 하는 간단한 헬퍼 함수
    """
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    # 마지막에 남은 것도 있으면 yield
    if batch:
        yield batch

class PackedSequenceIterableDataset(IterableDataset):
    """
    Hugging Face Dataset(tokenized_*)에서 꺼낸 {'input_ids': [...], ...} 들을
    -> GreedyBestFitSequencePacker로 처리하여
    -> 최종 (input_ids, labels, attention_mask, ...) 딕셔너리를 yield 하는 IterableDataset.
    """

    def __init__(
        self,
        hf_dataset,                 # tokenized_train_dataset 등 HuggingFace Dataset
        packer,                     # GreedyBestFitSequencePacker(or 다른 구현)
        src_batch_size: int = 8,   # Packer가 요구하는 "src_batch_size"만큼 묶어서 넘길 예정
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.packer = packer
        self.src_batch_size = src_batch_size
        
    def __len__(self):
        total_length = int(len(self.hf_dataset) / self.src_batch_size)
        self.total_length = total_length
        
        return self.total_length

    def _sample_generator(self):
        """
        1) HF dataset을 순회 -> item(딕셔너리) 하나씩 뽑기
        2) batchify(...)로 'list[dict]' 형태의 미니배치 생성
        3) packer에 src_iterable로 세팅 -> packer.__iter__()가 최종 배치를 yield
        """
        def gen_items():
            for item in self.hf_dataset:
                # item 예: {'input_ids': [...], 'attention_mask': [...], ...}
                # 필요한 최소 키인 "input_ids"는 있어야 함
                yield item

        # 1) list[dict] 단위로 묶기
        items_batches = batchify(gen_items(), self.src_batch_size)

        # 2) packer가 incoming_batch (list[dict])를 받으면,
        #    내부에서 for item in incoming_batch -> item["input_ids"] 로 접근 가능
        self.packer.src_iterable = items_batches

        # 3) packer.__iter__() -> _generate_batches()를 통해 최종 packed_batch yield
        for packed_batch in self.packer:
            # packed_batch는 예: {"input_ids": Tensor(...), "labels": Tensor(...), ...}
            yield packed_batch

    def __iter__(self):
        return self._sample_generator()