import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# Sentence Transformer 모델 로드
model = SentenceTransformer("BAAI/bge-m3")

def get_sorted_sentences_by_similarity(query, sentences):
    """
    주어진 쿼리와 문장 리스트에 대해 유사도를 계산하고,
    유사도 순서로 문장들을 정렬하여 반환합니다.
    
    Args:
        query (str): 유사도를 계산할 기준 쿼리.
        sentences (list of str): 비교할 문장들의 리스트.
    
    Returns:
        list of tuple: 유사도 순서로 정렬된 (문장, 유사도) 튜플 리스트.
    """

    # 쿼리와 문장 임베딩
    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # 유사도 계산
    similarities = util.cos_sim(query_embedding, sentence_embeddings)
    similarities_list = similarities[0].cpu().tolist()  # Tensor에서 Python List로 변환

    # 유사도를 기준으로 정렬
    sorted_sentences = sorted(
        zip(sentences, similarities_list),
        key=lambda x: x[1],  # 유사도 값 기준 정렬
        reverse=True  # 높은 유사도 순으로
    )

    return sorted_sentences

def exp_normalize(x):
    """
    입력 배열 x에 대해 softmax와 비슷한 normalization을 수행.
    """
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

tokenizer = AutoTokenizer.from_pretrained("Dongjin-kr/ko-reranker")
rerank_model = AutoModelForSequenceClassification.from_pretrained("Dongjin-kr/ko-reranker").to("cuda")
rerank_model.eval()

def calculate_similarity(query, sentences):
    """
    Query와 Sentences 리스트를 입력받아 유사도를 계산한 후 반환합니다.
    
    Args:
        query (str): 비교할 기준 문장.
        sentences (list of str): 비교할 문장들의 리스트.
    
    Returns:
        list of tuple: 유사도 점수와 함께 문장 리스트를 반환.
    """
    # 모델과 토크나이저 로드

    
    # Query와 Sentences를 Pair로 변환
    pairs = [[query, sentence] for sentence in sentences]

    # 입력 데이터 생성
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1).float()

        # exp_normalize를 통해 정규화
        scores = exp_normalize(scores.cpu().detach().numpy())

    # 유사도와 문장을 정렬하여 반환
    sorted_results = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

    return sorted_results