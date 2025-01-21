import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances

class Loss:
    def __init__(self):
        self.cos = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, anchor, positive, negative):
        """
        Anchor, Positive, Negative 임베딩을 입력받아 손실 계산.
        """
        # 코사인 유사도 계산
        positive_similarity = self.cos(anchor.unsqueeze(1), positive.unsqueeze(0)) / 0.05
        negative_similarity = self.cos(anchor.unsqueeze(1), negative.unsqueeze(0)) / 0.05

        # Positive와 Negative 유사도를 결합
        cosine_similarity = torch.cat([positive_similarity, negative_similarity], dim=1).to('cuda')

        # 정답 라벨 생성
        labels = torch.arange(cosine_similarity.size(0)).long().to('cuda')

        # CrossEntropy 손실 계산
        loss = self.criterion(cosine_similarity, labels)
        return loss

    def evaluation_during_training(self, embeddings1, embeddings2, labels):
        """
        학습 중 평가 함수.
        """
        # NumPy로 변환
        embeddings1 = embeddings1.cpu().numpy()
        embeddings2 = embeddings2.cpu().numpy()
        labels = labels.cpu().numpy().flatten()

        # 거리 및 유사도 계산
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        # Pearson 및 Spearman 상관계수 계산
        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        # 결과 딕셔너리 반환
        return {
            "eval_pearson_cosine": eval_pearson_cosine,
            "eval_spearman_cosine": eval_spearman_cosine,
            "eval_pearson_manhattan": eval_pearson_manhattan,
            "eval_spearman_manhattan": eval_spearman_manhattan,
            "eval_pearson_euclidean": eval_pearson_euclidean,
            "eval_spearman_euclidean": eval_spearman_euclidean,
            "eval_pearson_dot": eval_pearson_dot,
            "eval_spearman_dot": eval_spearman_dot,
        }
