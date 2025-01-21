import torch
from torch import nn
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from loss import Loss

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

class ModernBERTSimCSE(nn.Module):
    def __init__(self, modernbert, config, pooler_type="avg"):
        super().__init__()
        self.modernbert = modernbert
        self.loss_fn = Loss()
        self.config = config
        self.pooler = Pooler(pooler_type)

    @classmethod
    def from_pretrained(cls, model_name_or_path, pooler_type="avg", **kwargs):
        # Config 로드
        config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
        
        # ModernBertModel 로드
        modernbert = AutoModel.from_pretrained(model_name_or_path, config=config, **kwargs)

        # ModernBERTSimCSE 인스턴스 생성
        return cls(modernbert=modernbert, config=config, pooler_type=pooler_type)

    def forward(self, 
                anchor_input_ids=None, 
                anchor_attention_mask=None, 
                positive_input_ids=None, 
                positive_attention_mask=None, 
                negative_input_ids=None, 
                negative_attention_mask=None, 
                sentence_1_input_ids=None, 
                sentence_1_attention_mask=None, 
                sentence_2_input_ids=None, 
                sentence_2_attention_mask=None, 
                labels=None):
        try:
            if anchor_input_ids is not None:
                # NLI (train) 처리
                anchor_outputs = self.modernbert(
                    input_ids=anchor_input_ids,
                    attention_mask=anchor_attention_mask,
                    return_dict=True,
                )
                positive_outputs = self.modernbert(
                    input_ids=positive_input_ids,
                    attention_mask=positive_attention_mask,
                    return_dict=True,
                )
                negative_outputs = self.modernbert(
                    input_ids=negative_input_ids,
                    attention_mask=negative_attention_mask,
                    return_dict=True,
                )

                # Pooling 수행
                anchor_pooled = self.pooler(anchor_attention_mask, anchor_outputs)
                positive_pooled = self.pooler(positive_attention_mask, positive_outputs)
                negative_pooled = self.pooler(negative_attention_mask, negative_outputs)

                # 손실 계산
                loss = self.loss_fn.compute_loss(anchor_pooled, positive_pooled, negative_pooled)

                return SequenceClassifierOutput(
                    loss=loss,
                    logits=None,
                    hidden_states=(anchor_pooled, positive_pooled, negative_pooled),
                    attentions=None 
                )

            elif sentence_1_input_ids is not None and sentence_2_input_ids is not None:
                # STS (evaluation) 처리
                sentence_1_outputs = self.modernbert(
                    input_ids=sentence_1_input_ids,
                    attention_mask=sentence_1_attention_mask,
                    return_dict=True,
                )
                sentence_2_outputs = self.modernbert(
                    input_ids=sentence_2_input_ids,
                    attention_mask=sentence_2_attention_mask,
                    return_dict=True,
                )

                # Pooling 수행
                sentence_1_pooled = self.pooler(sentence_1_attention_mask, sentence_1_outputs)
                sentence_2_pooled = self.pooler(sentence_2_attention_mask, sentence_2_outputs)

                if labels is not None:
                    # Loss 계산
                    cosine_similarity = nn.CosineSimilarity(dim=-1)
                    scores = cosine_similarity(sentence_1_pooled, sentence_2_pooled)
                    mse_loss = nn.MSELoss()

                    loss = mse_loss(scores, labels)

                    return SequenceClassifierOutput(
                        loss=loss,  # Loss 값
                        logits=(sentence_1_pooled, sentence_2_pooled),
                        attentions=None 
                    )

                return None, (sentence_1_pooled, sentence_2_pooled)

            else:
                raise ValueError("Invalid input configuration for forward method.")
        except Exception as e:
            print(e)
            
    def encode(self, inputs):
        """
        768차원의 임베딩 벡터 추출
        """
        outputs = self.modernbert(
            input_ids=inputs["source"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
        )
        return self.pooler(inputs["attention_mask"], outputs)


    def compute_metrics(self, eval_pred):
        """
        """
        
        predictions, labels = eval_pred

        # 두 개의 문장 임베딩을 분리
        sentence_1_embeddings, sentence_2_embeddings = predictions[0], predictions[1]

        # NumPy로 변환
        embeddings1 = sentence_1_embeddings
        embeddings2 = sentence_2_embeddings
        labels = labels.flatten()

        # 거리 및 유사도 계산
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

        # Pearson 및 Spearman 상관계수 계산
        
        print(cosine_scores.shape)
        print(labels.shape)
            
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
            "pearson_cosine": eval_pearson_cosine,
            "spearman_cosine": eval_spearman_cosine,
            "pearson_manhattan": eval_pearson_manhattan,
            "spearman_manhattan": eval_spearman_manhattan,
            "pearson_euclidean": eval_pearson_euclidean,
            "spearman_euclidean": eval_spearman_euclidean,
            "pearson_dot": eval_pearson_dot,
            "spearman_dot": eval_spearman_dot,
        }
