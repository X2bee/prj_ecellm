from torch import Tensor
from torch import nn
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoConfig, AutoModel, PreTrainedModel, ModernBertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from loss import Loss

class ModernBERTSimCSE(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self, modernbert, config: AutoConfig,):
        super().__init__(config)
        self.modernbert = modernbert
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, 768, bias=True),
            nn.Tanh()
        )
        self.loss_fn = Loss()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Config 로드
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.architectures = ["ModernBertModel"]
        # Base model 로드
        base_model = ModernBertModel.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
        # ModernBERTSimCSE 초기화
        return cls(modernbert=base_model, config=config)

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
                anchor_pooled = self.mean_pooling(anchor_outputs.last_hidden_state, anchor_attention_mask)
                anchor_pooled = self.pooler(anchor_pooled)
                positive_pooled = self.mean_pooling(positive_outputs.last_hidden_state, positive_attention_mask)
                positive_pooled = self.pooler(positive_pooled)
                negative_pooled = self.mean_pooling(negative_outputs.last_hidden_state, negative_attention_mask)
                negative_pooled = self.pooler(negative_pooled)

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
                sentence_1_pooled = self.mean_pooling(sentence_1_outputs.last_hidden_state, sentence_1_attention_mask)
                sentence_1_pooled = self.pooler(sentence_1_pooled)
                sentence_2_pooled = self.mean_pooling(sentence_2_outputs.last_hidden_state, sentence_2_attention_mask)
                sentence_2_pooled = self.pooler(sentence_2_pooled)

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
        pooled = self.mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        return self.pooler(pooled)

    def mean_pooling(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Attention mask를 고려한 평균 풀링
        """
        weighted_sum = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
        mask_sum = attention_mask.sum(dim=1).unsqueeze(-1)
        return weighted_sum / mask_sum.clamp(min=1e-9)
    
    def compute_metrics(self, eval_pred):
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
