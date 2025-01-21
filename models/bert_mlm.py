import random
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from huggingface_hub import HfApi, login
with open('./api_key/HGF_TOKEN.txt', 'r') as hgf:
    login(token=hgf.read())
api = HfApi()
class ModernBertMLM:
    def __init__(self, model_id: str = "x2bee/ModernBert_MLM_kotoken_v03", subfolder = ""):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id, subfolder=subfolder).to("cuda")
        
    def modern_bert_mlm(self, text: str, top_k: int = 3):
        if "[MASK]" not in text:
            raise ValueError("MLM Model should include '[MASK]' in the sentence")
            
        else:
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
            outputs = self.model(**inputs)
            
            # 마스크 토큰의 위치를 찾음
            masked_index = inputs["input_ids"][0].tolist().index(self.tokenizer.mask_token_id)
            
            # 로짓에서 상위 top_k 개 예측 토큰을 선택
            logits = outputs.logits[0, masked_index]
            top_k_token_ids = logits.topk(top_k).indices
            top_k_scores = logits.topk(top_k).values  # 상위 top_k 개의 로짓 값

            # 예측된 토큰을 디코딩
            predicted_tokens = [self.tokenizer.decode(token_id) for token_id in top_k_token_ids]
            
            # 결과 출력 (토큰과 해당 로짓 스코어)
            for token, score in zip(predicted_tokens, top_k_scores):
                print(f"Predicted token: {token} with logits score: {score.item()}")
            
            return predicted_tokens, top_k_scores


    def modern_bert_convert_with_multiple_masks(self, text: str, top_k: int = 1, select_method:str = "Logit") -> str:
        """
        문장에 여러 개의 [MASK] 토큰이 있을 경우, 순차적으로 변환하여 최종 문장을 완성하는 함수.

        Args:
            text (str): [MASK] 토큰이 포함된 입력 문장.
            top_k (int): 각 [MASK] 위치에서 상위 k개의 예측값 중 하나를 선택 (기본값은 1).
            select_method (str): "Logit", "Random", "Best" 중 하나를 입력. Logit은 Logit에 따라 확률적으로 선택, Random은 완전 랜덤. Best는 가장 확률이 높은 것을 선택함.

        Returns:
            str: 모든 [MASK] 토큰이 예측된 값으로 치환된 문장.
        """
        if "[MASK]" not in text:
            raise ValueError("MLM Model should include '[MASK]' in the sentence")

        while "[MASK]" in text:
            # 입력 문장을 토크나이저로 처리
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
            outputs = self.model(**inputs)

            # 모든 [MASK] 토큰의 위치를 확인
            input_ids = inputs["input_ids"][0].tolist()
            mask_indices = [i for i, token_id in enumerate(input_ids) if token_id == self.tokenizer.mask_token_id]

            # 가장 앞에 있는 [MASK] 위치 선택
            current_mask_index = mask_indices[0]

            # 해당 [MASK] 위치의 로짓 가져오기
            logits = outputs.logits[0, current_mask_index]

            # 상위 top_k 예측값 가져오기
            top_k_tokens = logits.topk(top_k).indices.tolist()
            top_k_logits, top_k_indices = logits.topk(top_k)
            
            if select_method == "Logit":
                # softmax를 사용하여 확률로 변환
                probabilities = torch.softmax(top_k_logits, dim=0).tolist()
                # 확률 기반으로 랜덤하게 선택
                predicted_token_id = random.choices(top_k_indices.tolist(), weights=probabilities, k=1)[0]
                predicted_token = self.tokenizer.decode([predicted_token_id]).strip()
                
            elif select_method == "Random":
                # 랜덤하게 선택
                predicted_token_id = random.choice(top_k_tokens)
                predicted_token = self.tokenizer.decode([predicted_token_id]).strip()
                
            elif select_method == "Best":
                # 가장 확률이 높은 예측 토큰 선택
                predicted_token_id = top_k_tokens[0]
                predicted_token = self.tokenizer.decode([predicted_token_id]).strip()
                
            else:
                raise ValueError("select_method should be one of ['Logit', 'Random', 'Best']")

            # [MASK]를 예측된 토큰으로 대체
            text = text.replace("[MASK]", predicted_token, 1)

            print(f"Predicted: {predicted_token} | Current text: {text}")

        return text
