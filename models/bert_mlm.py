from transformers import AutoTokenizer, AutoModelForMaskedLM
from huggingface_hub import HfApi, login
with open('./api_key/HGF_TOKEN.txt', 'r') as hgf:
    login(token=hgf.read())
api = HfApi()

model_id = "x2bee/ModernBert_MLM_kotoken_v02"
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="last-checkpoint")
model = AutoModelForMaskedLM.from_pretrained(model_id, subfolder="last-checkpoint").to("cuda")

def modern_bert_mlm(text: str, top_k: int = 3):
    if "[MASK]" not in text:
        raise ValueError("MLM Model should include '[MASK]' in the sentence")
        
    else:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        
        # 마스크 토큰의 위치를 찾음
        masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
        
        # 로짓에서 상위 top_k 개 예측 토큰을 선택
        logits = outputs.logits[0, masked_index]
        top_k_token_ids = logits.topk(top_k).indices
        top_k_scores = logits.topk(top_k).values  # 상위 top_k 개의 로짓 값

        # 예측된 토큰을 디코딩
        predicted_tokens = [tokenizer.decode(token_id) for token_id in top_k_token_ids]
        
        # 결과 출력 (토큰과 해당 로짓 스코어)
        for token, score in zip(predicted_tokens, top_k_scores):
            print(f"Predicted token: {token} with logits score: {score.item()}")
        
        return predicted_tokens, top_k_scores
