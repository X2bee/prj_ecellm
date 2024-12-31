---
base_model: answerdotai/ModernBERT-base
---

# ModernBert_MLM_kotoken_v02
This model is a fine-tuned version of [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on [x2bee/Korean_wiki_corpus](https://huggingface.co/datasets/x2bee/Korean_wiki_corpus) and [x2bee/Korean_namuwiki_corpus](https://huggingface.co/datasets/x2bee/Korean_namuwiki_corpus). <br>
It achieves the following results on the evaluation set:
- Loss: 1.7133122

### Example Use.
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
from huggingface_hub import HfApi, login
with open('./api_key/HGF_TOKEN.txt', 'r') as hgf:
    login(token=hgf.read())
api = HfApi()

model_id = "x2bee/ModernBert_MLM_kotoken_v02"
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="last-checkpoint")
model = AutoModelForMaskedLM.from_pretrained(model_id, subfolder="last-checkpoint").to("cuda")

def modern_bert_convert_with_multiple_masks(text: str, top_k: int = 1, select_method:str = "Logit") -> str:
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
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = model(**inputs)

        # 모든 [MASK] 토큰의 위치를 확인
        input_ids = inputs["input_ids"][0].tolist()
        mask_indices = [i for i, token_id in enumerate(input_ids) if token_id == tokenizer.mask_token_id]

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
            predicted_token = tokenizer.decode([predicted_token_id]).strip()
            
        elif select_method == "Random":
            # 랜덤하게 선택
            predicted_token_id = random.choice(top_k_tokens)
            predicted_token = tokenizer.decode([predicted_token_id]).strip()
            
        elif select_method == "Best":
            # 가장 확률이 높은 예측 토큰 선택
            predicted_token_id = top_k_tokens[0]
            predicted_token = tokenizer.decode([predicted_token_id]).strip()
            
        else:
            raise ValueError("select_method should be one of ['Logit', 'Random', 'Best']")

        # [MASK]를 예측된 토큰으로 대체
        text = text.replace("[MASK]", predicted_token, 1)

        print(f"Predicted: {predicted_token} | Current text: {text}")

    return text

text = "30일 전남 무안[MASK]공항 활주로에 전날 발생한 제주항공 [MASK] 당시 기체가 [MASK]착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 [MASK]과 승무원 181명 중 179명이 숨지고 [MASK]는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다."
result = modern_bert_convert_with_multiple_masks(text, top_k=3, select_method="Logit")
```

# plateer_classifier_ModernBERT_v01
This model is a fine-tuned version of [x2bee/ModernBert_MLM_kotoken_v01](https://huggingface.co/x2bee/ModernBert_MLM_kotoken_v01) on [x2bee/plateer_category_data](https://huggingface.co/datasets/x2bee/plateer_category_data). <br>
It achieves the following results on the evaluation set:
- Loss: 0.3379

### Example Use.
```python
import joblib;
from huggingface_hub import hf_hub_download;
from transformers import AutoTokenizer, TextClassificationPipeline, AutoModelForSequenceClassification;
from huggingface_hub import HfApi, login
with open('./api_key/HGF_TOKEN.txt', 'r') as hgf:
    login(token=hgf.read())
api = HfApi()
repo_id = "x2bee/plateer_classifier_ModernBERT_v01"
data_id = "x2bee/plateer_category_data"

# Load Config, Tokenizer, Label_Encoder
tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="last-checkpoint")
label_encoder_file = hf_hub_download(repo_id=data_id, repo_type="dataset", filename="label_encoder.joblib")
label_encoder = joblib.load(label_encoder_file)

# Load Model
model = AutoModelForSequenceClassification.from_pretrained(repo_id, subfolder="last-checkpoint")

import torch
class TextClassificationPipeline(TextClassificationPipeline):
    def __call__(self, inputs, top_k=5, **kwargs):
        inputs = self.tokenizer(inputs, return_tensors="pt", truncation=True, padding=True, max_length=512, **kwargs)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        scores, indices = torch.topk(probs, top_k, dim=-1)
        
        results = []
        for batch_idx in range(indices.shape[0]):
            batch_results = []
            for score, idx in zip(scores[batch_idx], indices[batch_idx]):
                temp_list = []
                label = self.model.config.id2label[idx.item()]
                label = int(label.split("_")[1])
                temp_list.append(label)
                predicted_class = label_encoder.inverse_transform(temp_list)[0]
                            
                batch_results.append({
                    "label": label,
                    "label_decode": predicted_class,
                    "score": score.item(),
                })
            results.append(batch_results)
        
        return results

classifier_model = TextClassificationPipeline(tokenizer=tokenizer, model=model)

def plateer_classifier(text, top_k=3):
    result = classifier_model(text, top_k=top_k)
    return result

result = plateer_classifier("겨울 등산에서 사용할 옷")[0]
print(result)

############# result
-----------Category-----------
{'label': 2, 'label_decode': '기능성의류', 'score': 0.9214227795600891}
{'label': 8, 'label_decode': '스포츠', 'score': 0.07054771482944489}
{'label': 15, 'label_decode': '패션/의류/잡화', 'score': 0.0036312134470790625}
```

