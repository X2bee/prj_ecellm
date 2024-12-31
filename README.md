---
base_model: answerdotai/ModernBERT-base
model-index:
- name: ModernBert_MLM_kotoken_v01
  results: []
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

text = "대한민국의 수도는 [MASK]이다."
result = modern_bert_mlm(text, top_k=10)
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

