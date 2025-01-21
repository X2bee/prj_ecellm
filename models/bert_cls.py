import joblib;
from huggingface_hub import hf_hub_download;
from peft import PeftModel, PeftConfig;
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
