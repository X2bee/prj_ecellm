---
base_model: answerdotai/ModernBERT-base
---

# x2bee/KoModernBERT-base-mlm-v03-retry-ckp03
This model is a fine-tuned version of [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on [x2bee/Korean_wiki_corpus](https://huggingface.co/datasets/x2bee/Korean_wiki_corpus) and [x2bee/Korean_namuwiki_corpus](https://huggingface.co/datasets/x2bee/Korean_namuwiki_corpus). <br>

### Example Use.

```bash
git clone https://github.com/X2bee/prj_ecellm.git
```

```python
import os
os.chdir("/workspace")
from models.bert_mlm import ModernBertMLM

test_model = ModernBertMLM(model_id="x2bee/KoModernBERT-base-mlm-v03-retry-ckp03")

text = "30일 전남 무안국제[MASK] 활주로에 전날 발생한 제주항공 [MASK] 당시 기체가 [MASK]착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 [MASK]과 승무원 181명 중 179명이 숨지고 [MASK]는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다."
result = test_model.modern_bert_convert_with_multiple_masks(text, top_k=5)
result
```

### Output
```
Predicted: 공항 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 [MASK] 당시 기체가 [MASK]착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 [MASK]과 승무원 181명 중 179명이 숨지고 [MASK]는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 사고 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 [MASK]착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 [MASK]과 승무원 181명 중 179명이 숨지고 [MASK]는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 비상 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 [MASK]과 승무원 181명 중 179명이 숨지고 [MASK]는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 승객 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 [MASK]는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 잔해 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 [MASK]됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 훼손 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. [MASK] 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 사고 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 [MASK] 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 발생 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 [MASK]이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 주장 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 [MASK]에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 내부 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 내부에 설치된 [MASK](착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: ESP | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 내부에 설치된 ESP(착륙 유도 안전시설)가 [MASK]를 키웠다는 [MASK]이 나오고 있다.
Predicted: 사고 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 내부에 설치된 ESP(착륙 유도 안전시설)가 사고를 키웠다는 [MASK]이 나오고 있다.
Predicted: 주장 | Current text: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 내부에 설치된 ESP(착륙 유도 안전시설)가 사고를 키웠다는 주장이 나오고 있다.

```

### Compare with Real Value
```
Answer: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 대참사 당시 기체가 동체착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 항공기는 형체를 알아볼 수 없이 파손됐다. 피해 규모와 사고 원인 등에 대해 다양한 의문점이 제기되고 있는 가운데 활주로에 설치된 로컬라이저(착륙 유도 안전시설)가 피해를 키웠다는 지적이 나오고 있다.
Output: 30일 전남 무안국제공항 활주로에 전날 발생한 제주항공 사고 당시 기체가 비상착륙하면서 강한 마찰로 생긴 흔적이 남아 있다. 이 참사로 승객과 승무원 181명 중 179명이 숨지고 잔해는 형체를 알아볼 수 없이 훼손됐다. 사고 규모와 발생 원인 등에 대해 다양한 주장이 제기되고 있는 가운데 내부에 설치된 ESP(착륙 유도 안전시설)가 사고를 키웠다는 주장이 나오고 있다.

```

# plateer_classifier_ModernBERT_v01
This model is a fine-tuned version of [x2bee/ModernBert_MLM_kotoken_v01](https://huggingface.co/x2bee/ModernBert_MLM_kotoken_v01) on [x2bee/plateer_category_data](https://huggingface.co/datasets/x2bee/plateer_category_data). <br>
It achieves the following results on the evaluation set:
- Loss: 0.3379

### Example Use.
```python
import os
os.chdir("/workspace")
from models.bert_cls import plateer_classifier;

result = plateer_classifier("겨울 등산에서 사용할 옷")[0]
print(result)

############# result
-----------Category-----------
{'label': 2, 'label_decode': '기능성의류', 'score': 0.9214227795600891}
{'label': 8, 'label_decode': '스포츠', 'score': 0.07054771482944489}
{'label': 15, 'label_decode': '패션/의류/잡화', 'score': 0.0036312134470790625}
```

# CocoRoF/ModernBERT-SimCSE
This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on the [korean_nli_dataset](https://huggingface.co/datasets/x2bee/Korean_NLI_dataset) dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) <!-- at revision addb15798678d7f76904915cf8045628d402b3ce -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: ModernBertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': True, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Dense({'in_features': 768, 'out_features': 768, 'bias': True, 'activation_function': 'torch.nn.modules.activation.Tanh'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("CocoRoF/ModernBERT-SimCSE")
# Run inference
sentences = [
    '버스가 바쁜 길을 따라 운전한다.',
    '녹색 버스가 도로를 따라 내려간다.',
    '그 여자는 데이트하러 가는 중이다.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

### Metrics

#### Semantic Similarity

* Dataset: `sts_dev`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric             | Value      |
|:-------------------|:-----------|
| pearson_cosine     | 0.8273     |
| spearman_cosine    | 0.8298     |
| pearson_euclidean  | 0.8112     |
| spearman_euclidean | 0.8214     |
| pearson_manhattan  | 0.8125     |
| spearman_manhattan | 0.8226     |
| pearson_dot        | 0.7648     |
| spearman_dot       | 0.7648     |
| pearson_max        | 0.8273     |
| **spearman_max**   | **0.8298** |