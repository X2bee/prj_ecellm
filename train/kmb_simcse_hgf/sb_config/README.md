---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:392702
- loss:CosineSimilarityLoss
base_model: x2bee/KoModernBERT-base-mlm-v03-retry-ckp03
widget:
- source_sentence: 우리는 움직이는 동행 우주 정지 좌표계에 비례하여 이동하고 있습니다 ... 약 371km / s에서 별자리 leo
    쪽으로. "
  sentences:
  - 두 마리의 독수리가 가지에 앉는다.
  - 다른 물체와는 관련이 없는 '정지'는 없다.
  - 소녀는 버스의 열린 문 앞에 서 있다.
- source_sentence: 숲에는 개들이 있다.
  sentences:
  - 양을 보는 아이들.
  - 여왕의 배우자를 "왕"이라고 부르지 않는 것은 아주 좋은 이유가 있다. 왜냐하면 그들은 왕이 아니기 때문이다.
  - 개들은 숲속에 혼자 있다.
- source_sentence: '첫째, 두 가지 다른 종류의 대시가 있다는 것을 알아야 합니다 : en 대시와 em 대시.'
  sentences:
  - 그들은 그 물건들을 집 주변에 두고 가거나 집의 정리를 해칠 의도가 없다.
  - 세미콜론은 혼자 있을 수 있는 문장에 참여하는데 사용되지만, 그들의 관계를 강조하기 위해 결합됩니다.
  - 그의 남동생이 지켜보는 동안 집 앞에서 트럼펫을 연주하는 금발의 아이.
- source_sentence: 한 여성이 생선 껍질을 벗기고 있다.
  sentences:
  - 한 남자가 수영장으로 뛰어들었다.
  - 한 여성이 프라이팬에 노란 혼합물을 부어 넣고 있다.
  - 두 마리의 갈색 개가 눈 속에서 서로 놀고 있다.
- source_sentence: 버스가 바쁜 길을 따라 운전한다.
  sentences:
  - 우리와 같은 태양계가 은하계 밖에서 존재할 수도 있을 것입니다.
  - 그 여자는 데이트하러 가는 중이다.
  - 녹색 버스가 도로를 따라 내려간다.
datasets:
- x2bee/Korean_NLI_dataset
- CocoRoF/sts_dev
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
- pearson_euclidean
- spearman_euclidean
- pearson_manhattan
- spearman_manhattan
- pearson_dot
- spearman_dot
- pearson_max
- spearman_max
model-index:
- name: SentenceTransformer based on x2bee/KoModernBERT-base-mlm-v03-ckp00
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: sts dev
      type: sts_dev
    metrics:
    - type: pearson_cosine
      value: 0.6463764324668821
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.668749120795344
      name: Spearman Cosine
    - type: pearson_euclidean
      value: 0.6434649881382908
      name: Pearson Euclidean
    - type: spearman_euclidean
      value: 0.6535107003038169
      name: Spearman Euclidean
    - type: pearson_manhattan
      value: 0.6516759845194007
      name: Pearson Manhattan
    - type: spearman_manhattan
      value: 0.6679435004022668
      name: Spearman Manhattan
    - type: pearson_dot
      value: 0.6306152465572834
      name: Pearson Dot
    - type: spearman_dot
      value: 0.6496717700503837
      name: Spearman Dot
    - type: pearson_max
      value: 0.6516759845194007
      name: Pearson Max
    - type: spearman_max
      value: 0.668749120795344
      name: Spearman Max
---

# SentenceTransformer based on x2bee/KoModernBERT-base-mlm-v03-ckp00

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [x2bee/KoModernBERT-base-mlm-v03-ckp00](https://huggingface.co/x2bee/KoModernBERT-base-mlm-v03-ckp00) on the [korean_nli_dataset](https://huggingface.co/datasets/x2bee/Korean_NLI_dataset) dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [x2bee/KoModernBERT-base-mlm-v03-ckp00](https://huggingface.co/x2bee/KoModernBERT-base-mlm-v03-ckp00) <!-- at revision addb15798678d7f76904915cf8045628d402b3ce -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - [korean_nli_dataset](https://huggingface.co/datasets/x2bee/Korean_NLI_dataset)
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
model = SentenceTransformer("x2bee/sts_nli_tune_test")
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

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `sts_dev`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric             | Value      |
|:-------------------|:-----------|
| pearson_cosine     | 0.6464     |
| spearman_cosine    | 0.6687     |
| pearson_euclidean  | 0.6435     |
| spearman_euclidean | 0.6535     |
| pearson_manhattan  | 0.6517     |
| spearman_manhattan | 0.6679     |
| pearson_dot        | 0.6306     |
| spearman_dot       | 0.6497     |
| pearson_max        | 0.6517     |
| **spearman_max**   | **0.6687** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### korean_nli_dataset

* Dataset: [korean_nli_dataset](https://huggingface.co/datasets/x2bee/Korean_NLI_dataset) at [ef305ef](https://huggingface.co/datasets/x2bee/Korean_NLI_dataset/tree/ef305ef8e2d83c6991f30f2322f321efb5a3b9d1)
* Size: 392,702 training samples
* Columns: <code>sentence1</code>, <code>sentence2</code>, and <code>score</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence1                                                                         | sentence2                                                                         | score                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                          |
  | details | <ul><li>min: 4 tokens</li><li>mean: 35.7 tokens</li><li>max: 194 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 19.92 tokens</li><li>max: 64 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.48</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence1                                                                                                                                           | sentence2                                         | score            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------|:-----------------|
  | <code>개념적으로 크림 스키밍은 제품과 지리라는 두 가지 기본 차원을 가지고 있다.</code>                                                                                             | <code>제품과 지리학은 크림 스키밍을 작동시키는 것이다.</code>          | <code>0.5</code> |
  | <code>시즌 중에 알고 있는 거 알아? 네 레벨에서 다음 레벨로 잃어버리는 거야 브레이브스가 모팀을 떠올리기로 결정하면 브레이브스가 트리플 A에서 한 남자를 떠올리기로 결정하면 더블 A가 그를 대신하러 올라가고 A 한 명이 그를 대신하러 올라간다.</code> | <code>사람들이 기억하면 다음 수준으로 물건을 잃는다.</code>           | <code>1.0</code> |
  | <code>우리 번호 중 하나가 당신의 지시를 세밀하게 수행할 것이다.</code>                                                                                                      | <code>우리 팀의 일원이 당신의 명령을 엄청나게 정확하게 실행할 것이다.</code> | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Evaluation Dataset

#### sts_dev

* Dataset: [sts_dev](https://huggingface.co/datasets/CocoRoF/sts_dev) at [1de0cdf](https://huggingface.co/datasets/CocoRoF/sts_dev/tree/1de0cdfb2c238786ee61c5765aa60eed4a782371)
* Size: 1,500 evaluation samples
* Columns: <code>text</code>, <code>pair</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | text                                                                              | pair                                                                              | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                          |
  | details | <ul><li>min: 7 tokens</li><li>mean: 20.38 tokens</li><li>max: 52 tokens</li></ul> | <ul><li>min: 6 tokens</li><li>mean: 20.52 tokens</li><li>max: 54 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.42</li><li>max: 1.0</li></ul> |
* Samples:
  | text                                 | pair                                | label             |
  |:-------------------------------------|:------------------------------------|:------------------|
  | <code>안전모를 가진 한 남자가 춤을 추고 있다.</code> | <code>안전모를 쓴 한 남자가 춤을 추고 있다.</code> | <code>1.0</code>  |
  | <code>어린아이가 말을 타고 있다.</code>         | <code>아이가 말을 타고 있다.</code>          | <code>0.95</code> |
  | <code>한 남자가 뱀에게 쥐를 먹이고 있다.</code>    | <code>남자가 뱀에게 쥐를 먹이고 있다.</code>     | <code>1.0</code>  |

</details>


### Framework Versions
- Python: 3.11.10
- Sentence Transformers: 3.3.1
- Transformers: 4.48.0
- PyTorch: 2.5.1+cu124
- Accelerate: 1.2.1
- Datasets: 3.2.0
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->