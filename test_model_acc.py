import os
os.chdir("/workspace")
from models.bert_cls import plateer_classifier;
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm

data = pd.read_csv("./data/data_etc/sample_01.csv")
sample = data.sample(500)
labels = list(sample['category'])
text_list = list(sample['goods_nm'])

def plateer_model_use(text):
    result = plateer_classifier(text, top_k=5)[0]
    return result[0]["label_decode"], result[1]["label_decode"], result[2]["label_decode"], result[3]["label_decode"], result[4]["label_decode"]

results = list(map(lambda x: plateer_model_use(x), tqdm(text_list, desc="Processing", total=len(text_list))))
qwen_output_1, qwen_output_2, qwen_output_3, qwen_output_4, qwen_output_5 = zip(*results)

top1_accuracy_sklearn = accuracy_score(labels, qwen_output_1)
top2_accuracy_sklearn = accuracy_score(labels, [
    true if true in (top1, top2) else -1
    for true, top1, top2 in zip(labels, qwen_output_1, qwen_output_2)
])
top3_accuracy_sklearn = accuracy_score(labels, [
    true if true in (top1, top2, top3) else -1
    for true, top1, top2, top3 in zip(labels, qwen_output_1, qwen_output_2, qwen_output_3)
])
top4_accuracy_sklearn = accuracy_score(labels, [
    true if true in (top1, top2, top3, top4) else -1
    for true, top1, top2, top3, top4 in zip(labels, qwen_output_1, qwen_output_2, qwen_output_3, qwen_output_4)
])
top5_accuracy_sklearn = accuracy_score(labels, [
    true if true in (top1, top2, top3, top4, top5) else -1
    for true, top1, top2, top3, top4, top5 in zip(labels, qwen_output_1, qwen_output_2, qwen_output_3, qwen_output_4, qwen_output_5)
])

print(f"Top-1 Accuracy : {top1_accuracy_sklearn:.4f}")
print(f"Top-2 Accuracy : {top2_accuracy_sklearn:.4f}")
print(f"Top-3 Accuracy : {top3_accuracy_sklearn:.4f}")
print(f"Top-4 Accuracy : {top4_accuracy_sklearn:.4f}")
print(f"Top-5 Accuracy : {top5_accuracy_sklearn:.4f}")

print("qwen_output")
accuracy = accuracy_score(labels, qwen_output_1)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(labels, qwen_output_1))