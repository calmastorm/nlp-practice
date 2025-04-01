import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 固定随机种子
np.random.seed(42)

# 1. 加载数据
data = pd.read_csv('data/datasets/train.tsv', sep='\t')
X = data['Phrase']
y = data['Sentiment']

# 2. 生成N-gram特征（唯一修改处！）
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=None)
X_ngram = vectorizer.fit_transform(X)

# 3. 交叉验证（与原代码一致）
model = LogisticRegression(max_iter=5000, random_state=42)
y_pred = cross_val_predict(model, X_ngram, y, cv=5)

# 4. 评估与可视化（与原代码一致）
print("Accuracy:", accuracy_score(y, y_pred))
print("Macro-F1:", f1_score(y, y_pred, average='macro'))

# 混淆矩阵（颜色=正确率，数字=数量+占比）
cm = confusion_matrix(y, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
annot_labels = [f"{cm[i,j]}\n({cm_normalized[i,j]:.1%})" for i in range(5) for j in range(5)]
annot_labels = np.array(annot_labels).reshape(5,5)

plt.figure(figsize=(10,8))
sns.heatmap(
    cm_normalized,
    annot=annot_labels,
    fmt="",
    cmap="Blues",
    cbar_kws={"label": ""},
    xticklabels=['0','1','2','3','4'],
    yticklabels=['0','1','2','3','4']
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("N-gram (Unigram+Bigram) Confusion Matrix", fontsize=14, pad=20)
plt.savefig("confusion_matrix_ngram.png", dpi=300, bbox_inches="tight")
plt.show()