import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 固定随机种子确保可复现
np.random.seed(42)

# 1. 加载数据
data = pd.read_csv('data/datasets/train.tsv', sep='\t')
print("data samples:\n", data.head())

# 2. 提取特征和标签
X = data['Phrase']  # 文本列
y = data['Sentiment']  # 标签列（0-4）
print("label dist:\n", y.value_counts())

# 3. 生成BoW特征
vectorizer = CountVectorizer(max_features=None)  # 限制特征数量
X_bow = vectorizer.fit_transform(X)

# 4. 交叉验证（同时生成预测结果用于评估）
model = LogisticRegression(max_iter=1000, random_state=42)
y_pred = cross_val_predict(model, X_bow, y, cv=5)  # 5折CV

# 5. 评估指标
print("Accuracy:", accuracy_score(y, y_pred))
print("Macro-F1:", f1_score(y, y_pred, average='macro'))

# 3. 计算原始混淆矩阵和归一化混淆矩阵
cm = confusion_matrix(y, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 行归一化

# 4. 自定义注释文本（格式：数量\n占比百分比）
annot_labels = []
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot_labels.append(f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})")  # 数量 + 百分比
annot_labels = np.array(annot_labels).reshape(cm.shape)

# 5. 绘制混淆矩阵
plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    cm_normalized,
    annot=annot_labels,  # 显示自定义文本
    fmt="",             # 禁用默认格式（因手动生成annot_labels）
    cmap="Blues",
    cbar_kws={"label": ""},  # 右侧color bar名称
    xticklabels=['0', '1', '2', '3', '4'],
    yticklabels=['0', '1', '2', '3', '4']
)

# 6. 添加标题和标签
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("BoW Confusion Matrix", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("confusion_matrix_bow_normalized_with_percentage.png", dpi=300, bbox_inches="tight")
plt.show()