import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score

# 固定随机种子
np.random.seed(42)

# 1. 加载数据
data = pd.read_csv('data/datasets/train.tsv', sep='\t')
X = data['Phrase']
y = data['Sentiment']

# 2. 生成 BoW 特征（unigram）
vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=None)
X_bow = vectorizer.fit_transform(X)

# 3. 不同的正则化参数 C
C_values = [0.01, 0.05, 0.1, 0.5, 1]
results = []

for c in C_values:
    model = LinearSVC(C=c, max_iter=5000, class_weight='balanced')
    y_pred = cross_val_predict(model, X_bow, y, cv=5)
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    
    results.append({'C': c, 'CV Accuracy': acc, 'Macro-F1': f1})

# 4. 输出结果表格
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
