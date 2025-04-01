import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score

np.random.seed(42)

# 加载数据
data = pd.read_csv('data/datasets/train.tsv', sep='\t')
X = data['Phrase']
y = data['Sentiment']

# 使用 TF-IDF 特征（带 unigram 权重）
vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),        # 只用 unigram
    stop_words='english',      # 去掉常见功能词
    max_df=0.95,               # 出现在 95% 以上的词将被去除
    min_df=2,                  # 出现少于 2 次的词也被去除
    max_features=None          # 可设置 20000~50000 限制维度
)

X_tfidf = vectorizer.fit_transform(X)

# 不同正则化强度下的测试
C_values = [0.01, 0.05, 0.1, 0.5, 1]
results = []

for c in C_values:
    model = LogisticRegression(C=c, max_iter=5000, random_state=42)
    y_pred = cross_val_predict(model, X_tfidf, y, cv=5)
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    
    results.append({'C': c, 'CV Accuracy': acc, 'Macro-F1': f1})

df_results = pd.DataFrame(results)
print(df_results)
