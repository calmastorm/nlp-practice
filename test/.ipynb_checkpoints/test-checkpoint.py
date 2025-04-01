import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# 加载数据
train = pd.read_csv('data/datasets/train.tsv', sep='\t')
test = pd.read_csv('data/datasets/test.tsv', sep='\t')

# 取出字段
X_train = train['Phrase'].fillna("")
y_train = train['Sentiment']
X_test = test['Phrase'].fillna("")
phrase_ids = test['PhraseId']

# 构建 BoW 特征（unigram，不加 min/max df）
vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=None)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# 配置模型
model_configs = [
    {
        'name': 'baseline',
        'model': LogisticRegression(C=1.0, max_iter=5000),
        'filename': 'submission_baseline.csv'
    },
    {
        'name': 'lr_balanced',
        'model': LogisticRegression(C=0.5, max_iter=5000, class_weight='balanced'),
        'filename': 'submission_lr_balanced.csv'
    },
    {
        'name': 'svc_balanced',
        'model': LinearSVC(C=0.05, max_iter=5000, class_weight='balanced'),
        'filename': 'submission_svc_balanced.csv'
    }
]

# 训练并生成提交文件
for config in model_configs:
    print(f"Training model: {config['name']}")
    model = config['model']
    model.fit(X_train_bow, y_train)
    y_pred = model.predict(X_test_bow)

    submission = pd.DataFrame({
        'PhraseId': phrase_ids,
        'Sentiment': y_pred
    })
    submission.to_csv(config['filename'], index=False)
    print(f"Saved to: {config['filename']} ✅")

print("\nAll submissions generated. Ready to submit to Kaggle 🚀")
