import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

# 固定随机种子
np.random.seed(42)
torch.manual_seed(42)

# 1. 加载数据
data = pd.read_csv('data/datasets/train.tsv', sep='\t')
X = data['Phrase']
y = data['Sentiment'].values  # 转换为numpy数组

# 2. 生成TF-IDF特征
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_tfidf = vectorizer.fit_transform(X).astype(np.float32)  # 转换为float32

# 3. 定义PyTorch模型
class TextCNN(nn.Module):
    def __init__(self, input_dim, C=1.0):
        super().__init__()
        self.main = nn.Sequential(
            nn.Unflatten(1, (input_dim, 1)),  # 等价于Reshape
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        # L2正则化通过optimizer的weight_decay实现
        self.C = C

    def forward(self, x):
        return self.main(x)

# 4. 包装为sklearn兼容的Classifier
def make_net(C):
    return NeuralNetClassifier(
        module=TextCNN,
        module__input_dim=X_tfidf.shape[1],
        module__C=C,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=1.0/C,  # L2正则化
        max_epochs=10,
        batch_size=64,
        callbacks=[EarlyStopping(patience=2)],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=0
    )

# 5. 不同的正则化参数C
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
results = []

for c in C_values:
    model = make_net(C=c)
    y_pred = cross_val_predict(model, X_tfidf, y, cv=5, n_jobs=1)
    
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    
    results.append({'C': c, 'CV Accuracy': acc, 'Macro-F1': f1})

# 6. 输出结果表格
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))