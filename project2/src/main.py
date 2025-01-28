from torch import optim
import torch
from models import TextRNN, TextCNN
# from dataloader_torchtext import dataset2dataloader
from dataloader_hand import make_dataloader
import numpy as np

if __name__ == '__main__':
    model_names = ['RNN', 'CNN'] # LSTM修不好
    learning_rate = 1e-3
    epochs = 10
    num_classes = 5
    load_data_by_torchtext = False # torchtext有版本问题

    if load_data_by_torchtext:
        # train_iter, val_iter, word_vectors = dataset2dataloader(batch_size=8, debug=True)
        pass
    else:
        train_iter, val_iter, word_vectors, X_lang = make_dataloader(batch_size=8, debug=True)

    for model_name in model_names:
        print(f'Making {model_name} model... ')
        if model_name == 'RNN':
            model = TextRNN(vocab_size=len(word_vectors), embedding_dim=50, hidden_size=64, num_classes=num_classes, weights=word_vectors)
        elif model_name == 'CNN':
            model = TextCNN(vocab_size=len(word_vectors), embedding_dim=50, num_classes=num_classes, embedding_vectors=word_vectors)
        elif model_name == 'LSTM':
            model = TextRNN(vocab_size=len(word_vectors), embedding_dim=50, hidden_size=64, num_classes=num_classes, weights=word_vectors, rnn_type='LSTM')
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fun = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs): # 每个epoch
            model.train() # 训练
            for i, batch in enumerate(train_iter):
                if load_data_by_torchtext:
                    x, y = batch.sent.t(), batch.label
                else:
                    x, y, lens = batch
                logits = model(x)
                optimizer.zero_grad()
                loss = loss_fun(logits, y)
                loss.backward()
                optimizer.step()

            # with torch.no_grad()
            model.eval()
            train_accs = []
            for i, batch in enumerate(train_iter):
                if load_data_by_torchtext:
                    x, y = batch.sent.t(), batch.label
                else:
                    x, y, lens = batch
                _, y_pre = torch.max(logits, -1)
                acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
                train_accs.append(acc)
            train_acc = np.array(train_accs).mean()

            # 验证
            val_accs = []
            for i, batch in enumerate(val_iter):
                if load_data_by_torchtext:
                    x, y, batch.sent.t(), batch.label
                else:
                    x, y, lens = batch
                logits = model(x)
                _, y_pre = torch.max(logits, -1)
                acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
                val_accs.append(acc)
            val_acc = np.array(val_accs).mean()
            print(f'Epoch {epoch} train acc: {train_acc:2f} val acc: {val_acc:2f}%')
            if train_acc >= 0.99:
                break