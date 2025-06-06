import numpy as np

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    z = np.exp(z)
    return z / np.sum(z, axis=1, keepdims=True)

class SoftmaxRegression:
    # initialize the model
    def __init__(self):
        self.num_classes = None
        self.n = None # num data
        self.m = None # num features
        self.weight = None
        self.learning_rate = None

    # fit method is to train the model
    def fit(self, X, y, 
            # set hyperparameters
            learning_rate=0.01, 
            epochs=10, 
            num_classes=5, 
            print_loss_step=5, # print loss every print_loss_step epochs, -1 means not print
            update_strategy='batch'):
        
        self.n, self.m = X.shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight = np.random.randn(self.num_classes, self.m) # Num of classes x Num of features

        # one-hot encoding
        y_one_hot = np.zeros((self.n, self.num_classes)) # Number of data x Number of classes
        for i in range(self.n):
            y_one_hot[i, y[i]] = 1

        loss_history = []
        
        # train (基本上锁定是这里面的问题了把)
        for epoch in range(epochs):
            loss = 0
            # batch update strategy
            if update_strategy == 'batch':
                # prob = softmax(np.dot(self.weight, X.T)) # 使用当前的权重计算概率，prob尺寸和y_one_hot一样
                # prob = X.dot(self.weight.T) # 这个写法ok的 没有问题
                prob = np.dot(X, self.weight.T) # 这个写法也ok的 没有问题
                prob = softmax(prob) # 注意这里不需要flatten
                for i in range(self.n):
                    loss -= np.log(prob[i][y[i]]) # 计算并更新交叉熵损失
                weight_update = np.zeros_like(self.weight) # 初始化权重更新 zeros_like返回一个和输入形状相同的全零矩阵
                # 计算权重更新，这里的计算是对所有数据点的计算，即批量更新。
                for i in range(self.n):
                    # 公式是：W = W + lr * X.T * (y_one_hot - prob)
                    weight_update += X[i].reshape(1, self.m).T.dot((y_one_hot[i] - prob[i]).reshape(1, self.num_classes)).T
                self.weight += self.learning_rate * weight_update / self.n
            # stochastic update strategy
            elif update_strategy == 'stochastic':
                '''
                rand_index = np.arange(len(X))
                np.random.shuffle(rand_index)
                for index in list(rand_index):
                    Xi = X[index].reshape(1, -1) # reshape(1, -1) means convert to 2D array
                    # prob = Xi.dot(self.weight.T) # 这个写法ok的 没有问题
                    prob = np.dot(Xi, self.weight.T) # 这个写法也ok的 没有问题
                    # prob = softmax(prob).flatten() 这个是基本flatten的写法
                    # prob = softmax(prob).reshape(-1,) 使用reshape(-1,)等效于flatten
                    prob = softmax(prob).reshape(-1) # 使用reshape(-1)也等效于flatten
                    loss += -np.log(prob[y[index]])
                    self.weight += Xi.reshape(1, self.m).T.dot((y_one_hot[index] - prob).reshape(1, self.num_classes)).T
                '''
                rand_index = np.arange(len(X))
                np.random.shuffle(rand_index)
                batch_size = 64
                for i in range(0, len(X), batch_size):
                    batch_indices = rand_index[i:i+batch_size]
                    X_batch = X[batch_indices]
                    y_batch = y_one_hot[batch_indices]
                    prob = softmax(np.dot(X_batch, self.weight.T))
                    loss += -np.sum(np.log(prob[np.arange(len(batch_indices)), y[batch_indices]]))
                    self.weight += (X_batch.T.dot(y_batch - prob)).T / batch_size

            else:
                raise ValueError('Invalid update strategy')
            
            loss /= self.n
            loss_history.append(loss) # 完成了一个epoch 记录loss
            if print_loss_step > 0 and epoch % print_loss_step == 0: # 检查是否需要打印loss
                print(f'Epoch {epoch}, loss {loss}')
        return loss_history # 这里不需要weight，因为weight是类的属性，可以直接访问，比如下面的predict方法。
    
    def predict(self, X):
        prob = softmax(np.dot(self.weight, X.T))
        return np.argmax(prob, axis=0)
    
    def score(self, X, y): # 得到的值是准确率
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    