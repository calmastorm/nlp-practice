import numpy as np

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    z = np.exp(z)
    return z / np.sum(z, axis=1, keepdims=True)

class SoftmaxRegression:
    def __init__(self):
        self.num_classes = None
        self.n = None # num data
        self.m = None # num features
        self.weight = None
        self.learning_rate = None

    def fit(self, X, y, learning_rate=0.01, epochs=10, num_classes=5, print_loss_step=-1, update_strategy='batch'):
        self.n, self.m = X.shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight = np.random.randn(self.num_classes, self.m)

        # one-hot encoding
        y_one_hot = np.zeros((self.n, self.num_classes))
        for i in range(self.n):
            y_one_hot[i, y[i]] = 1

        loss_history = []

        for epoch in range(epochs):
            loss = 0
            # batch update strategy
            if update_strategy == 'batch':
                prob = softmax(np.dot(self.weight, X.T))
                for i in range(self.n):
                    loss -= np.log(prob[y[i], i])
                weight_update = np.zeros_like(self.weight)
                for i in range(self.n):
                    weight_update += X[i].reshape(-1, self.m).T.dot((y_one_hot[i] - prob[i]).reshape(1, self.num_classes)).T
                self.weight += self.learning_rate * weight_update / self.n
            # stochastic update strategy
            elif update_strategy == 'stochastic':
                rand_index = np.arrange(len(X))
                np.random.shuffle(rand_index)
                for index in list(rand_index):
                    Xi = X[index].reshape(1, -1) # reshape(1, -1) means convert to 2D array
                    prob = softmax(np.dot(self.weight, Xi.T))
                    prob = prob.reshape(-1) # reshape(-1) means convert to 1D array
                    loss += -np.log(prob[y[index]])
                    self.weight += Xi.reshape(1, self.m).T.dot((y_one_hot[index] - prob).reshape(1, self.num_classes)).T
            else:
                raise ValueError('Invalid update strategy')
            
            loss /= self.n
            loss_history.append(loss)
            if print_loss_step > 0 and epoch % print_loss_step == 0:
                print(f'Epoch {epoch}, loss {loss}')
        return loss_history
    
    def predict(self, X):
        prob = softmax(np.dot(self.weight, X.T))
        return np.argmax(prob, axis=0)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)