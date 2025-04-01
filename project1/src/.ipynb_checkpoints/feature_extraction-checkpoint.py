import numpy as np

# Model 1: Bag of Words
class BagOfWords:

    # What vocabs are in the dataset and how many of them are there
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
    
    # fit method is used to learn the vocabulary from the dataset
    # gives each word in the dataset a unique index
    def fit(self, X):
        for doc in X:
            for word in doc.strip().split(' '):
                word = word.lower()  # Convert word to lower case
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.vocab_size += 1

    # transform method is used to convert the dataset into a matrix
    # It counts the number of times each word appears in the document (frequency)
    def transform(self, X):
        X_transformed = np.zeros((len(X), self.vocab_size))
        for i, doc in enumerate(X):
            for word in doc.strip().split(' '):
                word = word.lower()
                if word in self.vocab:
                    X_transformed[i, self.vocab[word]] += 1
        return X_transformed
    
    # fit_transform method is used to learn the vocabulary and convert the dataset into a matrix
    # basically the combination of fit and transform methods
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
# Model 2: N-grams
class NGrams:
    # ngram is an integer
    def __init__(self, ngram):
        self.ngram = ngram
        self.vocab = {}
        self.vocab_size = 0
    
    # if feature is not in the vocabulary, add it to the vocabulary
    # feature is a sequence of n words
    def fit(self, X):
        for gram in self.ngram:
            for sentence in X:
                sentence = sentence.lower()
                sentence = sentence.split(' ')
                for i in range(len(sentence) - gram + 1):
                    feature = '_'.join(sentence[i:i+gram])
                    if feature not in self.vocab:
                        self.vocab[feature] = self.vocab_size
                        self.vocab_size += 1
    
    # transform method convert dataset into a matrix
    def transform(self, X):
        n = len(X)
        m = len(self.vocab)
        ngram_feature = np.zeros((n, m))
        for i, sentence in enumerate(X):
            sentence = sentence.lower()
            sentence = sentence.split(' ')
            for gram in self.ngram:
                for j in range(len(sentence) - gram + 1):
                    feature = '_'.join(sentence[j:j+gram])
                    if feature in self.vocab:
                        ngram_feature[i, self.vocab[feature]] += 1
        return ngram_feature
    
    # fit_transform method is combination of fit and transform methods
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
if __name__ == '__main__':
    bow = BagOfWords()
    # Unigram and Bigram at the same time
    gram = NGrams(ngram=[1, 2])
    sentences = ['I love machine learning', 'I love deep learning']
    transformed = bow.fit_transform(sentences)
    print('Bag of words features')
    print(bow.vocab)
    print(transformed)
    feature = gram.fit_transform(sentences)
    print('Ngrams features')
    print(gram.vocab)
    print(feature)