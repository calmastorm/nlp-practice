import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from project1.src.data_process import read_data
from project1.src.feature_extraction import BagOfWords, NGrams
from project1.src.softmax_regression import SoftmaxRegression
