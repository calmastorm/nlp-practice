import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '../data/datasets/train.tsv')

def read_data(file_path):
    try:
        train_df = pd.read_csv(file_path, sep='\t')
        # test_df = pd.read_csv(test_file)
        return train_df['Phrase'], train_df['Sentiment'].values
    except Exception as e:
        print(f"Error reading the data from {file_path}: {e}")
        return None
    
if __name__ == '__main__':
    X_data, y_data = read_data(file_path)
    print('train size ', len(X_data))