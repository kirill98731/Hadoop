import pandas as pd
import random

def get_data():
    df = pd.read_csv('data/IMDB Dataset.csv')
    random_index = random.sample(set(df.index), int(len(df.index)*0.7))
    df = df.loc[random_index]
    df.to_csv('data/train.csv')


if __name__ == "__main__":
    get_data()
