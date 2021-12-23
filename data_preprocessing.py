import re
import pandas as pd

def cleanhtml(raw_html):
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext

def preprocessing(data_path):
    df = pd.read_csv(data_path)
    df['review'] = df['review'].apply(lambda x: cleanhtml(x))
    df.to_csv(data_path)

if __name__ == "__main__":
    preprocessing('data/train.csv')