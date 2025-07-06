from datasets import load_dataset
import pandas as pd

def load_and_prepare_data():
    dataset = load_dataset("Abirate/english_quotes")
    df = pd.DataFrame(dataset['train'])
    df.dropna(inplace=True)
    df = df[df['quote'].str.strip() != ""]
    df['quote_clean'] = df['quote'].str.lower().str.strip()
    df['author_clean'] = df['author'].str.lower().str.strip()
    df['tags_clean'] = df['tags'].apply(lambda x: ', '.join([tag.lower() for tag in x]))
    df['combined_text'] = df['quote_clean'] + " - " + df['author_clean'] + " [" + df['tags_clean'] + "]"
    return df