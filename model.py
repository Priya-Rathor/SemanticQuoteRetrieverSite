from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

def train_and_save_model(df, output_path="fine-tuned-quote-model"):
    train_examples = [
        InputExample(texts=[row['combined_text'], row['quote_clean']]) for _, row in df.iterrows()
    ]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    model.save(output_path)
    return output_path