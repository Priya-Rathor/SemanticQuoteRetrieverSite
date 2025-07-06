# 💬 Semantic Quote Retrieval with RAG

A Retrieval-Augmented Generation (RAG) system that intelligently fetches and explains English quotes using a fine-tuned semantic search model and OpenAI's GPT-4.

---

## 🚀 Project Overview

This project allows users to:

- Ask natural language questions about quotes (e.g., _"What did Oscar Wilde say about hope?"_)
- Retrieve semantically similar quotes using a trained vector-based retriever
- Generate contextual answers using GPT-4
- Interact with everything via a simple and elegant Streamlit web app

---

## 🧠 How It Works

### 🔹 1. Data Preparation
- Loads the [`Abirate/english_quotes`](https://huggingface.co/datasets/Abirate/english_quotes) dataset
- Cleans and combines `quote`, `author`, and `tags` into one semantic field

### 🔹 2. Model Fine-Tuning
- Fine-tunes a `SentenceTransformer` model (`all-MiniLM-L6-v2`) to create embeddings tailored to quote similarity

### 🔹 3. Retrieval
- Uses **FAISS** for fast vector similarity search
- Retrieves top relevant quotes based on the user query

### 🔹 4. Generation
- Sends the retrieved quotes to **GPT-4** via OpenAI API
- GPT generates an answer grounded in retrieved context

### 🔹 5. Web UI
- Built with **Streamlit**
- Type a query and get immediate, explainable responses

---

## 🛠️ Setup Instructions

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-quote-retriever.git
cd rag-quote-retriever
✅ 2. Create Environment and Install Dependencies
bash
Copy
Edit
python setup_env.py
source rag_env/bin/activate
✅ 3. Train the Quote Embedding Model
python
Copy
Edit
from data_preparation import load_and_prepare_data
from model import train_and_save_model

df = load_and_prepare_data()
train_and_save_model(df)
✅ 4. Set Your OpenAI API Key
Create a .env file or export it in your shell:

bash
Copy
Edit
export OPENAI_API_KEY=your-api-key-here
✅ 5. Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
📸 App Preview
<!-- Optional: Add a screenshot here --> <!-- ![App Screenshot](screenshot.png) -->
🧪 Evaluation (Optional)
Test the RAG pipeline using example queries:

python
Copy
Edit
from data_preparation import load_and_prepare_data
from retriever import QuoteRetriever
from evaluate_rag import test_queries

df = load_and_prepare_data()
retriever = QuoteRetriever("fine-tuned-quote-model", df)
test_queries(["life quotes by Albert Einstein"], retriever)
📦 Project Structure
bash
Copy
Edit
├── app.py                # Streamlit user interface
├── setup_env.py          # Creates virtual environment and installs dependencies
├── data_preparation.py   # Cleans and prepares dataset
├── model.py              # Fine-tunes sentence-transformer
├── retriever.py          # Embeds and indexes quotes using FAISS
├── generator.py          # Uses GPT-4 to generate final answers
├── evaluate_rag.py       # Evaluation script for RAG performance
└── requirements.txt      # (Optional) Can be auto-generated from environment
📌 Requirements
Python 3.8+

OpenAI API Key

Linux/macOS (or Windows WSL)

🙌 Acknowledgements
Abirate/english_quotes Dataset

SentenceTransformers

FAISS

OpenAI GPT-4

Streamlit

