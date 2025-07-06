# Optional: Simplified dummy version of evaluation logic
from retriever import QuoteRetriever
from generator import generate_answer

def test_queries(queries, retriever):
    for q in queries:
        context = retriever.retrieve(q)
        answer = generate_answer(context, q)
        print("\nQuery:", q)
        print("Answer:", answer)