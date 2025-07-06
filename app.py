import streamlit as st
from data_preparation import load_and_prepare_data
from retriever import QuoteRetriever
from generator import generate_answer

st.title("📈 Semantic Quote Retriever")

query = st.text_input("Ask a quote-related question")

@st.cache_resource
def load_retriever():
    df = load_and_prepare_data()
    return QuoteRetriever("fine-tuned-quote-model", df), df

retriever, df = load_retriever()



if query:
    with st.spinner("Retrieving relevant quotes and generating answer..."):
        context_quotes = retriever.retrieve(query)
        answer = generate_answer(context_quotes, query)
        st.write("### Answer")
        st.success(answer)

        st.write("\n### Retrieved Quotes")
        for quote in context_quotes:
            st.info(quote)
