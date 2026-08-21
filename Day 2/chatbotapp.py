import os
import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader

os.environ["GROQ_API_KEY"]="YOUR_GROQ_API_KEY"
llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-20b"
)
client=chromadb.Client()
collection=client.get_or_create_collection("carrer_knowledge_base")
def ingest_pdf(file):
    reader=PdfReader(file)
    text=""
    for page in reader.pages:
        text += page.extract_text()+"\n"
    return text
st.title("Carrer Guidance Chatbot")
st.markdown("Get Personalized,grounded career advice using RAG+LLM.")
uploaded_files = st.file_uploader("Upload career Resources (PDFs) ", type=["pdf"],accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        text = ingest_pdf(file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=50)
        chunks=splitter.split_text(text)
        collection.add(
            documents=chunks,
            ids=[f"{file.name}_{i}"for i in range(len(chunks))]
        )
    st.success(f"{len(uploaded_files)} PDFs ingested successfully!")
user_query = st.text_input("Ask you career question:")
if st.button("Get Advice") and user_query:
    vector_results=collection.query(
        query_texts=user_query,
        n_results=5
    )
    vector_docs=vector_results["documents"][0]
    keywords=user_query.lower().split()
    keyword_docs=[
        doc for doc in vector_docs
        if any(k in doc.lower()
            for k in keywords)
    ]
    hybrid_docs=list(set(vector_docs+keyword_docs))
    rerank_prompt=PromptTemplate.from_template(
        """
        User query:
        {query}
        
        Documents:
        {docs}
        
        Rank these documents from most relevant to least relevant
        for providing career advice, including skills and companies.
        Return the ranked list.
        """

    )
    rerank_chain=rerank_prompt|llm
    reranked_output = rerank_chain.invoke({
        "query": user_query,
        "docs": "\n".join(hybrid_docs)
    })
    top_context=reranked_output.content.split("\n")[:3]
    final_prompt=PromptTemplate.from_template(
        """"
        You are a career guidance AI assistant.
        
        Based on the following resources:
        
        {context}
        
        Provide a personalized roadmap for the user:
        -Skills to learn
        -Recommend companies
        -Steps to improve career readiness
        
        User Query:
        {query} 
        """
    )
    rag_chain=final_prompt|llm
    career_advice=rag_chain.invoke({
        "context":"\n".join(top_context),
        "query": user_query
    })
    st.subheader("📄 Top Retrieved Context")
    for doc in top_context:
        st.write("-",doc)
    st.subheader("🧭 Personalized Career Advice")
    st.write(career_advice.content)

