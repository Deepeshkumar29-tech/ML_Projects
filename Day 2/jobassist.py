import os
import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"
llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-20b"
)
client=chromadb.Client()
collection=client.get_or_create_collection(
    "job_matching_knowledge_base"
)
def ingest_resume(file):
    reader=PdfReader(file)
    text=""
    for page in reader.pages:
        page_text=page.extract_text()
        if page_text:
            text+=page_text+"\n"
    return text
st.title("AI-Powered Job Matching")
st.markdown(
    "Match job descriptions with candidate resumes "
    "using RAG + LLM."
)
uploaded_files=st.file_uploader(
    "Upload Candidate Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)
if uploaded_files:
    for file in uploaded_files:
        text=ingest_resume(file)
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50
        )
        chunks=splitter.split_text(text)
        collection.add(
            documents=chunks,
            ids=[
                f"{file.name}_{i}"
                for i in range(len(chunks))
            ]
        )
    st.success(
        f"{len(uploaded_files)} resumes ingested successfully!"
    )
job_description=st.text_area(
    "Enter Job Description:"
)
if st.button("Find Matching Candidates") and job_description:
    vector_results = collection.query(
        query_texts=[job_description],
        n_results=5
    )
    vector_docs=vector_results["documents"][0]
    keywords=job_description.lower().split()
    keyword_docs=[
        doc
        for doc in vector_docs
        if any(
            k in doc.lower()
            for k in keywords
        )
    ]
    hybrid_docs=list(
        set(vector_docs+keyword_docs)
    )
    rerank_prompt=PromptTemplate.from_template(
        """
        You are an AI-powered recruitment assistant.

        Job Description:
        {query}

        Candidate Resumes:
        {docs}

        Task:

        Analyze each candidate resume against the job description.

        Rank the candidates from most suitable to least suitable.

        Consider:

        - Technical skills
        - Programming languages
        - Machine learning and AI skills
        - Frameworks and libraries
        - Relevant projects
        - Work experience
        - Education
        - Overall relevance to the job

        For each candidate, provide:

        1. Candidate name
        2. Matching skills
        3. Relevant experience/projects
        4. Overall match score out of 100
        5. Short reason for the ranking

        Return the candidates in descending order of suitability.

        Important:
        - Only use information present in the resumes.
        - Do not invent skills or experience.
        - Compare candidates specifically against the job description.
        """
    )
    rerank_chain=rerank_prompt|llm
    reranked_output=rerank_chain.invoke(
        {
            "query":job_description,
            "docs":"\n".join(hybrid_docs)
        }
    )
    top_context=reranked_output.content
    final_prompt=PromptTemplate.from_template(
        """
        You are an AI-powered job matching assistant.

        Your task is to explain why the selected candidates
        are suitable for the given job.

        Job Description:
        {query}

        Top Matching Candidates:
        {context}

        For each candidate, provide:

        1. Candidate Name
        2. Match Score
        3. Matching Skills
        4. Relevant Projects or Experience
        5. Why the candidate is a good fit
        6. Any important skill gaps

        Format the response like this:

        Job: [Job Title]

        Top Candidates:

        1. [Candidate Name]

           Match Score: [Score]%

           Matching Skills:
           [skills]

           Relevant Experience:
           [experience/projects]

           Why they are a good fit:
           [explanation]

           Skill Gaps:
           [missing skills, if any]


        2. [Candidate Name]

           Match Score: [Score]%

           Matching Skills:
           [skills]

           Relevant Experience:
           [experience/projects]

           Why they are a good fit:
           [explanation]

           Skill Gaps:
           [missing skills, if any]


        3. [Candidate Name]

           Match Score: [Score]%

           Matching Skills:
           [skills]

           Relevant Experience:
           [experience/projects]

           Why they are a good fit:
           [explanation]

           Skill Gaps:
           [missing skills, if any]


        Important:

        - Only use information available in the resumes.
        - Do not hallucinate qualifications.
        - Compare candidates specifically against the job description.
        - Clearly explain why the first candidate is ranked higher.
        - Do not create information that is not present in the resumes.
        """
    )
    rag_chain=final_prompt|llm
    job_matching=rag_chain.invoke(
        {
            "context":top_context,
            "query":job_description
        }
    )
    st.subheader("Top Candidates")
    st.write(top_context)
    st.subheader("Candidate Suitability Explanation")
    st.write(
        job_matching.content
    )