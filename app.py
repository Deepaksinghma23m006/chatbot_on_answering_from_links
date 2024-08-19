import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS

# Streamlit app

# Streamlit Title
st.title("Google Generative AI with LangChain")

# Load API key
os.environ['GOOGLE_API_KEY'] = st.text_input("your_api_key", type="password")

# Configure the API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize LLM with required params
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Load data
st.subheader("Load Data")
url = st.text_input("Enter URL to load data", "https://timesofindia.indiatimes.com/sports/cricket/icc-mens-t20-world-cup/team-india-becomes-t20-world-cup-2024-champions-virat-kohli-bids-adieu-to-t20is/articleshow/111369898.cms")
if st.button("Load Data"):
    loader = UnstructuredURLLoader(urls=[url])
    data = loader.load()
    st.write(f"Loaded {len(data)} documents.")

    # Split data to create chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    st.write(f"Created {len(docs)} document chunks.")

    # Create embeddings for these chunks and save them to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = [doc.page_content for doc in docs]
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local("faiss_index")
    st.success("FAISS index created and saved locally.")

# QA Chain setup
st.subheader("Question Answering")
question = st.text_input("Enter your question", "What was the result of ICC Men's T20 World Cup 2024?")

if st.button("Get Answer"):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_with_sources_chain(model, chain_type="stuff", prompt=prompt)

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)

    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    st.write(response["output_text"])

