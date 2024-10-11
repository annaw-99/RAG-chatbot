# Import relevant libraries
import streamlit as st
from openai import AzureOpenAI
from os import environ

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
import tempfile

from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LLM
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    temperature=0.2,
    api_version="2023-06-01-preview",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)  

# Initialize client
client = AzureOpenAI(
    api_key=environ['AZURE_OPENAI_API_KEY'],
    api_version="2023-03-15-preview",
    azure_endpoint=environ['AZURE_OPENAI_ENDPOINT'],
    azure_deployment=environ['AZURE_OPENAI_MODEL_DEPLOYMENT'],
)  

# Define document text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 250,
    chunk_overlap = 50
)

# Format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Allow multiple files
st.title("📝 File Q&A with OpenAI")
uploaded_files = st.file_uploader("Upload your file(s)", type=("txt", "pdf"), accept_multiple_files=True)

# Allow text input
question = st.chat_input(
    "Ask something about the uploaded files",
    disabled=not uploaded_files,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask something about your upload file(s)."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question and uploaded_files:
    # Define document chain
    documents = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "text/plain":
            # Run if file is a .txt
            with tempfile.NamedTemporaryFile(delete=True, suffix=".txt") as temp_txt:
                temp_txt.write(uploaded_file.read())
                
                loader = TextLoader(temp_txt.name, encoding="utf-8")
                all_text = loader.load()
                
                for doc in all_text:
                    doc.metadata.update({"source": uploaded_file.name})
                    documents.append(doc)

        elif uploaded_file.type == "application/pdf":
            # Run if file is a .pdf
            with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                
                loader = PyPDFLoader(temp_pdf.name)
                all_pdf = loader.load()
                
                for doc in all_pdf:
                    doc.metadata.update({"source": uploaded_file.name})
                    documents.append(doc)
        else: 
            # Shows if file is not a .txt or .pdf file
            st.warning(f"Unsupported File Type")

    # Chunk the documents
    chunks = text_splitter.split_documents(documents)
    
    # Index chunks into vectordb
    vectorstore = Chroma.from_documents(documents=chunks, embedding=AzureOpenAIEmbeddings(model="text-embedding-3-large"))
    
    # Define prompt
    template = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.
        
        Question: {question} 
        
        Context: {context} 
        
        Answer:
        """
    prompt = PromptTemplate.from_template(template)
    
    # Set up retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    
    # Build RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # Invoke RAG chain to get an answer
    response = rag_chain.invoke(question)

    with st.chat_message("assistant"):
        st.write(response)
        # Append the assistant's response to the messages
        st.session_state.messages.append({"role": "assistant", "content": response})
