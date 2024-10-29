from flask import Flask,jsonify,request
from utils import getCategoryOfInput,getResponseFromLLM, formatParagraphType,formatFlowchartType
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import re
from dotenv import load_dotenv

app = Flask(__name__)
model = "gpt-4o"

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

DB_FAISS_PATH = 'vectorstores/db_faiss'
embeddings = OpenAIEmbeddings(api_key=key)

def load_documents_from_txt(directory: str):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            loader = TextLoader(file_path)
            file_docs = loader.load()
            
            for doc in file_docs:
                doc.metadata = {"source": filename}
            
            docs.extend(file_docs)
    return docs

def setup_vector_store():
    if not os.path.exists(DB_FAISS_PATH):
        directory = 'sparkle_schemes2/'  # Ensure directory exists and contains .txt files
        documents = load_documents_from_txt(directory)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_documents = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            split_documents.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks])
        vectorstore = FAISS.from_documents(split_documents, embeddings)
        vectorstore.save_local(DB_FAISS_PATH)
    else:
        print("FAISS vector store already exists.")

def load_vector_store():
    if os.path.exists(DB_FAISS_PATH):
        print("Loading pre-built FAISS vector store...")
        return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        raise FileNotFoundError(f"FAISS vector store not found at {DB_FAISS_PATH}. Please upload the vector store.")


#setup_vector_store()
#vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vector_store()


#vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def retrieveDocuments(input_text):
    # Retrieve documents from FAISS based on the input
    retrieved_documents = vectorstore.as_retriever().get_relevant_documents(input_text)
    return retrieved_documents

@app.route('/',methods=["POST"])
def index():
    if request.method == "POST": 
        ip = request.form.get("body")
        
        # Determine category of the input
        #cat = getCategoryOfInput(model, ip)
        cat = getCategoryOfInput(ip)

        
        # Retrieve relevant documents for context
        retrieved_documents = retrieveDocuments(ip)
        context = "\n\n".join([doc.page_content for doc in retrieved_documents])
        
        # Generate the response with the retrieved context
        #content = getResponseFromLLM(model, ip, cat, context=context)
        content = getResponseFromLLM(model, ip, cat, context=context)
        
        if cat == "Informative Paragraph Question":
            headings, slugs = formatParagraphType(content)
            body = {
                "headings": headings,
                "slugs": slugs
            }
        elif cat == "Procedure-Based Question":
            body = formatFlowchartType(content)
        else:
            [val, cont] = content.split("\n\n", 1)
            body = {
                "value": val,
                "content": cont
            }

        data = {
            "type": cat,
            "body": body
        }
        return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
