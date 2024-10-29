from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableBranch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain
import re
import faiss
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import firebase_admin
from firebase_admin import credentials, storage
from scipy.spatial.distance import cdist
import flask
from flask import Flask, jsonify, request
app = Flask(_name_)


# Initialize Firebase Admin SDK with the service account credentials
cred = credentials.Certificate("actionable-welfare-schemes-firebase-adminsdk-2j0yf-644b151192.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'actionable-welfare-schemes.appspot.com'  # Replace with your actual Firebase bucket name
})

# Define the bucket variable
bucket = storage.bucket()

# Load environment variables and configurations
SCHEMES_DIR = 'sparkle_schemes'
DB_FAISS_PATH = 'vectorstores/db_faiss'
key = os.getenv("OPENAI_API_KEY")

# Load schemes from directory
def load_schemes():
    scheme_texts = []
    scheme_metadata = []
    
    for filename in os.listdir(SCHEMES_DIR):
        if filename.endswith('.txt'):
            file_path = os.path.join(SCHEMES_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                scheme_texts.append(content)
                # Create metadata for each scheme
                metadata = {
                    "filename": filename,
                    "content": content,
                    "state": "All"  # Default state, adjust if state info is available in content
                }
                scheme_metadata.append(metadata)
    
    return scheme_texts, scheme_metadata

# Load schemes
scheme_texts, scheme_metadata = load_schemes()

# Initialize OpenAI components
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
model = ChatOpenAI(temperature=0, model="gpt-4o", api_key=openai_api_key)

# Load FAISS vector store
def load_faiss_from_firebase():
    local_index_path_faiss = "loading/index.faiss"  # local path for faiss index
    local_index_path_pkl = "loading/index.pkl"  # local path for pkl index
    
    # Download index.faiss
    blob_faiss = bucket.blob("vectorstores/db_faiss/index.faiss")
    if blob_faiss.exists():
        blob_faiss.download_to_filename(local_index_path_faiss)
        print(f"Downloaded 'index.faiss' to {local_index_path_faiss}")
    else:
        raise ValueError(f"The blob 'vectorstores/db_faiss/index.faiss' does not exist in Firebase Storage.")
    
    # Download index.pkl
    blob_pkl = bucket.blob("vectorstores/db_faiss/index.pkl")
    if blob_pkl.exists():
        blob_pkl.download_to_filename(local_index_path_pkl)
        print(f"Downloaded 'index.pkl' to {local_index_path_pkl}")
    else:
        raise ValueError(f"The blob 'vectorstores/db_faiss/index.pkl' does not exist in Firebase Storage.")
    
    # Load FAISS index
    vectorstore = FAISS.load_local("loading", embeddings, allow_dangerous_deserialization=True)
    return vectorstore



# Replace the original line with the new function call
vectorstore = load_faiss_from_firebase()
# Load NER model for state detection
nlp = spacy.load("en_core_web_sm")
state_names = [
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh",
    "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir",
    "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "The Dadra And Nagar Haveli And Daman And Diu",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]
state_set = set(state_names)
STATE_BIAS = 10000

# Initialize TF-IDF Vectorizer with loaded schemes
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(scheme_texts)

# Initialize dense embeddings and FAISS index
# print("Creating dense embeddings...")
# dense_embeddings = np.array([embeddings.embed_query(text) for text in scheme_texts], dtype='float32')
# print(f"Created embeddings of shape: {dense_embeddings.shape}")

# index = faiss.IndexFlatL2(vectorstore.shape[1])
# index.add(vectorstore)

def detect_state(query):
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text in state_set:
            return ent.text
    return None

def hybrid_retrieve(query, k=1):
    # Sparse Retrieval
    sparse_query_vec = tfidf_vectorizer.transform([query])
    sparse_scores = np.dot(tfidf_matrix, sparse_query_vec.T).toarray().ravel()
    sparse_top_indices = np.argsort(sparse_scores)[-k:][::-1]
    
    # Dense Retrieval with similarity scores from vector store
    dense_query_vec = np.array(embeddings.embed_query(query), dtype='float32').reshape(1, -1)
    dense_results = vectorstore.similarity_search_with_score_by_vector(dense_query_vec[0], k=k)

    # State-Specific Bias and combine scores
    detected_state = detect_state(query)
    hybrid_results = []

    for doc, score in dense_results:
        scheme = doc.metadata
        if scheme.get("state") == detected_state:
            score -= STATE_BIAS  # Apply bias to score if state matches
        hybrid_results.append((scheme, score))
    
    # Combine sparse and dense results
    sparse_results = [(scheme_metadata[i], sparse_scores[i]) for i in sparse_top_indices]
    hybrid_results.extend(sparse_results)
    
    return sorted(hybrid_results, key=lambda x: x[1])[:k]

def get_context_from_hybrid(query: str):
    print("Retrieving results for query:", query)
    hybrid_results = hybrid_retrieve(query)
    
    # Check available keys for debugging
    for result in hybrid_results:
        print(f"Available keys in result[0]: {result[0].keys()}")

    # Try to use 'page_content' if 'content' is not available
    context = "\n\n".join([result[0].get("content", result[0].get("page_content", "")) for result in hybrid_results])
    print(f"Retrieved {len(hybrid_results)} relevant documents")
    return context

def formatFlowchartType(response):
    steps = response.strip().split("\n\n")
    flowchart = []

    questionMatcher = re.compile(r"^\d+\.\s*(.+)")
    yesMatcher = re.compile(r"-\s*Yes:\s*(.+?)(?=\s*-\s*No:|\Z)", re.DOTALL)
    noMatcher = re.compile(r"-\s*No:\s*(.+?)(?=\n|$)", re.DOTALL)
    
    for step in steps:
        question_match = questionMatcher.search(step)
        yes_action_match = yesMatcher.search(step)
        no_action_match = noMatcher.search(step)
        
        if question_match:
            question_text = question_match.group(1).strip()
            yes_text = yes_action_match.group(1).strip() if yes_action_match else None
            no_text = no_action_match.group(1).strip() if no_action_match else None
            flowchart.append({
                "question": question_text,
                "yes_action": yes_text,
                "no_action": no_text
            })

    return {"flowchart": flowchart}

def formatParagraphType(response):
    headingRegex = re.compile(r'\\.\\*')
    paragraphs = response.split("\n\n")
    headings = ["Introduction"] + [f"Section {i+1}" for i in range(1, len(paragraphs)-1)] + ["Conclusion"]
    bodies = paragraphs
    return {"headings": headings, "bodies": bodies}

def getCategoryOfInput(model_name, input_text):
    llm = ChatOpenAI(temperature=0, model=model_name, openai_api_key=openai_api_key)
    
    prompt_categ = ChatPromptTemplate.from_template(
        """Task Overview:
        Classify the question into one of the following:
        - Procedure-Based Question
        - Yes/No Question
        - Informative Paragraph Question
        
        Answer: {input}"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_categ)
    category = chain.run(input=input_text).strip()
    return category

# [Rest of the code remains the same - prompts and chain definitions]
prompt_what = ChatPromptTemplate.from_template(
        """Your job is to answer questions that need a bit more detail but keep your answers easy to understand. Follow these guidelines to help you:

        1. Use Simple Language: Explain things using basic words and short sentences. Avoid big or complicated words.
        2. Stick to the Facts: Give answers based on real information. Don’t guess or make things up. Make sure what you say is true for India.
        3. Answer in Points: Break down your answer into clear, numbered points. This makes it easier to read and understand.
        4. Keep Context in Mind: Remember, your answers should make sense to someone living in India.

        This is the context: {context}
        Here is the question : {user_input_eng}"""
    )
prompt_is = ChatPromptTemplate.from_template(
        """Your main task is to give a clear 'Yes' or 'No' answer to the question asked. After you answer, add a short paragraph explaining your answer in a simple way.

        This is the context: {context}
        Here is the question : {user_input_eng}"""
    )
    
prompt_how = ChatPromptTemplate.from_template(
        """Provide a step-by-step answer using Yes/No questions. Keep it straightforward and accurate for someone in India.

        This is the context: {context}
        Here is the question : {user_input_eng}"""
    )

prompt_categ = ChatPromptTemplate.from_template("""
Task Overview:
Categorize the question into one of these types:

Procedure-Based Questions:
- Require step-by-step guides
- Usually start with "How to," "What are the steps"

Yes/No Questions:
- Can be answered with "Yes" or "No" plus explanation
- Ask about possibility or availability

Informative Paragraph Questions:
- Need comprehensive explanations
- Ask for definitions or detailed information

Which category does this {user_input_eng} belong to?
Answer with exactly one of:
Procedure-Based Question
Yes/No Question
Informative Paragraph Question
""")

# Set up chain components
classification_chain = prompt_categ | model | StrOutputParser()
branches = RunnableBranch(
    (lambda x: "Procedure-Based Question" in x["category"], prompt_how | model | StrOutputParser()),
    (lambda x: "Informative Paragraph Question" in x["category"], prompt_what | model | StrOutputParser()),
    prompt_is | model | StrOutputParser()
)

chain = RunnableParallel(
    category=classification_chain,
    user_input_eng=lambda x: x["user_input_eng"],
    context=lambda x: x["context"]
) | branches

@app.route('/', methods=["POST"])
def index():
    if request.method == "POST": 
        input_text = request.form.get("body")
        
        # Retrieve documents
        context = get_context_from_hybrid(input_text)

        model_name = "gpt-4o"

        cat = getCategoryOfInput(model_name, input_text)
        
        result = chain.invoke(input={"context": context, "user_input_eng": input_text})

        if cat == "Informative Paragraph Question":
            headings, slugs = formatParagraphType(result)
            body = {
                "headings": headings,
                "slugs": slugs
            }
        elif cat == "Procedure-Based Question":
            body = formatFlowchartType(result)
        else:
            [val, cont] = result.split("\n\n", 1)
            body = {
                "value": val,
                "content": cont
            }

        data = {
            "type": cat,
            "body": body
        }
        return jsonify(data)


def main():
    with app.test_client() as client:
        print("Loading schemes...")
        # Schemes are already loaded at the start
        
        print("\nInitialization complete. Ready for queries.")
        
        # Example usage
        user_input = "do people with disabilities get financial aid? I live in Kerala."
        print("\nProcessing query:", user_input)
        
        response = client.post('/', data={'body': user_input})
        print(response.get_json())

if __name__ == "__main__":
    main()
