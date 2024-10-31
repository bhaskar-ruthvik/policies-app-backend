import os
from dotenv import load_dotenv
import re
import spacy
import numpy as np
from flask import Flask, jsonify, request
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables and Flask app
load_dotenv()
app = Flask(__name__)

# Configurations
DB_FAISS_PATH = 'vectorstores/db_faiss'
SCHEMES_DIR = 'sparkle_schemes2'
state_set = set([
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh",
    "Assam", "Bihar", "Chandigarh", "Chhattisgarh", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir",
    "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan", "Sikkim",
    "Tamil Nadu", "Telangana", "The Dadra And Nagar Haveli And Daman And Diu",
    "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
])
STATE_BIAS = 10000
key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI and FAISS components
embeddings = OpenAIEmbeddings(api_key=key)
vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
nlp = spacy.load("en_core_web_sm")

# Function for detecting state in the query
def detect_state(query):
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text in state_set:
            return ent.text
    return None

# Function for hybrid retrieval combining dense, sparse, and state-based ANN
def hybrid_retrieve(query, k=3):
    dense_query_vec = np.array(embeddings.embed_query(query), dtype='float32').reshape(1, -1)
    dense_distances, dense_indices = vectorstore.search(dense_query_vec, k)
    
    detected_state = detect_state(query)
    hybrid_results = []
    for idx, score in zip(dense_indices[0], dense_distances[0]):
        metadata = vectorstore.metadata[idx]
        if metadata.get("state") == detected_state:
            score -= STATE_BIAS  # Apply state-based bias
        hybrid_results.append((metadata, score))
    
    return sorted(hybrid_results, key=lambda x: x[1])[:k]

# Function to retrieve context from hybrid search results
def get_context_from_hybrid(query):
    hybrid_results = hybrid_retrieve(query)
    context = "\n\n".join([result[0]["content"] for result in hybrid_results])
    return context

# Function to categorize input query
def getCategoryOfInput(input_text):
    llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=key)
    prompt = ChatPromptTemplate.from_template("""
    Task Overview:
    Your objective is to categorize any presented question into one of the following distinct types, based on the nature of the response it seeks:

    Procedure-Based Questions:
    Definition: These questions require a detailed, step-by-step guide or process as an answer. They are focused on how to accomplish a specific task or achieve a particular outcome.

    Yes/No Questions:
    Definition: These questions can be directly answered with a "Yes" or "No," potentially followed by a succinct explanation. They typically inquire about the possibility, availability, or existence of something.

    Informative Paragraph Questions:
    Definition: These questions demand an answer in the form of a comprehensive, informative paragraph. They generally request explanations, definitions, or detailed information about a specific subject.

    Now which category does this {input} belong to?
    The answer should exactly with no other text be one of the following:
    Procedure-Based Question
    Yes/No Question
    Informative Paragraph Question
    """)
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(input=input_text).strip()

# Function to generate response based on category
def getResponseFromLLM(input_text):
    category = categorize_input(input_text)
    context = get_context_from_hybrid(input_text)
    
    if category == "Procedure-Based Question":
        response = run_chain("prompt_how", input_text, context)
    elif category == "Yes/No Question":
        response = run_chain("prompt_is", input_text, context)
    else:
        response = run_chain("prompt_what", input_text, context)
    
    return response

# Function to set up prompt chains for each category type
def run_chain(prompt_name, input_text, context):
    llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=key)
    
    if prompt_name == "prompt_how":
        prompt = ChatPromptTemplate.from_template("""
        When you receive a question that needs a step-by-step answer, your task is to break it down into simpler, straightforward Yes/No questions. These questions should guide someone with little to no background knowledge through understanding and action. Here’s how you can do it effectively:

        Create Simple Yes/No Questions: Turn the main question into smaller questions that can be answered with a 'Yes' or a 'No'. Each question should be easy to understand, using basic language.

        Provide Clear Outcomes: For every possible path through the Yes/No questions (every combination of 'Yes' and 'No' answers), give a clear, final outcome. This outcome should be straightforward and offer guidance or information in response to the original question. 

        Keep it Relevant to India: Make sure your questions and outcomes are suitable and accurate for someone in India. Use examples, language, and context that make sense locally.

        Be Elaborate and Accurate: Even though the language should be simple, ensure your answers cover all necessary details and are correct. Aim to leave no room for confusion or misinterpretation.

        When formatting how-to type questions, format them such that there are no "*"s used and the response is given as 1. question, next line "- Yes:" what to do, next line "- No:" what to do.

        Ensure Yes and No is given for each step.

        This is the context, based on this information, answer the question: {context}
        Here is the question: {user_input_eng}
        """)

    elif prompt_name == "prompt_is":
        prompt = ChatPromptTemplate.from_template("""
        Your main task is to give a clear 'Yes' or 'No' answer to the question asked. After you answer, add a short paragraph explaining your answer in a simple way. Here’s how to do it:

        1. Start with a Clear Answer: Begin by saying 'Yes' or 'No'. This makes sure the person asking knows the answer right away.

        2. Explain in Simple Words: After your clear 'Yes' or 'No', explain why this is the answer. Use easy words and short sentences that anyone can understand.

        3. Keep it Relevant to India: Make sure your explanation is accurate for someone in India. Use examples or reasons that make sense in the Indian context.

        4. Be Positive and Helpful: Even if the answer is 'No', try to keep your explanation positive and helpful. If possible, offer a brief suggestion or an alternative idea.

        This is the context, based on this information, answer the question: {context}
        Here is the question: {user_input_eng}
        """)

    else:
        prompt = ChatPromptTemplate.from_template("""
        Your job is to answer questions that need a bit more detail but keep your answers easy to understand. Follow these guidelines to help you:

        1. Use Simple Language: Explain things using basic words and short sentences. Avoid big or complicated words.
        
        2. Stick to the Facts: Give answers based on real information. Don't guess or make things up. Make sure what you say is true for India.
        
        3. Answer in Points: Break down your answer into clear, numbered points. This makes it easier to read and understand.
        
        4. Keep Context in Mind: Remember, your answers should make sense to someone living in India.

        This is the context, based on this information, answer the question: {context}
        Here is the question: {user_input_eng}
        """)

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(user_input_eng=input_text, context=context)

def formatFlowchartType(response: str):
    steps = response.strip().split("\n\n")

    questionMatcher = re.compile(r"^\d+\.\s*(.+)") 
    yesMatcher = re.compile(r"-\s*Yes:\s*(.+?)(?=\s*-\s*No:|\Z)", re.DOTALL)  
    noMatcher = re.compile(r"-\s*No:\s*(.+?)(?=\n|$)", re.DOTALL) 
    flowchart = []
    
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



def formatParagraphType(response: str):
    headingRegex = re.compile(r'\*\*.*\*\*')
    paragraphs = response.split("\n\n")
    headings = []
    bodies = []
    
    for i, para in enumerate(paragraphs):
        if i == 0:
            headings.append("Introduction")
            bodies.append(para)
        elif i == len(paragraphs) - 1:
            headings.append("Conclusion")
            bodies.append(para)
        else:
            headings.append(headingRegex.search(para).group(0).split("**")[1])
            pts = [x.split("\n")[0] for x in para.split("  - ")[1:]]
            temp = " ".join(pts)
            bodies.append(temp)
    
    return headings, bodies

# Flask endpoint to get the response from the LLM
# @app.route('/get_answer', methods=['POST'])
# def get_answer():
#     input_data = request.json.get("question")
#     response = get_response_from_llm(input_data)
#     return jsonify({"response": response})

