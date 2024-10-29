from flask import Flask, jsonify, request
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import spacy
import numpy as np
import re
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
key = os.getenv("OPENAI_API_KEY") 
DB_FAISS_PATH = 'vectorstores/db_faiss'
STATE_BIAS = 10000

# Load Embeddings and Vectorstore
embeddings = OpenAIEmbeddings(api_key=key)
vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Load NER model for state detection
nlp = spacy.load("en_core_web_sm")
state_names = ["Andhra Pradesh", "Tamil Nadu", "Karnataka", "Kerala"]  # Add more as needed
state_set = set(state_names)

# Initialize OpenAI LLM
model_name = "gpt-4o"
llm = ChatOpenAI(temperature=0, model=model_name, openai_api_key=key)

# Prompts for categorization
prompt_categ = ChatPromptTemplate.from_template(
    """Task Overview:
    Your objective is to categorize any presented question into one of the following distinct types, based on the nature of the response it seeks:

    Procedure-Based Questions:
    Definition: These questions require a detailed, step-by-step guide or process as an answer. They are focused on how to accomplish a specific task or achieve a particular outcome.
    
    Yes/No Questions:
    Definition: These questions can be directly answered with a "Yes" or "No," potentially followed by a succinct explanation.

    Informative Paragraph Questions:
    Definition: These questions demand an answer in the form of a comprehensive, informative paragraph.

    Now which category does this {input} belong to?
    The answer should exactly with no other text be one of the following:
    Procedure-Based Question
    Yes/No Question
    Informative Paragraph Question
    """
)

# Define response prompts
prompt_what = ChatPromptTemplate.from_template(
        """Your job is to answer questions that need a bit more detail but keep your answers easy to understand. Follow these guidelines to help you:

        1. *Use Simple Language:* Explain things using basic words and short sentences. Avoid big or complicated words.
        2. *Stick to the Facts:* Give answers based on real information. Don’t guess or make things up. Make sure what you say is true for India.
        3. *Answer in Points:* Break down your answer into clear, numbered points. This makes it easier to read and understand.
        4. *Keep Context in Mind:* Remember, your answers should make sense to someone living in India.

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

# Define the retrieval function
def detect_state(query):
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text in state_set:
            return ent.text
    return None

def retrieve_documents(input_text):
    return vectorstore.as_retriever().get_relevant_documents(input_text)

def categorize_input(input_text):
    chain = LLMChain(llm=llm, prompt=prompt_categ)
    category = chain.run(input=input_text).strip()
    return category

def get_response(model_name: str, input_text: str, category: str, context: str):
    if category == "Procedure-Based Question":
        prompt = prompt_how
    elif category == "Yes/No Question":
        prompt = prompt_is
    else:
        prompt = prompt_what
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(user_input_eng=input_text, context=context)
    return response


@app.route('/', methods=["POST"])
def index():
    input_text = request.form.get("body")
    
    # Retrieve documents
    context_documents = retrieve_documents(input_text)
    context = "\n\n".join([doc.page_content for doc in context_documents])
    
    # Categorize the input
    category = categorize_input(input_text)
    
    # Generate response
    response = get_response(model_name, input_text, category, context)
    
    # Format response
    if category == "Procedure-Based Question":
        formatted_response = format_flowchart(response)
    elif category == "Informative Paragraph Question":
        formatted_response = format_paragraph(response)
    else:
        formatted_response = {"response": response} # Direct response as per the old format
    
    return jsonify(formatted_response)

# Helper functions for formatting
def format_flowchart(response):
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

def format_paragraph(response):
    paragraphs = response.split("\n\n")
    headings = ["Introduction"] + [f"Section {i+1}" for i in range(1, len(paragraphs)-1)] + ["Conclusion"]
    bodies = paragraphs
    return {"headings": headings, "bodies": bodies}

def main():
    with app.test_client() as client:
        user_input = "Do people with disabilities get financial aid? I live in Kerala."
        
        # Simulate a POST request to the '/' endpoint
        response = client.post('/', data={'body': user_input})
        
        # Print the response data
        print("Response JSON:", response.get_json())

# Run the main function to test
if __name__ == "__main__":
    main()
