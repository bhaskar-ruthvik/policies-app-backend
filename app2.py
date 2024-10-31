import os
from dotenv import load_dotenv
import re
from flask import Flask, jsonify, request
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings  
from langchain.vectorstores import FAISS 
from langchain.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.schema import Document 
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy

load_dotenv()
key = os.getenv("OPENAI_API_KEY") 

DB_FAISS_PATH = 'vectorstores/db_faiss'
embeddings = OpenAIEmbeddings(api_key=key)
vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

SCHEMES_DIR = 'sparkle_schemes'
DB_FAISS_PATH = 'vectorstores/db_faiss'
openai_api_key = os.getenv("OPENAI_API_KEY") 

def load_schemes():
    scheme_texts = []
    scheme_metadata = []
    
    for filename in os.listdir(SCHEMES_DIR):
        if filename.endswith('.txt'):
            file_path = os.path.join(SCHEMES_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                scheme_texts.append(content)
                metadata = {
                    "filename": filename,
                    "content": content,
                    "state": "All"  
                }
                scheme_metadata.append(metadata)
    
    return scheme_texts, scheme_metadata

scheme_texts, scheme_metadata = load_schemes()

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

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(scheme_texts)


nlp = spacy.load("en_core_web_sm")
def detect_state(query):
    doc = nlp(query)
    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text in state_set:
            return ent.text
    return None

def hybrid_retrieve(query, k=1):
    sparse_query_vec = tfidf_vectorizer.transform([query])
    sparse_scores = np.dot(tfidf_matrix, sparse_query_vec.T).toarray().ravel()
    sparse_top_indices = np.argsort(sparse_scores)[-k:][::-1]
    
    dense_query_vec = np.array(embeddings.embed_query(query), dtype='float32').reshape(1, -1)
    dense_results = vectorstore.similarity_search_with_score_by_vector(dense_query_vec[0], k=k)

    detected_state = detect_state(query)
    hybrid_results = []

    for doc, score in dense_results:
        scheme = doc.metadata
        if scheme.get("state") == detected_state:
            score -= STATE_BIAS  
        hybrid_results.append((scheme, score))
    
    sparse_results = [(scheme_metadata[i], sparse_scores[i]) for i in sparse_top_indices]
    hybrid_results.extend(sparse_results)
    
    return sorted(hybrid_results, key=lambda x: x[1])[:k]


def get_context_from_hybrid(query: str):
    print("Retrieving results for query:", query)
    hybrid_results = hybrid_retrieve(query)
    
    context = "\n\n".join([result[0].get("content", result[0].get("page_content", "")) for result in hybrid_results])
    print(f"Retrieved {len(hybrid_results)} relevant documents")
          
    return context



def getResponseFromLLM(model_name: str, input: str, category: str, context: str):
    llm = ChatOpenAI(temperature=0, model=model_name, openai_api_key=key)
    
    prompt_what = ChatPromptTemplate.from_template(
    """Your job is to answer questions that need a bit more detail but keep your answers easy to understand. Follow these guidelines to help you:

    1. Use Simple Language: Explain things using basic words and short sentences. Avoid big or complicated words.

    2. Stick to the Facts: Give answers based on real information. Don’t guess or make things up. Make sure what you say is true for India.

    3. Answer in Points: Break down your answer into clear, numbered points. This makes it easier to read and understand.

    4. Keep Context in Mind: Remember, your answers should make sense to someone living in India. Use examples or explanations that fit with what's common or known in India.

    When answering a question that asks for a detailed explanation on a topic (like explaining a concept or providing a list of needed items or steps), use these rules to create your response. Aim to be helpful and clear without using complicated language or ideas.

    If the question is asked in a language other than english, answer in that language. 

    This is the context, based on this information, answer the question: {context}
    Here is the question : {user_input_eng}""")
    
    prompt_is = ChatPromptTemplate.from_template(
    """Your main task is to give a clear 'Yes' or 'No' answer to the question asked. After you answer, add a short paragraph explaining your answer in a simple way. Here’s how to do it:

    1. Start with a Clear Answer: Begin by saying 'Yes' or 'No'. This makes sure the person asking knows the answer right away.

    2. Explain in Simple Words: After your clear 'Yes' or 'No', explain why this is the answer. Use easy words and short sentences that anyone can understand.

    3. Keep it Relevant to India: Make sure your explanation is accurate for someone in India. Use examples or reasons that make sense in the Indian context.

    4. Be Positive and Helpful: Even if the answer is 'No', try to keep your explanation positive and helpful. If possible, offer a brief suggestion or an alternative idea.

    Your goal is to provide straightforward, helpful answers that anyone can understand, especially focusing on topics relevant to India. This approach helps make sure your response is both useful and easy to read for people with different levels of reading skills.

    If the question is asked in a language other than english, answer in that language. 

    This is the context, based on this information, answer the question: {context}
    Here is the question : {user_input_eng}""")
    
    prompt_how = ChatPromptTemplate.from_template(
    """When you receive a question that needs a step-by-step answer, your task is to break it down into simpler, straightforward Yes/No questions. These questions should guide someone with little to no background knowledge through understanding and action. Here’s how you can do it effectively:

    Create Simple Yes/No Questions: Turn the main question into smaller questions that can be answered with a 'Yes' or a 'No'. Each question should be easy to understand, using basic language.

    Provide Clear Outcomes: For every possible path through the Yes/No questions (every combination of 'Yes' and 'No' answers), give a clear, final outcome. This outcome should be straightforward and offer guidance or information in response to the original question. Don't make a seperate section for this. Incorporate it in the Yes/No Questions.

    Keep it Relevant to India: Make sure your questions and outcomes are suitable and accurate for someone in India. Use examples, language, and context that make sense locally.

    Be Elaborate and Accurate: Even though the language should be simple, ensure your answers cover all necessary details and are correct. Aim to leave no room for confusion or misinterpretation.

    When formatting how type questions, format it such that there are no "*"s used and the response is given as 1. question, next line "- Yes:" what to do, next line "- No:" what to do

    Ensure Yes and No is given for each step.

    If the question is asked in a language other than english, answer in that language. 

    This is the context, based on this information, answer the question: {context}
    Here is the question: {user_input_eng}"""
    )
    
    if category == "Procedure-Based Question":
        chain = LLMChain(llm=llm, prompt=prompt_how)
        response = chain.run(user_input_eng=input, context=context)
    elif category == "Yes/No Question":
        chain = LLMChain(llm=llm, prompt=prompt_is)
        response = chain.run(user_input_eng=input, context=context)
    elif category == "Informative Paragraph Question":
        chain = LLMChain(llm=llm, prompt=prompt_what)
        response = chain.run(user_input_eng=input, context=context)

    return response


def getCategoryOfInput(model_name: str, input: str):
    llm = ChatOpenAI(temperature=0, model=model_name, openai_api_key=key)
    
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
    
    chain = LLMChain(llm=llm, prompt=prompt_categ)
    category = chain.run(input=input)

    return category

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
    headingRegex = re.compile(r'\\.\\*')
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


model_name= "gpt-4o"