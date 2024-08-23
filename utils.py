import os
from dotenv import load_dotenv
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv()

key = os.environ.get('OPENAI_API_KEY')

def getResponseFromLLM(model_name: str, input: str, category: str):
    llm = ChatOpenAI(temperature=0, model=model_name, openai_api_key=key)
    
    prompt_what = ChatPromptTemplate.from_template(
    """Your job is to answer questions that need a bit more detail but keep your answers easy to understand. Follow these guidelines to help you:

    1. *Use Simple Language:* Explain things using basic words and short sentences. Avoid big or complicated words.

    2. *Stick to the Facts:* Give answers based on real information. Don’t guess or make things up. Make sure what you say is true for India.

    3. *Answer in Points:* Break down your answer into clear, numbered points. This makes it easier to read and understand.

    4. *Keep Context in Mind:* Remember, your answers should make sense to someone living in India. Use examples or explanations that fit with what's common or known in India.

    When answering a question that asks for a detailed explanation on a topic (like explaining a concept or providing a list of needed items or steps), use these rules to create your response. Aim to be helpful and clear without using complicated language or ideas.
    Here is the question : {user_input_eng}""")
    
    prompt_is = ChatPromptTemplate.from_template(
    """Your main task is to give a clear 'Yes' or 'No' answer to the question asked. After you answer, add a short paragraph explaining your answer in a simple way. Here’s how to do it:

    1. *Start with a Clear Answer:* Begin by saying 'Yes' or 'No'. This makes sure the person asking knows the answer right away.

    2. *Explain in Simple Words:* After your clear 'Yes' or 'No', explain why this is the answer. Use easy words and short sentences that anyone can understand.

    3. *Keep it Relevant to India:* Make sure your explanation is accurate for someone in India. Use examples or reasons that make sense in the Indian context.

    4. *Be Positive and Helpful:* Even if the answer is 'No', try to keep your explanation positive and helpful. If possible, offer a brief suggestion or an alternative idea.

    Your goal is to provide straightforward, helpful answers that anyone can understand, especially focusing on topics relevant to India. This approach helps make sure your response is both useful and easy to read for people with different levels of reading skills.
    Here is the question : {user_input_eng}""")
    
    prompt_how = ChatPromptTemplate.from_template(
    """When you receive a question that needs a step-by-step answer, your task is to break it down into simpler, straightforward Yes/No questions. These questions should guide someone with little to no background knowledge through understanding and action. Here’s how you can do it effectively:

    Create Simple Yes/No Questions: Turn the main question into smaller questions that can be answered with a 'Yes' or a 'No'. Each question should be easy to understand, using basic language.

    Design a Flowchart: Arrange these Yes/No questions in a flowchart format. This means that each 'Yes' or 'No' answer will lead to the next step or question. Ensure the flowchart is logical and guides the user towards a final answer or action.

    Provide Clear Outcomes: For every possible path through the Yes/No questions (every combination of 'Yes' and 'No' answers), give a clear, final outcome. This outcome should be straightforward and offer guidance or information in response to the original question.

    Keep it Relevant to India: Make sure your questions and outcomes are suitable and accurate for someone in India. Use examples, language, and context that make sense locally.

    Be Elaborate and Accurate: Even though the language should be simple, ensure your answers cover all necessary details and are correct. Aim to leave no room for confusion or misinterpretation.

    At the end of the flowchart, based on the paths taken through the Yes/No questions, provide a final answer or advice that directly addresses the user's original query. Remember, your goal is to make the process as clear and helpful as possible, even for someone who might not be familiar with the topic.
    Here is the question: {user_input_eng}"""
    )
    
    if category == "Procedure-Based Question":
        chain = LLMChain(llm=llm, prompt=prompt_how)
        response = chain.run(user_input_eng=input)
    elif category == "Yes/No Question":
        chain = LLMChain(llm=llm, prompt=prompt_is)
        response = chain.run(user_input_eng=input)
    elif category == "Informative Paragraph Question":
        chain = LLMChain(llm=llm, prompt=prompt_what)
        response = chain.run(user_input_eng=input)

    return response

def getCategoryOfInput(model_name: str, input: str):
    llm = ChatOpenAI(temperature=0, model=model_name, openai_api_key=key)
    
    prompt_categ = ChatPromptTemplate.from_template(
    """Task Overview:
    Your objective is to categorize any presented question into one of the following distinct types, based on the nature of the response it seeks:

    Procedure-Based Questions:
    Definition: These questions require a detailed, step-by-step guide or process as an answer. They are focused on how to accomplish a specific task or achieve a particular outcome.
    Key Indicators: Look for questions that ask "How to," "What are the steps," or "What should I do to."
    Examples:
    "How can a woman with no financial support find a safe place to live?"
    "I want to buy a house. How much money can I withdraw from my PF?"
    "How do I get support if I lost my job and need money?"
    Yes/No Questions:
    Definition: These questions can be directly answered with a "Yes" or "No," potentially followed by a succinct explanation. They typically inquire about the possibility, availability, or existence of something.
    Key Indicators: Questions that can be directly responded to with a binary answer.
    Examples:
    "Can women who left home due to violence get shelter?"
    "Is there support for women who have been trafficked?"
    "Are children allowed in shelters for women in difficulty?"
    Informative Paragraph Questions:
    Definition: These questions demand an answer in the form of a comprehensive, informative paragraph. They generally request explanations, definitions, or detailed information about a specific subject.
    Key Indicators: Questions that ask "What," "Which," or "Who" that seek detailed information or explanation, but not in a step-by-step format.
    Examples:
    "What aid is available for women affected by HIV without support?"
    "Which programs offer help for women in India above 58?"
    "What aid is there for students from poor families for higher education?"

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

def formatFlowchartType(response: str):
    steps = response.split("\n\n")
    questionMatcher = re.compile(r"\*\*.*\*\*")
    answerMatcher = re.compile(r"-\s+Yes:.+[^\n]")
    noMatcher = re.compile(r"-\s+No:.+[^\n]")
    
    questions = []
    yes_actions = []
    no_actions = []
    q_json = []
    
    for step in steps:
        if questionMatcher.search(step) != None:
            questions.append(questionMatcher.search(step).group().split("**")[1])
        else:
            questions.append(None)

        if answerMatcher.search(step) != None:
            yes_action = answerMatcher.search(step).group().split(": ")[1]
            if "question " in yes_action:
                yes_actions.append(int(yes_action.split("question ")[1].split(" ")[0].split(".")[0]))
            else:
                yes_actions.append(yes_action)
        else:
            yes_actions.append(None)

        if noMatcher.search(step) != None:
            no_action = noMatcher.search(step).group().split(": ")[1]
            if "question " in no_action:
                no_actions.append(int(no_action.split("question ")[1].split(" ")[0].split(".")[0]))
            else:
                no


# test_inputs = [
#     "How can a woman with no financial support find a safe place to live?",
#     "Can women who left home due to violence get shelter?",
#     "What aid is available for women affected by HIV without support?",
#     "How do I get support if I lost my job and need money?",
#     "Are children allowed in shelters for women in difficulty?"
# ]

# # Expected categories
# expected_categories = [
#     "Procedure-Based Question",
#     "Yes/No Question",
#     "Informative Paragraph Question",
#     "Procedure-Based Question",
#     "Yes/No Question"
# ]

model_name= "gpt-4o"

# print("Testing getCategoryOfInput function...")
# for i, input_text in enumerate(test_inputs):
#     category = getCategoryOfInput(model_name, input_text)
#     print(f"Input: {input_text}")
#     print(f"Expected Category: {expected_categories[i]}")
#     print(f"Predicted Category: {category}")
#     print(f"Test Passed: {category == expected_categories[i]}")
#     print()

# Test getResponseFromLLM function
# print("Testing getResponseFromLLM function...")
# for i, input_text in enumerate(test_inputs):
#     category = expected_categories[i]
#     response = getResponseFromLLM(model_name, input_text, category)
#     print(f"Input: {input_text}")
#     print(f"Category: {category}")
#     print(f"Response: {response}")
#     print()