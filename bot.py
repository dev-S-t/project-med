import os 
import retrieval
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,}

ret = retrieval.retriever


llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro-latest', google_api_key=google_api_key, safety_settings=safety_settings)

template = """
You are a medical chatbot.reply greetings with greetings,
understand the question and respond accordingly by suggesting lifestyle changes , food recommendations and exercises which could solve users problem 
Context: {context}
Question: {question}

1. Lifestyle Changes:
2. Food Recommendations:
3. Exercise Suggestions:
4. Potential Medicines:

Helpful Answer:
"""
prompt_template = PromptTemplate.from_template(template=template)

set_ret = RunnableParallel(
    {"context": ret, "question": RunnablePassthrough()} 
)

rag_chain = set_ret |  prompt_template | llm | StrOutputParser()



