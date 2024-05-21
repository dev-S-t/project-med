# %%
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
# %%
google_api_key = os.getenv('GOOGLE_API_KEY')
pinecone_key = os.getenv("PINECONE_API_KEY")

# %%
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key= google_api_key)

# %%
data_dir = r"data\random.txt.txt"

def get_csv_text(path):
    df = pd.read_csv(path)
    text=""
    for index, row in df.iterrows():
        text += row['Answer'] + " "
    return  text

txt = get_csv_text(r"data\random.csv")

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

chunks = get_text_chunks(txt)

# %%
index_name = "project-med"
vectorstore = PineconeVectorStore.from_texts(chunks , index_name= index_name, embedding= embeddings)

# %%
retriever = vectorstore.as_retriever()

