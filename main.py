from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

# from langchain_community.document_loaders import TextLoader

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters  import RecursiveCharacterTextSplitter


from langchain_core.prompts import ChatPromptTemplate

load_dotenv()



# data = TextLoader("./documentLoaders/notes.txt")
data = PyPDFLoader("./documentLoaders/DBMS_Notes.pdf")
# data = WebBaseLoader("https://www.radhavallabh.com/")
docs = data.load()

spiltter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunk = spiltter.split_documents(docs)



template = ChatPromptTemplate.from_messages([
    ("system" , "you are a AI that summerize the text"),
    ("human" , "{data}")
])

prompt = template.format_messages( data = (docs[0].page_content))

model = ChatMistralAI(model = "ministral-8b-2512")

response = model.invoke(prompt)

print(response.content)