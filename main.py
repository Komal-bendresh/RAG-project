# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI

# # from langchain_community.document_loaders import TextLoader

# from langchain_community.document_loaders import PyPDFLoader

# from langchain_community.document_loaders import WebBaseLoader

# from langchain_text_splitters  import RecursiveCharacterTextSplitter


# from langchain_core.prompts import ChatPromptTemplate

# load_dotenv()



# # data = TextLoader("./documentLoaders/notes.txt")
# data = PyPDFLoader("./documentLoaders/DBMS_Notes.pdf")
# # data = WebBaseLoader("https://www.radhavallabh.com/")
# docs = data.load()

# spiltter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 200
# )

# chunk = spiltter.split_documents(docs)



# template = ChatPromptTemplate.from_messages([
#     ("system" , "you are a AI that summerize the text"),
#     ("human" , "{data}")
# ])

# prompt = template.format_messages( data = (docs[0].page_content))

# model = ChatMistralAI(model = "ministral-8b-2512")

# response = model.invoke(prompt)

# print(response.content)

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings = HuggingFaceEmbeddings()


vectorstore = Chroma(
    persist_directory= "chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 4,
        "fetch_k":10,
        "lambda_mult" :0.5
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")

#prompt template 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
    