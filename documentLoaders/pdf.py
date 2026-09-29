from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("./documentLoaders/OOPGEN.pdf")
docs = data.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap= 10)

texts = text_splitter.split_documents(docs)

print(len(texts))

# print(texts[0].page_content)