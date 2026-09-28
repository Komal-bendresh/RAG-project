from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("./documentLoaders/OOPGEN.pdf")

docs = data.load()

print(docs[10].page_content)