from langchain_community.document_loaders import TextLoader

data = TextLoader("./documentLoaders/notes.txt")

docs = data.load()

print(len(docs))

# print(docs[0].page_content)