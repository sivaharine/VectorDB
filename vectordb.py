from sentence_transformers import SentenceTransformer
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings  # base class

# ✅ Wrap the model
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()  # Chroma expects a list of lists

    def embed_query(self, text):
        return self.model.encode([text]).tolist()[0]

# Initialize
embeddings = SentenceTransformerEmbeddings("hkunlp/instructor-base")

# Create Chroma DB
db = Chroma(persist_directory="db", embedding_function=embeddings)

# Add texts
db.add_texts(["AI helps doctors detect diseases", "Cars use sensors for automation"])

# Search
results = db.similarity_search("Cars and sensors")
print(results[0].page_content)
# C:\Users\HARINE\OneDrive\Documents\shellKode3\vectordb.