import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# ----------------------------
# Step 1: Load preprocessed CSV
# ----------------------------
df = pd.read_csv("data/preprocessed_sample.csv")

# ----------------------------
# Step 2: Select the text column for embeddings
# ----------------------------
texts = df["combined_text"].tolist()

# ----------------------------
# Step 3: Initialize embeddings model
# ----------------------------
# You can use any HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# Step 4: Create ChromaDB
# ----------------------------
chroma_db = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db")

# ----------------------------
# Step 5: Persist to disk
# ----------------------------
chroma_db.persist()
print("Embeddings stored in ChromaDB at ./chroma_db")