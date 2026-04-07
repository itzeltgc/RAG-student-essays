import os
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# function to retrieve both filename and text
def load_essays(folder_path):
    essays = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, filename))
            text = ""
            for page in reader.pages:
                text+= page.extract_text()
            essays.append({"filename": filename, "text": text})
    
    return essays

 
# function to split essay text into chunks
def split_into_chunks(essays):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    chunks = []
    for essay in essays:
        splits = splitter.split_text(essay["text"])
        for split in splits:
            chunks.append({"filename": essay["filename"], "text": split})
    return chunks


# function to store in chromeDB
def store_in_chromadb(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="data/chromadb")

    try:
        client.delete_collection("essays")
    except:
        pass
    collection = client.get_or_create_collection("essays")


    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk["text"]).tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"filename": chunk["filename"]}]

        )
    print(f"✅ {len(chunks)} chunks saved in chromaDB")


if __name__ == "__main__":
    print("📂 loading essays...")
    essays = load_essays("data/essays")
    print(f"✅ {len(essays)} load essays")
    
    print("✂️ dividing in chunks...")
    chunks = split_into_chunks(essays)
    print(f"✅ {len(chunks)} chunks generated")
    
    print("🔢 generating embedding and saving in chromaDB...")
    store_in_chromadb(chunks)