import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()


client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))               # conection with appi key
client_chroma = chromadb.PersistentClient(path="data/chromadb")     # local vectorial database
collection = client_chroma.get_collection("essays")                 # 87 chuncks that we indexed
model = SentenceTransformer("all-MiniLM-L6-v2")                     # model 




def search_similar_essays(query_text, n_results=3):
    query_embedding = model.encode(query_text).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0]




def get_feedback(student_essay):
    similar_chunks = search_similar_essays(student_essay)
    context = "\n\n".join(similar_chunks)
    
    prompt = f"""Eres un experto en ensayos de admisión universitaria.
    
Tienes acceso a los siguientes fragmentos de ensayos de referencia:
{context}

Con base en estos ejemplos, proporciona retroalimentación específica y útil para mejorar el siguiente ensayo:

{student_essay}

Retroalimentación:"""

    response = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content






if __name__ == "__main__":
    print("🎓 Essay Coach RAG - Ingresa tu ensayo:")
    with open("data/test/test_essay.txt", 'r', encoding='utf-8') as f:
        student_essay = f.read()
    
    print("\n🔍 Buscando ensayos similares...")
    print("\n📝 Generando retroalimentación...\n")
    
    feedback = get_feedback(student_essay)
    print(feedback)