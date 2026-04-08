import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
client_chroma = chromadb.PersistentClient(path="data/chromadb")
collection = client_chroma.get_collection("essays")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def search_context(query: str) -> str:
    embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=3)
    return "\n\n".join(results["documents"][0])

@tool
def give_feedback(essay: str) -> str:
    """Give general feedback on a college admission essay."""
    context = search_context(essay)
    prompt = f"""You are an expert in college admission essays.
    
Reference essays:
{context}

Give specific, actionable feedback to improve this essay:
{essay}"""
    return llm.invoke(prompt).content

@tool
def analyze_structure(essay: str) -> str:
    """Analyze the structure of a college admission essay: introduction, body, and conclusion."""
    context = search_context(essay)
    prompt = f"""You are an expert in college admission essays.

Reference essays:
{context}

Analyze the structure of this essay. Evaluate the introduction, body paragraphs, and conclusion separately. Be specific:
{essay}"""
    return llm.invoke(prompt).content

@tool
def rewrite_section(input: str) -> str:
    """Rewrite a specific section of a college admission essay. Input format: 'section: [intro/body/conclusion] | essay: [full essay]'"""
    context = search_context(input)
    prompt = f"""You are an expert in college admission essays.

Reference essays:
{context}

Based on the following request, rewrite only the requested section. Make it compelling and authentic:
{input}"""
    return llm.invoke(prompt).content



tools = [give_feedback, analyze_structure, rewrite_section]
llm_with_tools = llm.bind_tools(tools)

def run_agent(user_input: str):
    messages = [{"role": "user", "content": user_input}]
    response = llm_with_tools.invoke(messages)
    
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"\n🔧 Agent selected tool: {tool_name}\n")
        
        tool_map = {
            "give_feedback": give_feedback,
            "analyze_structure": analyze_structure,
            "rewrite_section": rewrite_section
        }
        result = tool_map[tool_name].invoke(tool_args)
        return result
    else:
        return response.content



if __name__ == "__main__":
    with open("data/test/test_essay.txt", "r", encoding="utf-8") as f:
        essay = f.read()

    print("🎓 Essay Coach Agent")
    print("Options:")
    print("1. General feedback")
    print("2. Analyze structure")
    print("3. Rewrite a section (specify: intro/body/conclusion)")
    print()
    
    user_choice = input("What would you like to do with your essay? ")
    user_input = f"{user_choice} | essay: {essay}"
    
    result = run_agent(user_input)
    print("\n📝 Response:\n")
    print(result)