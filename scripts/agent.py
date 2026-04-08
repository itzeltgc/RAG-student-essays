import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
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

prompt = ChatPromptTemplate.from_messages([
    ('system', 'Your are an expert college admissions essay coach. Use the avilable tools to help students improve their essays.'),
    ('human', '{input}'),
    ('placeholder', '{agent_scratchpad}')
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)



if __name__ == "__main__":
    print("🎓 Essay Coach Agent — How can I help you today?")
    print("(Paste your request, e.g. 'Analyze the structure of my essay: [your essay]')\n")
    user_input = input("You: ")
    result = agent_executor.invoke({"input": user_input})
    print("\n📝 Response:\n")
    print(result["output"])