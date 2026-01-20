from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import wikipedia
from langchain.agents import create_agent
from dotenv import load_dotenv
from pprint import pprint
from langchain.tools import tool


load_dotenv()

# -------------------------
# LLM (Groq)
# -------------------------
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1
)

# -------------------------
# Tools
# -------------------------
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(query) -> str:
    return "1729"

search_tool = TavilySearch(max_results=2)
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
    wiki_client=wikipedia
))

tools = [search_tool, wikipedia_tool, calc]

# -------------------------
# Agent
# -------------------------
agent = create_agent(model, tools=tools)

# -------------------------
# Run
# -------------------------
query = (
    "Research the latest trends in agentic AI systems. "
    "Summarize key ideas in bullet points and include sources."
)

query2 = "What is the value of 12^3 + 1? Who is the discoverer of this special number? Find more about it from wikipedia."

print("Starting research...")
result = agent.invoke(
    {"messages": [{"role": "user", "content": query2}]}
)

pprint("\nFINAL OUTPUT:\n")
pprint(result)
