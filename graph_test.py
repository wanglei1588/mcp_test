from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from env_utils import QWEN_API_KEY, QWEN_BASE_URL
from langchain_core.tools import tool
# _ = load_dotenv()

# tool = TavilySearchResults(max_results=4)  # increased number of results


@tool("my_weather_tool", description="You can use this tool to get weather information.")
def get_weather(city: str) -> str:
    """Get weather for a given city
    Args:
        city:需要获取天气的城市
    """
    return f"It's alwasy sunny in {city}!"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")    # 表示设置起始节点

        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:  # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

model = ChatOpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    model="qwen-plus",
    temperature=0.2,
    # top_p=0.5,
    # max_tokens=1000,
    # max_retries=2,
    # timeout=30
)
abot = Agent(model, [get_weather], system=prompt)

messages = [HumanMessage(content="What is the weather in sf?")]
result = abot.graph.invoke({"messages": messages})

# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_PvPN1v7bHUxOdyn4J2xJhYOX'}

print(result['messages'][-1].content)
# 'The weather in San Francisco today is cloudy with overcast skies
# and a temperature of 63°F during the day and 54°F at night.
# There is no precipitation expected.'

messages = [HumanMessage(content="What is the weather in SF and LA?")]
result = abot.graph.invoke({"messages": messages})

# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_1SqGYuEtOOFN1yiIHSQTPnvE'}
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in Los Angeles'}, 'id': 'call_8RiM72Y7G8V7c3HEEAML1SKP'}

print(result['messages'][-1].content)
# 'The weather in San Francisco today is cloudy with overcast skies
# and temperatures around 63°F during the day and 54°F at night.
# There is no precipitation expected.\n\n
# In Los Angeles, the weather is also cloudy with temperatures reaching around 75°F during the day and 61°F at night.
# There is no precipitation expected in Los Angeles as well.'

query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
What is the GDP of that state? Answer each question."
messages = [HumanMessage(content=query)]

model = ChatOpenAI(model="gpt-4o")
abot = Agent(model, [tool], system=prompt)
result = abot.graph.invoke({"messages": messages})

# Calling: {'name': 'tavily_search_results_json', 'args': {'query': '2024 Super Bowl winner'}, 'id': 'call_HBUU1Lo9WSgKCPKYCAStSb7g'}
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'Kansas City Chiefs headquarters location'}, 'id': 'call_byHxMKTlXnnfKkOx2KwaWAKi'}
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'Missouri GDP 2024'}, 'id': 'call_tvMjaXlqtIWseC2qBCsUGpSF'}

print(result['messages'][-1].content)
# 1. **Who won the Super Bowl in 2024?**
#    - The Kansas City Chiefs won the Super Bowl in 2024, defeating the San Francisco 49ers with a score of 25-22 in overtime.
# 2. **In what state is the winning team's headquarters located?**
#    - The Kansas City Chiefs are headquartered in Kansas City, Missouri.
# 3. **What is the GDP of that state?**
#    - Missouri's GDP was approximately $460.7 billion in the fourth quarter of 2024.
