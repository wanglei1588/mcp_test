from langchain.agents import create_agent
import dotenv
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool, Tool
from langchain_community.embeddings import DashScopeEmbeddings
from env_utils import QWEN_API_KEY, QWEN_BASE_URL, SMITH_API_KEY


llm = ChatTongyi(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    model="qwen-plus",
    temperature=0.5,
    top_p=0.5,
    max_tokens=1000,
    max_retries=2,
    timeout=30
)


# 工具使用1：tool---------------------------
@tool("aaa", description="You can use this tool to get weather information.")
def get_weather(city: str) -> str:
    """Get weather for a given city
    Args:
        city:需要获取天气的城市
    """
    return f"It's alwasy sunny in {city}!"


print(get_weather.name)
print(get_weather.description)

# 工具使用2：Tool---------------------------
def get_city_weather(city: str) -> str:
    # """Get weather for a given city
    # Args:
    #     city:需要获取天气的城市
    # """
    return f"It's alwasy sunny in {city}!"


weather_tool = Tool(
    func=get_city_weather,
    name="Search",
    description="Get weather for a given city.",
)


agent = create_agent(
    model=llm,
    tools=[get_weather],
    # tools=[weather_tool],
)


results = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in ShangHai"}]})
print(results)
messages = results["messages"]
print(f"历史消息：{len(messages)} 条")
for message in messages:
    message.pretty_print()
