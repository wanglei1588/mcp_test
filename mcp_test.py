from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from env_utils import QWEN_API_KEY, QWEN_BASE_URL, SMITH_API_KEY
import asyncio

mcp_client = MultiServerMCPClient(
    {
        "amap-map-streamableHTTP": {
            "transport": "streamable_http",
            "url": "https://restapi.amap.com/v3/weather/weatherInfo?city=110101&key="+GAODE_API_KEY
        }
    }

)

tools = mcp_client.get_tools()

llm = ChatOpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    model="qwen-plus",
    temperature=0.2,
    # top_p=0.5,
    # max_tokens=1000,
    # max_retries=2,
    # timeout=30
)

agent = create_agent(
    model=llm,
    tools=[tools],
    # tools=[weather_tool],
    # tools=[mcp_tools],
)


if __name__ == '__main__':

    results = agent.invoke({
        "messages": [{"role": "user",
                      "content": "What is the weather in ShangHai"}]})
    print(results)
    messages = results["messages"]
    print(f"历史消息：{len(messages)} 条")
    for message in messages:
        message.pretty_print()
