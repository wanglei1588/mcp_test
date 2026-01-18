from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, Tool, BaseTool
from langchain_community.embeddings import DashScopeEmbeddings
from env_utils import QWEN_API_KEY, QWEN_BASE_URL, SMITH_API_KEY, GAODE_API_KEY
from langchain_community.agent_toolkits.load_tools import load_tools
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio


class PaperInfo(BaseModel):
    title: str = Field(..., description="The title of the paper")
    author: list[str] = Field(..., description="The auther of the paper")
    time: Optional[str] = Field(default=None, description="The time of the paper")     # 时间,可选
    keywords: list[str] = Field(..., description="The keywords of the paper")


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

# llm.with_structured_output(PaperInfo)


# weather_schema = {
#     # "title": "PaperInfo",
#     "type": "object",
#     "properties": {
#         "city": {"type": "string", "description": "需要获取天气的城市"},
#         # "author": {"type": "array", "items": {"type": "string"}, "description": "The auther of the paper"},
#         # "time": {"type": "string", "description": "The time of the paper"},
#         # "keywords": {"type": "array", "items": {"type": "string"}, "description": "The keywords of the paper"},
#     }
#     ,
#     "required": ["city", ]
# }


class weather_schema(BaseModel):
    city: str = Field(..., description="需要获取天气的城市")     # "..."表示必传


# 工具使用1：tool---------------------------
@tool("aaa", description="You can use this tool to get weather information.", args_schema=weather_schema)
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


arxiv_tool = load_tools(["arxiv"])


weather_tool = Tool(
    func=get_city_weather,
    name="Search",
    description="Get weather for a given city.",
)

# mcp_client = MultiServerMCPClient(
#     {
#         "amap-map-streamableHTTP": {
#             "transport": "streamable_http",
#             "url": "https://restapi.amap.com/v3/weather/weatherInfo?city=110101&key="+GAODE_API_KEY
#         }
#     }
#
# )


# async def get_weather_from_mcp() -> list[BaseTool]:
#     """Get weather for a given city
#     Args:
#         city:需要获取天气的城市
#     """
#     return await mcp_client.get_tools()
#
# mcp_tools = get_weather_from_mcp()

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    model=llm,
    tools=[get_weather],
    # tools=[weather_tool],
    # tools=[mcp_tools],
)

arxiv_agent = create_agent(
    model=llm,
    tools=arxiv_tool,
    system_prompt="""
    You are an expert in arxiv.
    You can use the arxiv tool to search for papers.
    """,
    # 结构化输出
    response_format=ToolStrategy(
        schema=PaperInfo,
        tool_message_content="调用了arxiv工具"
    ),
    # middleware=[
    #     SummarizationMiddleware(
    #         model=llm,
    #         system_prompt="请总结并返回结果",
    #         # keep=("message", 20)
    #     )
    # ],
    checkpointer=checkpointer,
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

    # results = arxiv_agent.invoke({"messages": [{"role": "user",
    #                                             "content": "What is the paper 2312.08874 about?"}]}, config=config)
    #
    # print(results)
    # print("-------------------------------------\n")
    # print(results["structured_response"])
    # print("-------------------------------------\n")
    # messages = results["messages"]
    # print(f"历史消息：{len(messages)} 条")
    # for message in messages:
    #     message.pretty_print()



"""
result:
{'messages': [HumanMessage(content='What is the weather in ShangHai', 
                            additional_kwargs={}, 
                            response_metadata={}, 
                            id='35fefa49-0ee2-4c87-a069-c0803533b8be'
                            ), 
                AIMessage(content='', 
                            additional_kwargs={'refusal': None}, 
                            response_metadata={'token_usage': {'completion_tokens': 21, 
                                                                'prompt_tokens': 166, 
                                                                'total_tokens': 187, 
                                                                'completion_tokens_details': None, 
                                                                'prompt_tokens_details': {'audio_tokens': None, 
                                                                                            'cached_tokens': 0}}, 
                                                                'model_provider': 'openai', 
                                                                'model_name': 'qwen-plus', 
                                                                'system_fingerprint': None, 
                                                                'id': 'chatcmpl-68d6f435-3a89-9c09-bd84-abdf87a49723', 
                                                                'finish_reason': 'tool_calls', 
                                                                'logprobs': None
                                                                }, 
                            id='lc_run--019bd130-49fc-7ce3-9a9a-c7b7f4106548-0', 
                            tool_calls=[{'name': 'aaa', 
                                        'args': {'city': 'ShangHai'}, 
                                        'id': 'call_f226f0130acc42a0ab026e', 
                                        'type': 'tool_call'
                                        }], 
                            invalid_tool_calls=[], 
                            usage_metadata={'input_tokens': 166, 
                                            'output_tokens': 21, 
                                            'total_tokens': 187, 
                                            'input_token_details': {'cache_read': 0}, 
                                            'output_token_details': {}
                                            }
                            ), 
                ToolMessage(content="It's alwasy sunny in ShangHai!", 
                            name='aaa', 
                            id='098dbd70-ec88-427f-8326-a13059bf6b14', 
                            tool_call_id='call_f226f0130acc42a0ab026e'
                            ), 
                AIMessage(content="It seems like it's always sunny in Shanghai! Enjoy the bright weather! 😊☀️", 
                        additional_kwargs={'refusal': None}, 
                        response_metadata={'token_usage': {'completion_tokens': 19, 
                                                            'prompt_tokens': 212, 
                                                            'total_tokens': 231, 
                                                            'completion_tokens_details': None, 
                                                            'prompt_tokens_details': {'audio_tokens': None, 
                                                                                        'cached_tokens': 0}}, 
                                                            'model_provider': 'openai', 
                                                            'model_name': 'qwen-plus', 
                                                            'system_fingerprint': None, 
                                                            'id': 'chatcmpl-579f69a4-1d0d-95c8-953b-c7cce4171bf3', 
                                                            'finish_reason': 'stop', 
                                                            'logprobs': None}, 
                                            id='lc_run--019bd130-5189-7c82-b3a9-c80593d4710a-0', 
                                            tool_calls=[], 
                                            invalid_tool_calls=[], 
                                            usage_metadata={'input_tokens': 212, 
                                                            'output_tokens': 19, 
                                                            'total_tokens': 231, 
                                                            'input_token_details': {'cache_read': 0}, 
                                                            'output_token_details': {}
                                                            }
                        )
                ]
}

"""