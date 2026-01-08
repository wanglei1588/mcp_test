from langchain.agents import create_agent
import dotenv
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi
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


def get_weather(city: str) -> str:
    """Get weather for a given city"""
    return f"It's alwasy sunny in {city}!"


agent = create_agent(
    model=llm,
    tools=[get_weather],
)


# print(agent)
# langgraph.graph.state.CompiledStateGraph
# Graph: nodes - edges 网状

print(agent.nodes)
# {
#   '__start__': <langgraph.pregel._read.PregelNode object at 0x000001FD444BDC50>,
#   'model': <langgraph.pregel._read.PregelNode object at 0x000001FD44A67250>
# }

# for event in agent.stream({"messages": [{"role": "user", "content": "What is the weather in SF"}]}, stream_mode="values"):
#     messages = event["messages"]
#     print(f"历史消息：{len(messages)} 条\n")
#     # for message in messages:
#     #     message.pretty_print()
#     # print(messages[-1].content)
#     print(messages[-1].pretty_print())

for chunk in agent.stream(
    {"messages":[{"role": "user", "content": "What is the weather in SF"}]},
    stream_mode="messages"  # token by token
):

    print(chunk[0].content, end='')
    # print(chunk)
    # (AIMessage(content="Glad to hear it's always sunny in SF! Let me know if you'd like weather updates for any other cities. ☀️",
    #           additional_kwargs={},
    #           response_metadata={'model_name': 'qwen-plus',
    #                               'finish_reason': 'stop',
    #                               'request_id': 'd628d5f5-c734-4228-b70c-2d037a5993cf',
    #                               'token_usage': {'input_tokens': 193,
    #                                               'output_tokens': 28,
    #                                               'total_tokens': 221,
    #                               'prompt_tokens_details': {'cached_tokens': 0}}},
    #            id='lc_run--019b6517-a07a-7e93-8da6-e9cd1c7f2372-0'),
    #  {'langgraph_step': 3,
    #  'langgraph_node': 'model',
    #  'langgraph_triggers': ('branch:to:model',),
    #  'langgraph_path': ('__pregel_pull', 'model'),
    #  'langgraph_checkpoint_ns': 'model:1ec38ca2-4276-3cfa-1f50-68f1a16c111f',
    #  'checkpoint_ns': 'model:1ec38ca2-4276-3cfa-1f50-68f1a16c111f',
    #  'ls_provider': 'tongyi',
    #  'ls_model_type': 'chat',
    #  'ls_model_name': 'qwen-plus'})

