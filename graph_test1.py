from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send, interrupt
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from env_utils import QWEN_API_KEY, QWEN_BASE_URL
from langchain_core.tools import tool
# _ = load_dotenv()


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


def human_approval(state: MessagesState):
    """Ask human for approval."""
    is_approval=interrupt({
        "question": "是否调用大模型",
        "choices": ["Yes", "No"]
    })
    if is_approval == "Yes":
        return Command(goto="call_model")
    else:
        return Command(goto=END)


def call_model(state: MessagesState):
    """Call the model with the given messages and return the result."""
    response = llm.invoke(state["messages"])
    return {"messages": response}


checkpoint_saver = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)

builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

graph = builder.compile(checkpointer=checkpoint_saver)


if __name__ == '__main__':
    for chunk in graph.stream({"messages": [HumanMessage(content="中国有几个自治区")]},
                              stream_mode="messages",
                              config=config):
        print(chunk)
        # print(chunk["messages"].content, end='')

    for chunk in graph.stream({"messages": [HumanMessage(content="面积分别是多大")]},
                              stream_mode="messages",
                              config=config):
        # print(chunk)
        print(chunk[0].content, end='')


"""
{'call_model': 
            {'messages': AIMessage(content='中国有**5个自治区**，分别是：\n\n1. **内蒙古自治区**（1947年成立，中国第一个省级民族自治区）  \n2. **广西壮族自治区**  \n3. **西藏自治区**  \n4. **宁夏回族自治区**  \n5. **新疆维吾尔自治区**\n\n这些自治区是中华人民共和国的省级行政区，依法享有宪法和《民族区域自治法》赋予的自治权，包括自主管理本民族内部事务、使用和发展本民族语言文字、保持或改革本民族风俗习惯等权利。\n\n✅ 补充说明：  \n- 自治区与省、直辖市、特别行政区同属中国**4类省级行政区**之一；  \n- 全国共有**34个省级行政区**（23个省、5个自治区、4个直辖市、2个特别行政区）。\n\n如需了解各自治区的成立时间、首府、主要民族或地理特点，也欢迎继续提问！', 
                                    additional_kwargs={'refusal': None}, 
                                    response_metadata={'token_usage': {'completion_tokens': 199, 
                                                                            'prompt_tokens': 11, 
                                                                            'total_tokens': 210, 
                                                                            'completion_tokens_details': None, 
                                                                            'prompt_tokens_details': {'audio_tokens': None, 
                                                                                                        'cached_tokens': 0}}, 
                                                        'model_provider': 'openai', 
                                                        'model_name': 'qwen-plus', 
                                                        'system_fingerprint': None, 
                                                        'id': 'chatcmpl-ae132099-e97c-9d12-a981-0abbb273506a', 
                                                        'finish_reason': 'stop', 
                                                        'logprobs': None}, 
                                    id='lc_run--019bf59a-0f6e-7ea1-8a43-dc2f773561cb-0', 
                                    tool_calls=[], 
                                    invalid_tool_calls=[], 
                                    usage_metadata={'input_tokens': 11, 
                                                    'output_tokens': 199, 
                                                    'total_tokens': 210, 
                                                    'input_token_details': {'cache_read': 0}, 
                                                    'output_token_details': {}})}}

"""


