from langchain.agents import create_agent
import dotenv
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import DashScopeEmbeddings
from env_utils import QWEN_API_KEY, QWEN_BASE_URL, SMITH_API_KEY
from langchain_chroma import Chroma
import dotenv


# dotenv.load_dotenv()
import os


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "first-pro"
os.environ["LANGSMITH_API_KEY"] = SMITH_API_KEY

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


embeddings = DashScopeEmbeddings(
        dashscope_api_key=QWEN_API_KEY,
        model="text-embedding-v1",
    )

vector_store = Chroma(
    persist_directory="./asset/chroma-1",
    embedding_function=embeddings,
    collection_name="my_collection",
)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query"""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    content = '\n\n'.join(
        f"Source:{doc.metadata}\nContent:{doc.page_content}" for doc in retrieved_docs
    )
    return content, retrieved_docs


system_prompt = """
    你可以使用信息检索工具，回答用户的问题，如果用户使用中文提问，那么请你用中文回答。
"""


agent = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt=system_prompt
)


if __name__ == '__main__':

    results = agent.invoke({"messages": [{"role": "user", "content": "Houston 数据集中有多少类别？"}]})
    print(results)
    messages = results["messages"]
    print(f"历史消息：{len(messages)} 条")
    for message in messages:
        message.pretty_print()
