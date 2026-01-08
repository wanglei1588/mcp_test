# This is a sample Python script.
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import langchain
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from env_utils import QWEN_API_KEY, QWEN_BASE_URL, SMITH_API_KEY


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "first-pro"
os.environ["LANGSMITH_API_KEY"] = SMITH_API_KEY


# 载入对话模型/非对话模型/嵌入模型
llm1 = ChatOpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
    model_name="qwen-plus",
    temperature=0.5,
    top_p=0.5,
    max_tokens=1000,
    max_retries=2,
    timeout=30
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名理想汽车的研发工程师"),
    ("user", "{input}")  # {input}为变量
])

chain = prompt | llm1

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(langchain.__version__)

    res = chain.invoke({"input": "你最喜欢的汽车是什么？为什么？"})
    print(res)


