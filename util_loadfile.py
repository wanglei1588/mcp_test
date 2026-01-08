from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

import os
from openai import OpenAI
from env_utils import QWEN_API_KEY, QWEN_BASE_URL
from langchain_chroma import Chroma


if __name__ == '__main__':

    # 读取文件，按页管理-------------------------------------------
    file_path = r"C:\Users\lenovo\Desktop\paper\wl_igarss.pdf"
    loader = PyPDFLoader(file_path)
    doc = loader.load()

    print("文本总页数：", len(doc))
    print(type(doc[0]))
    # print(doc[0])

    # 将文档分块-------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(doc)
    print("总分块数：", all_splits.__len__())
    print(all_splits[0].page_content)
    print(type(all_splits[0].page_content))

    # 嵌入模型--------------------------------------------------------
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=QWEN_API_KEY,
        model="text-embedding-v1",
    )

    vector0 = embeddings.embed_query(text="what is SMF Excitation ?",)
    print("vector dim", len(vector0))
    print("vector: \n", vector0)

    vector_store = Chroma(
        persist_directory="./asset/chroma-1",
        embedding_function=embeddings,
        collection_name="my_collection",
    )

    ids = vector_store.add_documents(documents=all_splits)
    print(len(ids))
    print(ids)




