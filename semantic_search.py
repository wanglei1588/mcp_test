from langchain_community.embeddings import DashScopeEmbeddings
from env_utils import QWEN_API_KEY, QWEN_BASE_URL
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import chain


embeddings = DashScopeEmbeddings(
        dashscope_api_key=QWEN_API_KEY,
        model="text-embedding-v1",
    )

vector_store = Chroma(
    persist_directory="./asset/chroma-1",
    embedding_function=embeddings,
    collection_name="my_collection",
)


@chain
def retriever(question: str) -> list[Document]:
    return vector_store.similarity_search(question, k=1)


if __name__ == '__main__':

    # 相似度查询
    print("相似度查询---------------------------------------------------------------")
    results = vector_store.similarity_search("what is SMF Excitation ?", k=3)
    for id, result in enumerate(results):
        print(id)
        print(result.page_content[:50], "\n")

    # 带分数相似度查询
    print("带分数相似度查询---------------------------------------------------------------")
    results = vector_store.similarity_search_with_score("what is SMF Excitation ?", k=3)
    for search_res, score in results:
        print(score)
        print(search_res.page_content[:50], "\n")

    # 用向量进行相似度查询
    print("用向量进行相似度查询---------------------------------------------------------------")
    vct = embeddings.embed_query(text="what is SMF Excitation ?")
    results = vector_store.similarity_search_by_vector(vct, k=3)
    for id, result in enumerate(results):
        print(id)
        print(result.page_content[:50], "\n")

    # 用检索器进行查询
    print("用检索器进行查询---------------------------------------------------------------")
    results = retriever.invoke("what is SMF Excitation ?")
    for id, result in enumerate(results):
        print(id)
        print(result.page_content[:50], "\n")

# 检索器工具
    print("检索器工具---------------------------------------------------------------")
    retriever_tool = vector_store.as_retriever()
    results = retriever_tool.invoke("what is SMF Excitation ?", k=3)
    for id, result in enumerate(results):
        print(id)
        print(result.page_content[:50], "\n")




