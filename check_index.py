import json
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore

vs_path = Path.cwd() / "vs"
vs_docs_path = Path.cwd() / "docs"
embedding_model = OllamaEmbeddings(model="bge-m3")
vectorstore = FAISS.load_local(vs_path, embedding_model, allow_dangerous_deserialization=True)
docstore = LocalFileStore(vs_docs_path)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key="doc_id",
    search_type="similarity",   # similarity or mmr最大边际相关检索
    search_kwargs={"k": 3},     # 控制返回文档数量
)

# 进行查询测试
query = "出现呼吸困难怎么办？"
results = retriever.invoke(query)

for i, raw_bytes in enumerate(results):
    print(f"\n--- 原始文档 {i+1} ---")
    decoded = raw_bytes.decode("utf-8")
    # 转换为 JSON 对象（字典）
    try:
        doc_dict = eval(decoded)
        print(json.dumps(doc_dict, indent=2, ensure_ascii=False))
    except Exception as e:
        print("⚠️ 解码失败:", e)


# 遍历 vectorstore 中的所有分块，
# print(f"📦 当前向量数量: {vectorstore.index.ntotal}")
# print(f"🔗 索引到文档ID映射数: {len(vectorstore.index_to_docstore_id)}")

# for i in range(vectorstore.index.ntotal):
#     doc_id = vectorstore.index_to_docstore_id[i]
#     doc = vectorstore.docstore.search(doc_id)
#
#     print(f"\n--- Chunk {i+1} ---")
#     print(f"🆔 文档ID: {doc_id}")
#     print(f"📄 内容片段:\n{doc.page_content[:300]}...")
#     print(f"📎 元数据: {doc.metadata}")


