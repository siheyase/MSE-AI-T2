import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid
import faiss
from tqdm import tqdm


splitter = RecursiveJsonSplitter(max_chunk_size=2000)
embedding_model = OllamaEmbeddings(model="bge-m3")
vs_docs_path = Path.cwd() / "docs"
vs_path = Path.cwd() / "vs"
docs_store = LocalFileStore(vs_docs_path)

# 确定距离向量索引
index = faiss.IndexFlatL2(len(embedding_model.embed_query("test")))
vectorstore = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

id_key = "doc_id"
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docs_store,
    id_key=id_key,
)


with open("./doc/medical.json", "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

with open("./doc/medical.json", "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines, desc="Building FAISS index"):
        entry = json.loads(line)
        # 提取元数据
        doc_id = str(uuid.uuid4())
        metadata = {
            id_key: doc_id,
            "name": entry.get("name", ""),
            "category": ",".join(entry.get("category", [])),
        }

        sub_docs = splitter.create_documents(
            texts=[entry],
            convert_lists=True,  # 推荐设为 True：能将 list 转换为 dict，便于嵌套处理
            ensure_ascii=False,
            metadatas=[metadata],
        )
        retriever.vectorstore.add_documents(sub_docs)
        retriever.docstore.mset([(doc_id, str(entry).encode())])

        vectorstore.save_local(vs_path)

# ✅ 输出统计信息
print(f"\n📦 当前向量分块数量: {vectorstore.index.ntotal}")
print(f"🔗 索引到文档ID映射数: {len(vectorstore.index_to_docstore_id)}")
