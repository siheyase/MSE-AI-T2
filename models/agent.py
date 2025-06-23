# agent.py
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.reasoning import ReasoningTools
from config.settings import DEFAULT_MODEL, VS_PATH, VS_DOCS_PATH, EMBEDDING_MODEL, TOP_K, SEARCH_TYPE
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from agno.storage.sqlite import SqliteStorage
from typing import Annotated
import json
import logging
from textwrap import dedent

logger = logging.getLogger(__name__)

storage = SqliteStorage(
    table_name="agent_sessions",
    db_file="history/session.db",
    auto_upgrade_schema=True
)

# 初始化向量数据库组件一次，避免每次调用都重复加载
embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(VS_PATH, embedding_model, allow_dangerous_deserialization=True)
docstore = LocalFileStore(VS_DOCS_PATH)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key="doc_id",
    search_type=SEARCH_TYPE,
    search_kwargs={"k": TOP_K},
)

def retrieve_medical(query: Annotated[str, "需要查询的医学问题"]) -> str:
    results = retriever.invoke(query)
    contexts = []
    for raw_bytes in results:
        try:
            decoded = raw_bytes.decode("utf-8")
            doc_dict = eval(decoded)
            contexts.append(json.dumps(doc_dict, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"解码失败: {e}")
    return "\n\n".join(contexts) if contexts else "未找到相关医学资料。"


def get_agent(model_id: str = DEFAULT_MODEL, session_id=None, user_id=None) -> Agent:
    return Agent(
        name="Medical Assistant",
        model=Ollama(id=model_id),
        session_id=session_id,
        user_id=user_id,
        instructions=dedent("""\
                        你是一个智能助手，可以回答用户的各种问题。
                        向量数据库数据来源权威的医药网站“寻医问药”网，处理成结构化数据
                        数据示例：
                        {
                        "name": "肺泡蛋白质沉积症",
                        "desc": "肺泡蛋白质沉积症(简称PAP)...男性发病约3倍于女性。",
                        "category": ["疾病百科","内科","呼吸内科"],
                        "prevent": "1、避免感染分支杆菌病...因此目前一般认为本病与清除能力下降有关。",
                        "symptom": ["紫绀","胸痛","呼吸困难","乏力","毓卓"],
                        "acompany": ["多重肺部感染"],
                        "cure_department": ["内科","呼吸内科"],
                        "cure_way": ["支气管肺泡灌洗"],
                        "check": ["胸部CT检查","肺活检","支气管镜检查"],
                        "recommand_drug":...,
                        "drug_detail":...
                        }
                        如果你认为用户当前的问题无法凭借自身内部知识直接回答，需要检索类似上述的医学知识，那么使用retrieve_medical工具，例如：retrieve_medical(query)，否则无需检索直接回答\
                    """),
        tools=[ReasoningTools(add_instructions=True), retrieve_medical],
        show_tool_calls=True,
        markdown=True,
        num_history_responses=3,
        add_history_to_messages=True,
        storage=storage,
        search_previous_sessions_history=True,
        num_history_sessions=2,
    )