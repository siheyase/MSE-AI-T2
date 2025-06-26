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
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

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

def db_history_queries():
    """
    从 SQLite 数据库中获取历史查询记录
    """
    # 连接到 SQLite 数据库
    conn = sqlite3.connect('history/session.db')
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM agent_sessions")
        history_queries = cursor.fetchall()
    except sqlite3.Error as e:
        print(f"数据库查询失败: {e}")
        return []
    # 关闭数据库连接
    conn.close()

    # 提取第二个字段中的JSON内容
    json_data = []
    for session in history_queries:
        try:
            json_data.append(json.loads(session[2]))
        except json.JSONDecodeError:
            continue  # 跳过无效数据

    # 提取历史消息
    messages = []
    for parsed_data in json_data:
        for message in parsed_data["runs"][-1]["messages"]: # 该session最后一次运行的message包含该次全部对话
            if message["role"] == "user":
                query = message["content"]
            elif message["role"] == "assistant":
                response = message["content"]
                # 将查询和回应一起存储
                messages.append({
                    "query": query,
                    "response": response
                })
    print(f"从数据库中获取到 {len(messages)} 条历史查询记录。")
    return messages

# RAG优化-获取与当前查询相关的历史查询记录，并返回格式化后的问答对
def get_relevant_history_queries(current_query: str):
    """
    current_query: str
    """
    # 从数据库中获取历史查询记录
    history_queries = db_history_queries()
    if len(history_queries) < 2:
        print("历史查询记录不足，无法进行相关性检索。")
        return "未找到相关历史会话。"

    # 将当前查询向量化
    current_query_embedding = np.array(embedding_model.embed_query(current_query)).reshape(1, -1)
    # 存储历史记录的嵌入向量
    history_embeddings = []

    # 对每条历史记录分别向量化
    for entry in history_queries[:-2]:  # 排除最近的2条记录
        history_text = entry["query"] + " " + entry["response"]
        history_embedding = np.array(embedding_model.embed_query(history_text)).reshape(1, -1)
        history_embeddings.append(history_embedding)

    # 将历史嵌入列表转化为一个二维数组 (m, n)，m是历史记录数，n是每个嵌入的维度
    history_embeddings = np.vstack(history_embeddings)
    # 计算当前查询与历史查询的余弦相似度
    similarities = cosine_similarity(current_query_embedding, history_embeddings)
    # 获取最相关的历史查询记录（选择相似度最高的2条）
    top_n = 2
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    # 输出最相关的历史记录
    relevant_history = [history_queries[i] for i in top_indices]
    print(f"当前查询: {current_query}，相关历史记录索引: {top_indices}, 相似度: {similarities[0][top_indices]}")

    # 构造格式化后的问答对字典
    history_context = []
    for entry in relevant_history:
        history_context.append({
            "query": entry["query"],
            "response": entry["response"]
        })

    # 将字典列表转换为JSON格式字符串
    return json.dumps(history_context, ensure_ascii=False) if history_context else "未找到相关历史会话。"


# agno的agent推理功能依据prompt就会自动进行简单的查询重写，以及工具函数的返回结果并不会直接作为调用该工具的agent的返回，
# 而是会进一步用于推理回答，所以工具函数无需再实例化一个agent
# 即retrieve_medical只需要返回检索结果
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

storage = SqliteStorage(
    table_name="agent_sessions",
    db_file="history/session.db",
    auto_upgrade_schema=True
)

# agent
def get_agent(model_id: str = DEFAULT_MODEL, session_id=None, user_id=None) -> Agent:
    return Agent(
        name="Medical Assistant",
        model=Ollama(id=model_id),
        session_id=session_id,
        user_id=user_id,
        instructions=dedent("""\
                            你是一位专业的医学智能问诊助手，擅长通过自然对话的方式，与用户进行多轮问诊，逐步了解其症状和病情，并提供科学、合理的健康建议。

                            你的行为遵循以下原则：

                            【角色定位】
                            - 你不是医生，不能下诊断结论，但你可以辅助用户了解自身状况，并建议下一步行动。
                            - 你风格亲切、耐心、结构化，面对非专业用户时表达清晰易懂。

                            【核心任务】
                            1. 从用户的描述中提取主诉（例如“我头痛”、“我咳嗽”）。
                            2. 引导用户补充关键问诊信息，包括但不限于：
                               - 起始时间、持续时间、症状部位、强度、频率
                               - 是否有伴随症状（如发热、乏力、恶心、咳痰等）
                               - 是否有外因诱发因素，例如：
                                 * 最近是否接触过**花粉、粉尘、宠物、特殊食物、药物**？
                                 * 是否有**过敏史**或**新环境暴露**（装修、旅游等）？
                                 * 是否近期接触感冒患者、天气变化、工作环境变化等？
                               - 有无基础疾病史（如哮喘、胃病、糖尿病等）
                            3. 请使用get_relevant_history_queries工具指令 `get_relevant_history_queries(query)` 获取与当前提问相关的历史查询记录，如有完全相同的提问可以直接返回历史回答，并在此基础上询问用户是否哪里理解不清楚。
                            4. 在合适时机使用工具指令 `retrieve_medical("疾病名")` 查询相关疾病的结构化信息。
                            5. 在信息收集充分后，整理并输出一份面向医生的简要病例描述，并建议用户就诊方向（如科室或检查类型）。

                            【行为规范】
                            - 不要急于给出结论，应先通过追问获取更多上下文。
                            - 每轮回复中包含：
                               ① 对已有信息的简要分析  
                               ② 引导用户补充具体细节（如典型症状、发作特点、外因诱因）  
                               ③ 如需要，可建议调用 `retrieve_medical` 工具来辅助分析。

                            【示例】
                            用户：我最近咳嗽，有时候喉咙痒
                            助手：
                            明白了，请问咳嗽持续了多久了？是否是干咳还是有痰？有没有发烧、气喘、胸闷等伴随症状？  
                            另外，最近是否接触过花粉、灰尘、宠物或吃了不寻常的食物？是否有过敏史？

                            【最终目标】
                            通过每一轮自然对话，逐步完善用户的病情信息，最后输出结构化摘要供医生参考，如：
                            - 主诉：咳嗽伴喉咙痒，持续5天
                            - 伴随症状：无发烧，有轻微胸闷
                            - 诱因：宠物接触后加重，有过敏史
                            - 疑似方向：过敏性咽炎或轻度哮喘
                            - 建议：建议前往呼吸科就诊，必要时进行过敏原检测
                            如果你认为用户当前的问题无法凭借自身内部知识直接回答，需要检索类似上述的医学知识，那么使用retrieve_medical工具，例如：retrieve_medical(query)，否则无需检索直接回答\
                        """),
        tools=[ReasoningTools(add_instructions=True), retrieve_medical, get_relevant_history_queries],
        show_tool_calls=True,
        markdown=True,
        num_history_responses=3,
        add_history_to_messages=True,
        storage=storage,
        search_previous_sessions_history=True,
        num_history_sessions=2,
    )

