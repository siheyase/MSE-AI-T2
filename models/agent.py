from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.reasoning import ReasoningTools
from agno.tools.function import Function
from config.settings import DEFAULT_MODEL, VS_PATH, VS_DOCS_PATH, EMBEDDING_MODEL, TOP_K, SEARCH_TYPE
import logging
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from typing import Annotated
from textwrap import dedent
import json


logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG智能体类，封装了与模型交互的功能
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        初始化RAG智能体
        """
        self.model_name = model_name
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """
        创建Agent实例
        :return: Agent: Agno Agent实例
        """
        # 初始化向量数据库
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(VS_PATH, embedding_model, allow_dangerous_deserialization=True)
        docstore = LocalFileStore(VS_DOCS_PATH)
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key="doc_id",
            search_type=SEARCH_TYPE,  # similarity or mmr最大边际相关检索
            search_kwargs={"k": TOP_K},  # 控制返回文档数量
        )

        def retrieve_medical(
            query: Annotated[str, "需要查询的医学问题"]
        ) -> str:
            """
            agent工具，用于RAG检索向量数据库，返回相关医学上下文
            :param query: 查询问题
            :return: str: 依据召回上下文的回答
            """
            results = retriever.invoke(query)
            contexts = []
            for raw_bytes in results:
                try:
                    decoded = raw_bytes.decode("utf-8")
                    doc_dict = eval(decoded)    # 将一个字符串形式的 Python 对象（如字典）转换成真正的 Python 对象。
                    contexts.append(json.dumps(doc_dict, ensure_ascii=False))
                except Exception as e:
                    logger.warning(f"解码失败: {e}")
            ctx_str = "\n\n".join(contexts) if contexts else "未找到相关医学资料。"

            rag_agent = Agent(
                name="Qwen2.5 RAG Agent",
                model=Ollama(id=self.model_name),
                instructions=f"""你是一个RAG智能助手，可以依据检索召回的上下文知识回答用户的各种问题。""",
                tools=[
                    ReasoningTools(add_instructions=True),
                ],
                show_tool_calls=True,
                markdown=True
            )
            prompt = f"【检索内容】\n{ctx_str}\n\n【用户问题】\n{query}\n\n请严格按照【检索内容】作答。"
            print(prompt)
            rag_response = rag_agent.run(prompt, stream=True, stream_intermediate_steps=True)
            return rag_response


        return Agent(
            name="Qwen2.5 Agent",
            model=Ollama(id=self.model_name),
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
            tools=[
                ReasoningTools(add_instructions=True),
                retrieve_medical
            ],
            show_tool_calls=True,
            markdown=True
        )

    # def run(self, prompt: str) -> str:
    #     """
    #     运行agent处理查询
    #     :param prompt: 用户输入的提示
    #     :return: str:agent的响应
    #     """
    #     full_prompt = f"【用户问题】\n{prompt}\n\n请提供准确、有帮助的回答。"
    #     # 让Agent处理请求，会自动判断是否需要检索
    #     response = self.agent.run(full_prompt)
    #     # self.agent.print_response(full_prompt, stream=True)
    #     return response.content
    def run(self, prompt: str):
        """
        运行agent处理查询
        :param prompt: 用户输入的提示
        :return: str:agent的响应
        """
        full_prompt = f"【用户问题】\n{prompt}\n\n请判断是否需要调用工具检索向量知识库，以最终返回准确、有帮助的回答。"
        # 让Agent处理请求，会自动判断是否需要检索
        self.agent.print_response(full_prompt, stream=True)



