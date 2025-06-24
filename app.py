import os
import tempfile
from typing import List
import requests
import nest_asyncio
from weasyprint import HTML
import markdown2
import requests
import streamlit as st
from agno.document.reader.csv_reader import CSVReader
from agno.document.reader.pdf_reader import PDFReader
from agno.document.reader.text_reader import TextReader
from agno.document.reader.website_reader import WebsiteReader
from agno.utils.log import logger
from app_utils import (
    CUSTOM_CSS,
    about_widget,
    add_message,
    display_tool_calls,
    export_chat_history,
    rename_session_widget,
    session_selector_widget,
)
from models.agent import get_agent
nest_asyncio.apply()
st.set_page_config(
    page_title="MedRAG",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Add custom CSS

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def restart_agent():
    """Reset the agent and clear chat history"""
    logger.debug("---*--- Restarting Agent ---*---")
    # 只有当前会话是最新会话时才保存
    expected_name = f"会话{len(st.session_state['chat_sessions'])+1}"
    if (
        st.session_state.get("messages")
        and len(st.session_state["messages"]) > 0
        and st.session_state.get("current_session_name") not in st.session_state["chat_sessions"]
    ):
        st.session_state["chat_sessions"][st.session_state.get("current_session_name", expected_name)] = st.session_state["messages"].copy()
    # 清空
    st.session_state["agentic_rag_agent"] = None
    st.session_state["agentic_rag_agent_session_id"] = None
    st.session_state["messages"] = []
    st.session_state["current_session_name"] = f"会话{len(st.session_state['chat_sessions'])+1}"
    st.rerun()


def get_reader(file_type: str):
    """Return appropriate reader based on file type."""
    readers = {
        "pdf": PDFReader(),
        "csv": CSVReader(),
        "txt": TextReader(),
    }
    return readers.get(file_type.lower(), None)

# 生成病例并下載
#user_id = st.session_state.get("user_id", "default_user")
#session_id = st.session_state.get("agentic_rag_agent_session_id")

payload = {
    "user_id": "user_001",
    "session_id": "session_001"  # 替换成有效 ID
}

def download_case():
    base_url = "http://localhost:8000"

    try:
        # 向后端请求 PDF 数据
        response = requests.post(f"{base_url}/generate_case_pdf", json=payload)

        # 成功返回 PDF 内容（字节流）
        if response.status_code == 200:
            return response.content  # 返回 PDF 的二进制内容
        else:
            st.error(f"请求病例PDF失败: {response.status_code} {response.text}")
            return None
    except Exception as e:
        st.error(f"下载病例报告出错: {e}")
        return None


def main():
    ####################################################################
    # App header
    ####################################################################
    st.markdown("<h1 class='main-title'>MedRAG </h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>你的智能医疗AI问答助手～</p>",
        unsafe_allow_html=True,
    )

    ####################################################################
    # Model selector
    ####################################################################
    model_options = {
        "qwen2.5-14b": "qwen2.5:14b-instruct-fp16",
        # "qwen2.5-0.5b": "qwen2.5:0.5b",
    }
    selected_model = st.sidebar.selectbox(
        "请选择你需要的模型",
        options=list(model_options.keys()),
        index=0,
        key="model_selector",
    )
    model_id = model_options[selected_model]

    ####################################################################
    # Initialize Agent
    ####################################################################
    # agentic_rag_agent: Agent
    if (
        "agentic_rag_agent" not in st.session_state
        or st.session_state["agentic_rag_agent"] is None
        or st.session_state.get("current_model") != model_id
    ):
        logger.info("---*--- Creating new Agentic RAG  ---*---")
        agentic_rag_agent = get_agent()
        st.session_state["agentic_rag_agent"] = agentic_rag_agent
        st.session_state["current_model"] = model_id
    else:
        agentic_rag_agent = st.session_state["agentic_rag_agent"]

    if "chat_sessions" not in st.session_state:
        st.session_state["chat_sessions"] = {}
    if "current_session_name" not in st.session_state:
        st.session_state["current_session_name"] = f"会话{len(st.session_state['chat_sessions'])+1}"

    ####################################################################
    # Load Agent Session from the database
    ####################################################################
    # Check if session ID is already in session state
    session_id_exists = (
        "agentic_rag_agent_session_id" in st.session_state
        and st.session_state["agentic_rag_agent_session_id"]
    )

    if not session_id_exists:
        try:
            st.session_state["agentic_rag_agent_session_id"] = (
                agentic_rag_agent.load_session()
            )
        except Exception as e:
            logger.error(f"Session load error: {str(e)}")
            st.warning("无法创建会话，请检查数据库是否运行！")
            # Continue anyway instead of returning, to avoid breaking session switching
    elif (
        st.session_state["agentic_rag_agent_session_id"]
        and hasattr(agentic_rag_agent, "memory")
        and agentic_rag_agent.memory is not None
        and not agentic_rag_agent.memory.runs
    ):
        # If we have a session ID but no runs, try to load the session explicitly
        try:
            agentic_rag_agent.load_session(
                st.session_state["agentic_rag_agent_session_id"]
            )
        except Exception as e:
            logger.error(f"Failed to load existing session: {str(e)}")
            # Continue anyway

    ####################################################################
    # Load runs from memory
    ####################################################################
    agent_runs = []
    if hasattr(agentic_rag_agent, "memory") and agentic_rag_agent.memory is not None:
        agent_runs = agentic_rag_agent.memory.runs

    # Initialize messages if it doesn't exist yet
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Only populate messages from agent runs if we haven't already
    if len(st.session_state["messages"]) == 0 and len(agent_runs) > 0:
        logger.debug("Loading run history")
        for _run in agent_runs:
            # Check if _run is an object with message attribute
            if hasattr(_run, "message") and _run.message is not None:
                add_message(_run.message.role, _run.message.content)
            # Check if _run is an object with response attribute
            if hasattr(_run, "response") and _run.response is not None:
                add_message("assistant", _run.response.content, _run.response.tools)
    elif len(agent_runs) == 0 and len(st.session_state["messages"]) == 0:
        logger.debug("No run history found")

    if prompt := st.chat_input("👋 请尽情向我提问任何关于医疗的问题!"):
        add_message("user", prompt)

    ####################################################################
    # Track loaded URLs and files in session state
    ####################################################################
    if "loaded_urls" not in st.session_state:
        st.session_state.loaded_urls = set()
    if "loaded_files" not in st.session_state:
        st.session_state.loaded_files = set()
    if "knowledge_base_initialized" not in st.session_state:
        st.session_state.knowledge_base_initialized = False

    # st.sidebar.markdown("#### 📚 文档管理")
    # input_url = st.sidebar.text_input("向知识库添加URL")
    # if (
    #     input_url and not prompt and not st.session_state.knowledge_base_initialized
    # ):  # Only load if KB not initialized
    #     if input_url not in st.session_state.loaded_urls:
    #         alert = st.sidebar.info("URL 处理中...", icon="ℹ️")
    #         if input_url.lower().endswith(".pdf"):
    #             try:
    #                 # Download PDF to temporary file
    #                 response = requests.get(input_url, stream=True, verify=False)
    #                 response.raise_for_status()

    #                 with tempfile.NamedTemporaryFile(
    #                     suffix=".pdf", delete=False
    #                 ) as tmp_file:
    #                     for chunk in response.iter_content(chunk_size=8192):
    #                         tmp_file.write(chunk)
    #                     tmp_path = tmp_file.name

    #                 reader = PDFReader()
    #                 docs: List[Document] = reader.read(tmp_path)

    #                 # Clean up temporary file
    #                 os.unlink(tmp_path)
    #             except Exception as e:
    #                 st.sidebar.error(f"PDF处理失败: {str(e)}")
    #                 docs = []
    #         else:
    #             scraper = WebsiteReader(max_links=2, max_depth=1)
    #             docs: List[Document] = scraper.read(input_url)

    #         if docs:
    #             agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
    #             st.session_state.loaded_urls.add(input_url)
    #             st.sidebar.success("URL添加成功")
    #         else:
    #             st.sidebar.error("URL处理失败")
    #         alert.empty()
    #     else:
    #         st.sidebar.info("URL已存在")

    # uploaded_file = st.sidebar.file_uploader(
    #     "上传文件 (.pdf, .csv, or .txt)", key="file_upload"
    # )
    # if (
    #     uploaded_file and not prompt and not st.session_state.knowledge_base_initialized
    # ):  # Only load if KB not initialized
    #     file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
    #     if file_identifier not in st.session_state.loaded_files:
    #         alert = st.sidebar.info("文件处理中...", icon="ℹ️")
    #         file_type = uploaded_file.name.split(".")[-1].lower()
    #         reader = get_reader(file_type)
    #         if reader:
    #             docs = reader.read(uploaded_file)
    #             agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
    #             st.session_state.loaded_files.add(file_identifier)
    #             st.sidebar.success(f"{uploaded_file.name} 添加成功")
    #             st.session_state.knowledge_base_initialized = True
    #         alert.empty()
    #     else:
    #         st.sidebar.info(f"{uploaded_file.name} 已存在")

    # if st.sidebar.button("清空知识库"):
    #     agentic_rag_agent.knowledge.vector_db.delete()
    #     st.session_state.loaded_urls.clear()
    #     st.session_state.loaded_files.clear()
    #     st.session_state.knowledge_base_initialized = False  # Reset initialization flag
    #     st.sidebar.success("知识库已清空")
    ###############################################################
    # Sample Question
    ###############################################################
    st.sidebar.markdown("#### ❓ 提问示例")
    if st.sidebar.button("📝 总结"):
        add_message(
            "user",
            "请总结当前会话的内容",
            # "Can you summarize what is currently in the knowledge base (use `search_knowledge_base` tool)?",
        )

    ###############################################################
    # Utility buttons
    ###############################################################
    st.sidebar.markdown("#### 🛠️ 功能")
    #col1, col2, col3 = st.sidebar.columns([1, 1, 1])  # Equal width columns

    if st.sidebar.button(
        "🔄 新聊天", use_container_width=True
    ):  # Added use_container_width
        restart_agent()
    if st.sidebar.download_button(
        "💾 导出聊天",
        export_chat_history(),
        file_name="rag_chat_history.md",
        mime="text/markdown",
        use_container_width=True,  # Added use_container_width
    ):
        st.sidebar.success("聊天记录已导出!")

    if "pdf_ready" not in st.session_state:
        st.session_state["pdf_ready"] = False
        st.session_state["pdf_bytes"] = None

    if st.sidebar.button("📃 生成病例报告", use_container_width=True):
        with st.spinner("正在生成病例报告..."):
            try:
                pdf_bytes = download_case()
                st.session_state["pdf_bytes"] = pdf_bytes
                st.session_state["pdf_ready"] = True
                st.sidebar.success("病例报告已生成！")
            except Exception as e:
                st.session_state["pdf_ready"] = False
                st.sidebar.error(f"生成失败：{e}")

    # 按钮2：下载病例报告（仅在生成成功后显示）
    if st.session_state["pdf_ready"]:
        st.sidebar.download_button(
            label="⬇️ 下载病例报告",
            data=st.session_state["pdf_bytes"],
            file_name="病例报告.pdf",
            mime="application/pdf",
            use_container_width=True
        )


    ####################################################################
    # Display chat history
    ####################################################################
    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            _content = message["content"]
            if _content is not None:
                with st.chat_message(message["role"]):
                    # Display tool calls if they exist in the message
                    if "tool_calls" in message and message["tool_calls"]:
                        display_tool_calls(st.empty(), message["tool_calls"])
                    st.markdown(_content)

    ####################################################################
    # Generate response for user message
    ####################################################################
    last_message = (
        st.session_state["messages"][-1] if st.session_state["messages"] else None
    )
    if last_message and last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            # Create container for tool calls
            tool_calls_container = st.empty()
            resp_container = st.empty()
            with st.spinner("🤔 思考中..."):
                response = ""
                try:
                    # Run the agent and stream the response
                    run_response = agentic_rag_agent.run(question, stream=True)
                    for _resp_chunk in run_response:
                        # Display tool calls if available
                        if hasattr(_resp_chunk, "tool") and _resp_chunk.tool:
                            display_tool_calls(tool_calls_container, [_resp_chunk.tool])

                        # Display response
                        if _resp_chunk.content is not None:
                            response += _resp_chunk.content
                            resp_container.markdown(response)

                    add_message(
                        "assistant", response, agentic_rag_agent.run_response.tools
                    )
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    add_message("assistant", error_message)
                    st.error(error_message)

    ####################################################################
    # Session selector
    ####################################################################
    session_selector_widget(agentic_rag_agent, model_id)
    rename_session_widget(agentic_rag_agent)

    ####################################################################
    # About section
    ####################################################################
    # about_widget()

    ####################################################################
    # History session selector
    ####################################################################
    
    st.sidebar.markdown("#### 💬 历史会话")
    session_names = [i for i in list(st.session_state["chat_sessions"].keys())]
    with st.sidebar.container():
        for idx, name in enumerate(session_names):
            button_label = f"👉 {name}" if name == st.session_state.get("current_session_name") else name
            if st.sidebar.button(button_label, key=f"session_{idx}"):
                # 1. 保存当前会话（如果有内容且不是当前历史会话）
                current_name = st.session_state.get("current_session_name")
                if (
                    st.session_state.get("messages")
                    and len(st.session_state["messages"]) > 0
                    and current_name is not None
                    and current_name not in st.session_state["chat_sessions"]
                ):
                    st.session_state["chat_sessions"][current_name] = st.session_state["messages"].copy()
                # 2. 切换到目标历史会话
                st.session_state["messages"] = st.session_state["chat_sessions"][name].copy()
                st.session_state["current_session_name"] = name
                st.rerun()


if __name__ == "__main__":
    main()