from typing import Any, Dict, List, Optional

import streamlit as st
# from agentic_rag import get_agentic_rag_agent
from agno.agent import Agent
from agno.models.response import ToolExecution
from agno.utils.log import logger
import sqlite3
from utils.case_db import CaseStorage

DB_PATH = "history/case.db"

def add_message(
    role: str, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Safely add a message to the session state"""
    if "messages" not in st.session_state or not isinstance(
        st.session_state["messages"], list
    ):
        st.session_state["messages"] = []
    st.session_state["messages"].append(
        {"role": role, "content": content, "tool_calls": tool_calls}
    )


def export_chat_history():
    """Export chat history as markdown"""
    if "messages" in st.session_state:
        chat_text = "# Auto RAG Agent - Chat History\n\n"
        for msg in st.session_state["messages"]:
            role = "🤖 Assistant" if msg["role"] == "agent" else "👤 User"
            chat_text += f"### {role}\n{msg['content']}\n\n"
            if msg.get("tool_calls"):
                chat_text += "#### Tools Used:\n"
                for tool in msg["tool_calls"]:
                    if isinstance(tool, dict):
                        tool_name = tool.get("name", "Unknown Tool")
                    else:
                        tool_name = getattr(tool, "name", "Unknown Tool")
                    chat_text += f"- {tool_name}\n"
        return chat_text
    return ""


def display_tool_calls(tool_calls_container, tools: List[ToolExecution]):
    """Display tool calls in a streamlit container with expandable sections.

    Args:
        tool_calls_container: Streamlit container to display the tool calls
        tools: List of tool call dictionaries containing name, args, content, and metrics
    """
    if not tools:
        return

    with tool_calls_container.container():
        for tool_call in tools:
            # Handle different tool call formats
            _tool_name = tool_call.tool_name or "Unknown Tool"
            _tool_args = tool_call.tool_args or {}
            _content = tool_call.result or ""
            _metrics = tool_call.metrics or {}

            # Safely create the title with a default if tool name is None
            title = f"🛠️ {_tool_name.replace('_', ' ').title() if _tool_name else 'Tool Call'}"

            with st.expander(title, expanded=False):
                if isinstance(_tool_args, dict) and "query" in _tool_args:
                    st.code(_tool_args["query"], language="sql")
                # Handle string arguments
                elif isinstance(_tool_args, str) and _tool_args:
                    try:
                        # Try to parse as JSON
                        import json

                        args_dict = json.loads(_tool_args)
                        st.markdown("**Arguments:**")
                        st.json(args_dict)
                    except:
                        # If not valid JSON, display as string
                        st.markdown("**Arguments:**")
                        st.markdown(f"```\n{_tool_args}\n```")
                # Handle dict arguments
                elif _tool_args and _tool_args != {"query": None}:
                    st.markdown("**Arguments:**")
                    st.json(_tool_args)

                if _content:
                    st.markdown("**Results:**")
                    if isinstance(_content, (dict, list)):
                        st.json(_content)
                    else:
                        try:
                            st.json(_content)
                        except Exception:
                            st.markdown(_content)

                if _metrics:
                    st.markdown("**Metrics:**")
                    st.json(
                        _metrics if isinstance(_metrics, dict) else _metrics._to_dict()
                    )


def rename_session_widget(agent: Agent) -> None:
    """Rename the current session of the agent and save to storage"""

    container = st.sidebar.container()

    # Initialize session_edit_mode if needed
    if "session_edit_mode" not in st.session_state:
        st.session_state.session_edit_mode = False

    # 只有 session_id 存在时才允许重命名
    can_rename = hasattr(agent, "session_id") and agent.session_id

    if st.sidebar.button("✎ 重命名当前会话", disabled=not can_rename):
        st.session_state.session_edit_mode = True
        st.rerun()

    if st.session_state.session_edit_mode and can_rename:
        new_session_name = st.sidebar.text_input(
            "Enter new name:",
            value=agent.session_name,
            key="session_name_input",
        )
        if st.sidebar.button("Save", type="primary"):
            if new_session_name:
                # --- 新增：同步数据库 session_id ---
                old_session_name = st.session_state.get("current_session_name")
                case_db = CaseStorage()
                case_db.update_session_id(old_session_name, new_session_name)
                agent.rename_session(new_session_name)
                # 同步Streamlit会话名
                st.session_state["current_session_name"] = new_session_name
                # 同步历史会话key
                if (
                    "chat_sessions" in st.session_state
                    and old_session_name in st.session_state["chat_sessions"]
                ):
                    st.session_state["chat_sessions"][new_session_name] = st.session_state["chat_sessions"].pop(old_session_name)
                st.session_state.session_edit_mode = False
                st.rerun()
    elif st.session_state.session_edit_mode and not can_rename:
        st.sidebar.warning("当前会话未初始化，无法重命名。")


def get_all_sessions(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, user_id, role, message, created_at FROM case_records ORDER BY created_at")
    rows = cursor.fetchall()
    conn.close()
    # 按 session_id 分组
    sessions = {}
    for session_id, user_id, role, message, created_at in rows:
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append({
            "role": role,
            "content": message,
            "user_id": user_id,
            "created_at": created_at
        })
    return sessions



CUSTOM_CSS = """
    <style>
    /* Main Styles */
   .main-title {
        text-align: center;
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
        padding: 1em 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2em;
    }
    .stButton button {
        width: 100%;
        border-radius: 20px;
        margin: 0.2em 0;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .chat-container {
        border-radius: 15px;
        padding: 1em;
        margin: 1em 0;
        background-color: #f5f5f5;
    }
    .tool-result {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1em;
        margin: 1em 0;
        border-left: 4px solid #3B82F6;
    }
    .status-message {
        padding: 1em;
        border-radius: 10px;
        margin: 1em 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
    }
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .chat-container {
            background-color: #2b2b2b;
        }
        .tool-result {
            background-color: #1e1e1e;
        }
    }
    </style>
"""

def insert_messages(DB_PATH, session_id, user_id, messages):
    """批量插入一组消息到数据库"""
    if not messages or len(messages) == 0:
        return  # 空会话不写入数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for msg in messages:
        cursor.execute(
            "INSERT INTO case_records (session_id, user_id, role, message) VALUES (?, ?, ?, ?)",
            (session_id, user_id, msg["role"], msg["content"])
        )
    conn.commit()
    conn.close()

def restart_agent():
    """Reset the agent and clear chat history"""
    logger.debug("---*--- Restarting Agent ---*---")
    # 只有当前会话是最新会话时才保存
    if (
        st.session_state.get("messages")
        and len(st.session_state["messages"]) > 0
        and st.session_state.get("current_session_name") not in st.session_state["chat_sessions"]
    ):
        st.session_state["chat_sessions"][st.session_state.get("current_session_name")] = st.session_state["messages"].copy()
        # === 新增：写入数据库 ===
        user_id = st.session_state.get("user_id", "default_user")
        # 生成 session_id，格式为 session_00数字
        all_sessions = get_all_sessions(DB_PATH)
        session_count = len(all_sessions)
        session_id = f"session_{session_count+1:03d}"
        insert_messages(DB_PATH, session_id, user_id, st.session_state["messages"])
    # 清空
    st.session_state["agentic_rag_agent"] = None
    st.session_state["agentic_rag_agent_session_id"] = None
    st.session_state["messages"] = []
    # 新会话名也用 session_00数字
    all_sessions = get_all_sessions(DB_PATH)
    session_count = len(all_sessions)
    st.session_state["current_session_name"] = f"session_{session_count+1:03d}"
    # 关键：从数据库重新加载 chat_sessions
    st.session_state["chat_sessions"] = all_sessions
    st.rerun()

def get_answer(resp):
    """提取 Agent 回复中最可能是"人话"的那条"""
    if hasattr(resp, "messages"):
        for msg in reversed(resp.messages):
            if msg.role == "assistant" and isinstance(msg.content, str):
                # 简单规则：过滤明显不是自然语言的 JSON / ToolCall 输出
                if msg.content.strip().startswith("{") and msg.content.strip().endswith("}"):
                    continue
                if msg.content.strip().startswith("<") and msg.content.strip().endswith(">"):
                    continue
                if len(msg.content.strip()) < 5:
                    continue
                return msg.content.strip()
    # 兜底
    if hasattr(resp, "content") and isinstance(resp.content, str):
        return resp.content.strip()
    return "[无有效回答]"


def save_message_to_db(session_id, user_id, role, content):
    """保存单条消息到 case.db"""
    case_db = CaseStorage()
    case_db.save_message(session_id, user_id, role, content)