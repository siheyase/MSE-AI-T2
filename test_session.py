import os, sys
import time
from models.agent import get_agent
from utils.case_db import CaseStorage

case_db = CaseStorage()


def test_session_and_save():
    user_id = "user_001"

    # === 第一次会话 ===
    session_1 = "session_001"
    agent1 = get_agent(session_id=session_1, user_id=user_id)

    q1 = "什么是肺泡蛋白质沉积症？"
    a1 = get_answer(agent1.run(q1))
    case_db.save_message(session_1, user_id, "user", q1)
    case_db.save_message(session_1, user_id, "assistant", a1)

    q2 = "它的治疗方式有哪些？"
    a2 = get_answer(agent1.run(q2))
    case_db.save_message(session_1, user_id, "user", q2)
    case_db.save_message(session_1, user_id, "assistant", a2)

    # === 第二次会话（跨 session 接着问）===
    session_2 = "session_002"
    agent2 = get_agent(session_id=session_2, user_id=user_id)

    # ✅ 显式补充“我上次说的病是...”
    q3 = "我上次说的肺泡蛋白质沉积症，还有什么并发症？"
    a3 = get_answer(agent2.run(q3))
    case_db.save_message(session_2, user_id, "user", q3)
    case_db.save_message(session_2, user_id, "assistant", a3)



def get_answer(resp):
    """提取 Agent 回复中最可能是“人话”的那条"""
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

def is_valid_message(msg: str) -> bool:
    msg = msg.strip()
    if not msg or len(msg) < 5:
        return False
    if msg.startswith("{") and msg.endswith("}"):
        return False
    if msg.startswith("<") and msg.endswith(">"):
        return False
    if msg.lower() in ["astreet", "index", "</tool_call>"]:
        return False
    return True


def generate_case(user_id, session_id):
    """生成病例摘要（去重，单 session）"""
    messages = case_db.get_messages(user_id, session_id)
    seen = set()
    lines = []
    for role, msg in messages:
        if not is_valid_message(msg):
            continue
        key = (role, msg.strip())
        if key in seen:
            continue
        seen.add(key)
        prefix = "👤用户：" if role == "user" else "🤖助手："
        lines.append(f"{prefix}{msg.strip()}")
    return "\n\n".join(lines)

def generate_case_all_sessions(user_id):
    """汇总该用户所有 session 的问诊记录（去重）"""
    messages = case_db.get_messages(user_id)
    seen = set()
    lines = []
    for role, msg in messages:
        if not is_valid_message(msg):
            continue
        key = (role, msg.strip())
        if key in seen:
            continue
        seen.add(key)
        prefix = "👤用户：" if role == "user" else "🤖助手："
        lines.append(f"{prefix}{msg.strip()}")
    return "\n\n".join(lines)



if __name__ == "__main__":
    test_session_and_save()

    print("\n===== 📝 病例摘要输出 =====\n")
    print(generate_case("user_001", "session_001"))
    print("\n===== 📝 跨session病例摘要输出 =====\n")
    print(generate_case_all_sessions("user_001"))