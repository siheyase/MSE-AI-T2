# utils/case_db.py
import sqlite3
from datetime import datetime

class CaseStorage:
    def __init__(self, db_path="history/case.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS case_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                role TEXT,        -- 'user' or 'assistant'
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save_message(self, session_id, user_id, role, message):
        self.conn.execute(
            "INSERT INTO case_records (session_id, user_id, role, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, role, message, datetime.utcnow())
        )
        self.conn.commit()

    def get_messages(self, user_id, session_id=None):
        cursor = self.conn.cursor()
        if session_id:
            cursor.execute(
                "SELECT role, message FROM case_records WHERE user_id=? AND session_id=? ORDER BY id",
                (user_id, session_id)
            )
        else:
            cursor.execute(
                "SELECT role, message FROM case_records WHERE user_id=? ORDER BY id",
                (user_id,)
            )
        return cursor.fetchall()

    def get_answer(self, resp):
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

    def is_valid_message(self, msg: str) -> bool:
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


    def generate_case(self, user_id, session_id):
        """生成病例摘要（去重，单 session）"""
        messages = self.get_messages(user_id, session_id)
        # debug
        print("test debug: ", messages)
        seen = set()
        lines = []
        for role, msg in messages:
            if not self.is_valid_message(msg):
                continue
            key = (role, msg.strip())
            if key in seen:
                continue
            seen.add(key)
            prefix = "👤用户：" if role == "user" else "🤖助手："
            lines.append(f"{prefix}{msg.strip()}")
        return "\n\n".join(lines)

    def generate_case_all_sessions(self, user_id):
        """汇总该用户所有 session 的问诊记录（去重）"""
        messages = self.get_messages(user_id)
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

    # def generate_case_pdf(user_id: str, session_id: str) -> str:
    #     case_text = self.generate_case(user_id=user_id, session_id=session_id)

    #     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     disclaimer = "\n\n⚠️ 本报告由 AI 辅助生成，仅供参考，不构成医疗建议。\n"

    #     full_md = f"# 🏥 病例报告\n\n🕒 生成时间：{timestamp}\n\n{case_text}{disclaimer}"
    #     html_content = markdown2.markdown(full_md)

    #     pdf_path = "/mnt/data/case_output.pdf"
    #     HTML(string=html_content).write_pdf(pdf_path)

    #     return pdf_path

    def update_session_id(self, old_session_id, new_session_id):
        """批量更新 session_id"""
        self.conn.execute(
            "UPDATE case_records SET session_id=? WHERE session_id=?",
            (new_session_id, old_session_id)
        )
        self.conn.commit()