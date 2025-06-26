# 多轮对话历史存储开发记录

## 项目背景与需求

在本项目中，我主要负责多轮对话历史的后端存储模块，利用 SQLite 数据库（`history/case.db`）持久化保存每条消息，并在 `agent.py` 中集成存储逻辑，确保对话交互数据可靠持久。模块需求如下：

* **会话管理**：生成、加载、重命名和删除 session。
* **消息存储**：在用户或助手发送消息时，自动调用存储接口将对话写入数据库；支持单条及批量读写。

---

## 多轮对话存储功能说明

本项目支持用户 **多轮对话上下文存储**，可实现跨 session 的上下文调用与病例摘要生成。

## 功能简介

* 每次用户与助手对话的内容（包括用户问题与助手回答）将自动记录至 SQLite 数据库。
* 支持跨 session 合并历史记录生成完整病例摘要。
* 支持通过 `user_id` 和 `session_id` 唯一标识一段会话。

## 数据库路径

对话数据存储在以下路径：

```
history/session.db
```

使用的表名为：

```
agent_sessions
```

### 如何使用（代码中示例）

```python
from models.agent import get_agent
from utils.case_db import generate_case_all_sessions

# 初始化 Agent，并指定 user_id 和 session_id
agent = get_agent(session_id="session001", user_id="user001")

# 运行一次对话
response = agent.run("什么是病毒性心肌炎？")
print(response)

# 在任意时刻生成该用户的所有病例摘要（跨 session）
summary = generate_case_all_sessions("user001")
print("生成的病例摘要：")
print(summary)
```

可以参考测试脚本 `test_session.py`，其中演示了如何进行多轮对话存储与病例生成。
如需测试，请运行：

```bash
python test_session.py
```

---

## 关键代码模块及文件位置

### 1. `utils/case_db.py`

#### 1.1 数据库连接与表结构（第1–20行）

```python
# utils/case_db.py:1
import sqlite3
class CaseStorage:
    def __init__(self, db_path: str = "history/case.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute(r"""
        CREATE TABLE IF NOT EXISTS case_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id TEXT,
            role TEXT,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """")
        self.conn.commit()
```

#### 1.2 `save_message` 单条插入（第20–25行）

```python
# utils/case_db.py:22
    def save_message(self, session_id: str, user_id: str, role: str, message: str) -> None:
        sql = "INSERT INTO case_records (session_id, user_id, role, message) VALUES (?, ?, ?, ?)"
        self.cursor.execute(sql, (session_id, user_id, role, message))
        self.conn.commit()
```

#### 1.3 `get_messages` 查询历史（第27–39行）

```python
# utils/case_db.py:36
    def get_messages(self, user_id: str, session_id: Optional[str] = None) -> List[Tuple[str, str]]:
        if session_id:
            sql = "SELECT role, message FROM case_records WHERE user_id=? AND session_id=? ORDER BY id"
            params = (user_id, session_id)
        else:
            sql = "SELECT role, message FROM case_records WHERE user_id=? ORDER BY id"
            params = (user_id,)
        self.cursor.execute(sql, params)
        return self.cursor.fetchall()
```


---

### 2. `agent.py` — 存储集成点

#### 2.1 引入存储（第1–10行）

```python
# agent.py:1
from utils.case_db import CaseStorage

```

#### 2.2 在 `get_agent` 中传入 `storage`（第137–145行）
```python
# agent.py:137
def get_agent(model_id: str = DEFAULT_MODEL, session_id=None, user_id=None) -> Agent:
    return Agent(
        name="Medical Assistant",
        model=Ollama(id=model_id),
        session_id=session_id,
        user_id=user_id,
        instructions=dedent(...),
        tools=[...],
        show_tool_calls=True,
        markdown=True,
        num_history_responses=3,
        add_history_to_messages=True,
        storage=storage,
        search_previous_sessions_history=True,
        num_history_sessions=2,
    )
```


---

### 3. `tests/test_session.py` — 会话保存与摘要测试

#### 3.1 文件头与初始化（第1–6行）

```python
# tests/test_session.py:1
import time
from models.agent import get_agent
from utils.case_db import CaseStorage

case_db = CaseStorage()
```

#### 3.2 `test_session_and_save` 函数（第9–90行）

```python
# tests/test_session.py:9
def test_session_and_save():
    user_id = "user_001"

    # 第一次会话
    session_1 = "session_001"
    agent1 = get_agent(session_id=session_1, user_id=user_id)
    q1 = "什么是肺泡蛋白质沉积症？"
    a1 = agent1.chat(q1)

    # 第二次会话
    session_2 = "session_002"
    agent2 = get_agent(session_id=session_2, user_id=user_id)
    q2 = "肺泡蛋白质沉积症有哪些临床表现？"
    a2 = agent2.chat(q2)

    # 验证数据库中会话记录
    msgs1 = case_db.get_messages(user_id, session_1)
    msgs2 = case_db.get_messages(user_id, session_2)
    assert any("肺泡蛋白质沉积症" in m for _, m in msgs1)
    assert any("临床表现" in m for _, m in msgs2)
```

#### 3.3 主流程调用与摘要输出（第104–110行）

```python
# tests/test_session.py:104
if __name__ == "__main__":
    test_session_and_save()
    # 输出单 session 病例摘要
    print("\n===== 📝 病例摘要输出 =====\n")
    print(case_db.generate_case(user_id, session_1))
    # 输出所有 session 病例摘要
    print("\n===== 📝 跨session病例摘要输出 =====\n")
    print(case_db.generate_case_all_sessions(user_id))
```

---

## 与前端同学的协作流程

最初，我和前端同学各自快速迭代：我把对话历史直接存入 SQLite；他先将消息缓存到 `app.py` 的全局变量中用于渲染。联调时，我们发现全局变量模式下刷新页面会丢失历史，也不利于后续分析和导出。于是他主动将前端发送和接收的消息调用同样的存储接口，打通前后端，确保任何界面刷新或新接口调用都能读取到完整对话历史。

---

## 开发过程中遇到的困难
* **SQL 注入风险**：初版用字符串拼接 SQL，遇到特殊字符报错或安全问题；统一改用参数化查询后完全解决。
* **内存上下文污染**：`load_history` 未清空旧内存导致数据混合；改为每次加载前清空 `agent.memory`。

---

## 开发思考与心得

- **职责分明**：数据库读写都在 `utils/case_db.py`，`agent.py` 仅负责流程，降低耦合。
- **一致性优先**：参数化查询和统一存取接口，保证前后端共享同一数据源。
- **先行落地，后续优化**：先用 SQLite 快速跑通端到端，再按需升级数据库或引入缓存。
- **模块化设计**：把存储逻辑抽象为 `CaseStorage` 类和 `SqliteStorage`，后续更换存储方式或添加功能都很方便。
- **团队协作**：与前端同学密切沟通，及时调整接口和数据格式，确保数据流畅打通。

---
