# MSE-AI-T2
python环境：python 3.12  
Ollama部署的模型如下：  
Embedding model:bge-m3  
Agent model: qwen2.5:14b-instruct-fp16  

```
├── main.py 暂时的程序运行代码 
├── models 
│   ├── agent.py agent搭建，智能体封装逻辑 
├── requirements.txt
└── vs 向量数据库存储位置
    ├── index.faiss
    └── index.pkl
├── app.py 暂时为空，应用主入口
├── build_index.py 向量数据库创建及向量化
├── check_index.py 用于测试向量数据库检索和获取当前分块个数
├── config
│   ├── settings.py 模型、参数等配置文件
├── doc
│   ├── medical.json 知识库文档
└── docs 存储向量库对应分块的父文档
```
把docs、vs的压缩包下的内容解压存放到对应文件夹下。
```
pip install -r requirements.txt
```
运行以进行测试
```
python main.py
```
前端运行命令  
```
streamlit run app.py
```
病例生成接口运行命令  
```
uvicorn generate_case:app --reload --host 0.0.0.0 --port 8000
```
病例生成接口测试  
```
python test_generate_case.py
```

---

# 多轮对话存储功能说明

本项目支持用户 **多轮对话上下文存储**，可实现跨 session 的上下文调用与病例摘要生成。

# 功能简介

* 每次用户与助手对话的内容（包括用户问题与助手回答）将自动记录至 SQLite 数据库。
* 支持跨 session 合并历史记录生成完整病例摘要。
* 支持通过 `user_id` 和 `session_id` 唯一标识一段会话。

# 数据库路径

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


