# 基于RAG技术的医疗AI问答助手

本项目基于RAG技术与LLM模型，构建了一个面向患者与医务人员的医疗AI问答助手，聚焦提供权威、可信、易用的健康信息服务，助力健康科普与基层医疗支持。提供智能问答、引导式多轮交互、病例报告生成下载等功能。

访问地址：http://172.23.166.108:8501/

---
 
## 核心目录结构

```
        ├── config
        |   └── settings.py 模型、参数等配置文件
        ├── docs 存储向量库对应分块的父文档
        ├── models
        |   └── agent.py agent搭建，智能体封装逻辑
        ├── record_docs 功能开发文档
        ├── utils
        |   └── case_db.py 数据库存储模块
        ├── vs 向量数据库存储位置
            ├── index.faiss
            └── index.pkl
        ├── README.md
        ├── product.md 产品文档
        ├── team.md 团队记录文档
        ├── app.py 应用主入口
        ├── app_utils.py 
        ├── build_index.py 向量数据库创建及向量化
        ├── check_index.py 用于测试向量数据库检索和获取当前分块个数
        ├── generate_case.py 病例生成模块
        ├── main.py
        └── requirements.txt
```
## 环境准备

python环境：python 3.12  
Ollama部署的模型如下：  
- Embedding model:bge-m3  
- Agent model: qwen2.5:14b-instruct-fp16 

把docs、vs的压缩包下的内容解压存放到对应文件夹下。

```
pip install -r requirements.txt
```
运行以进行测试
```
python main.py
```

## 运行指南
1. 运行主应用  

    ```
    streamlit run app.py
    ```
2. 运行病例生成接口  

    ```
    uvicorn generate_case:app --reload --host 0.0.0.0 --port 8000
    ```

---

## 多轮对话存储功能说明

本项目支持用户 **多轮对话上下文存储**，可实现跨 session 的上下文调用与病例摘要生成。

### 功能简介

* 每次用户与助手对话的内容（包括用户问题与助手回答）将自动记录至 SQLite 数据库。
* 支持跨 session 合并历史记录生成完整病例摘要。
* 支持通过 `user_id` 和 `session_id` 唯一标识一段会话。

### 数据库路径

对话数据存储在以下路径：

```
history/session.db
```

使用的表名为：

```
agent_sessions
```

---

## 病例报告生成功能说明

本项目内置 **病例报告自动生成模块**，可将用户与助手的完整对话转换为结构化的医学病例文本与 PDF 文件，供下载与归档。

### 功能简介

* **Markdown / PDF 双格式输出**

  * `POST /generate_case_text` 返回 Markdown 字符串
  * `POST /generate_case_pdf` 返回可下载的 PDF 报告


### 核心文件

| 文件                 | 作用                                 |
| ------------------ | ---------------------------------- |
| `generate_case.py` | FastAPI 路由：Markdown & PDF 生成、文件流下载 |
| `generated_cases/` | 生成的 PDF 临时保存目录            |

### 主要接口

| 路径                    | 方法   | 传参                         | 返回                          |
| --------------------- | ---- | -------------------------- | --------------------------- |
| `/generate_case_text` | POST | `{"user_id","session_id"}` | `{ "markdown": "..." }`     |
| `/generate_case_pdf`  | POST | 同上                         | PDF 文件流 (`application/pdf`) |


---
