## RAG的优化

**基于对话历史和当前提问的查询重写**

#### 1. 对话历史优化：

将当前查询与存储在数据库中的历史对话分别向量化，然后计算余弦相似度，返回相似度最高的两个历史对话。(ps: 由于后来发现agent可以自动调用历史两条记录，因此此处只对除去最新两条对话做相似度计算查询)

```python
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
```

最后将该函数作为工具嵌入agent，并设置相关Instruction，让其自行使用

#### 2. 查询重写：

*原先：*

*专门定义了另外一个agent专门负责实现查询重写*

```python
# 重写用户查询的agent
def rewrite_agent(history_queries, model_id: str = DEFAULT_MODEL):
    prompt = dedent(f"""\
        你是一个智能助手，用户会向你提出医学相关的问题。
        然而用户当前的问题可能不够清晰或具体，其中可能包含对历史查询记录的引用，也可能包含一些模糊的描述，或者不专业的用语。
        以下是与当前问题相关的历史查询记录和答案：\n{history_queries}
        历史数据示例：
        ""
        Q: 什么是RAG模型?
        A: RAG（Retrieval-Augmented Generation）是一种将检索机制与生成模型结合的方法，常用于增强生成的回答质量。
        Q: 它模型如何与LLM结合?
        A: RAG模型通过结合大规模语言模型（LLM）来生成基于检索内容的更准确的答案。
        Q: 有哪些常见的RAG应用?
        A: RAG模型常用于问答系统、推荐系统以及对话系统等领域，能够提高信息的准确性和生成的质量。
        Q: 如何优化RAG模型的性能?
        A: RAG模型的优化可以通过改进检索机制、精细化prompt设计以及增加历史查询信息来提升生成结果的相关性和质量。
        ""
        接下来会输入用户的当前问题，请根据这些信息改写用户的问题，使其更清晰、更专业，并且能够更好地反映用户的真实意图。
        如果你认为用户当前的问题涉及很多医学领域的专业知识无法凭借自身内部知识直接改写，需要检索类似上述的医学知识，那么使用retrieve_medical工具，例如：retrieve_medical(query)，否则无需检索直接回答\
    """)
    agent = Agent(
        model=Ollama(id=model_id),
        instructions=prompt,
        tools=[ReasoningTools(add_instructions=True), retrieve_medical],
        show_tool_calls=True,
        markdown=True,
    )
    return agent
```

*后续：*

*发现基于agno框架的agent自带查询重写功能，考虑到提升查询效率，因此将这块内容舍去*

