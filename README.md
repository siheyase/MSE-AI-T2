# MSE-AI-T2
python环境：python 3.12  
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

