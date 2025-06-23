import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

DEFAULT_MODEL = "qwen2.5:14b-instruct-fp16"
EMBEDDING_MODEL = "bge-m3"

ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  

VS_PATH = os.path.join(ROOT_DIR, "vs")
VS_DOCS_PATH = os.path.join(ROOT_DIR, "docs")

TOP_K = 2
SEARCH_TYPE = "similarity"