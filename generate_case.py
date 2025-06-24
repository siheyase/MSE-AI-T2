import os
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from utils.case_db import CaseStorage
from models.agent import get_agent
from test_session import is_valid_message
from markdown2 import markdown
from weasyprint import HTML
from datetime import datetime

app = FastAPI()
case_db = CaseStorage()

class GenerateCaseRequest(BaseModel):
    user_id: str
    session_id: str

@app.post("/generate_case_text")
def generate_case_summary(req: GenerateCaseRequest):
    print(req.user_id, req.session_id)
    context = case_db.generate_case(req.user_id, req.session_id)

    if not context:
        return JSONResponse(status_code=404, content={"error": "No messages found."})

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = f"""你是一名医学助手。以下是患者与AI助手的问诊对话记录。请根据对话内容，尽可能提取出患者的基本信息（如姓名、年龄、性别），如果无法确定，请保留空项。然后生成完整的病例摘要，格式为markdown，并严格按照如下结构输出：

## 病例报告

### 【基本信息】
- 姓名：<提取的姓名或空>
- 性别：<提取的性别或空>
- 年龄：<提取的年龄或空>

### 【主诉】
...

### 【现病史】
...

### 【既往史】
...

### 【体格检查】
...

### 【辅助检查】
...

### 【诊断与建议】
...

生成时间：{current_time}

本病例报告由人工智能助手生成，结果仅供参考，不构成医疗建议。

以下是对话内容（供参考）：
{context}
"""

    agent = get_agent()
    response = agent.run(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return {"markdown": content}




@app.post("/generate_case_pdf")
def generate_case_pdf(req: GenerateCaseRequest):
    result = generate_case_summary(req)
    if isinstance(result, JSONResponse) and result.status_code != 200:
        return result

    markdown_text = result["markdown"]
    # ✅ 删除解释性开头，只保留从 "## 病例报告" 开始的内容
    if "## 病例报告" in markdown_text:
        markdown_text = markdown_text.split("## 病例报告", 1)[1]
        markdown_text = "## 病例报告" + markdown_text  # 保留标题

    html_text = markdown(markdown_text)

    # 生成临时 PDF 文件
    os.makedirs("generated_cases", exist_ok=True)
    pdf_path = f"generated_cases/case_{req.user_id}_{req.session_id}.pdf"
    HTML(string=html_text).write_pdf(pdf_path)

    return FileResponse(path=pdf_path, filename=os.path.basename(pdf_path), media_type="application/pdf")
