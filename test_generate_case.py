import requests

base_url = "http://localhost:8000"

# 测试数据（请替换为实际存在的 user_id 和 session_id）
payload = {
    "user_id": "user_001",
    "session_id": "session_001"  # 替换成有效 ID
}

# 1. 测试 markdown 生成
resp_text = requests.post(f"{base_url}/generate_case_text", json=payload)
print("=== Markdown 响应 ===")
if resp_text.status_code == 200:
    print(resp_text.json()["markdown"])
else:
    print("❌ 请求失败:", resp_text.status_code, resp_text.text)

# 2. 测试 PDF 生成并保存
resp_pdf = requests.post(f"{base_url}/generate_case_pdf", json=payload)
print("\n=== PDF 生成 ===")
if resp_pdf.status_code == 200:
    with open("downloaded_case.pdf", "wb") as f:
        f.write(resp_pdf.content)
    print("✅ PDF 已保存为 downloaded_case.pdf")
else:
    print("❌ PDF 请求失败:", resp_pdf.status_code, resp_pdf.text)
