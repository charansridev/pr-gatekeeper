import os
import requests
import base64
import json
import re
import nest_asyncio
import asyncio
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks
from pyngrok import ngrok
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")
nest_asyncio.apply()

# ── Keys ──────────────────────────────────────────────────
GITHUB_HEADERS = {
    "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
    "Accept": "application/vnd.github+json"
}
REPO = "charansridev/pr-gatekeeper-test"

# ── LLM ──────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ── GitHub Tools ──────────────────────────────────────────


def get_pr_diff(pr_url: str) -> str:
    r = requests.get(pr_url, headers={
                     **GITHUB_HEADERS, "Accept": "application/vnd.github.diff"})
    return r.text[:6000]


def get_file_content(repo_full_name: str, file_path: str) -> str:
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}"
    r = requests.get(url, headers=GITHUB_HEADERS).json()
    if "content" not in r:
        return "File not found"
    return base64.b64decode(r["content"]).decode("utf-8")[:4000]


def post_github_comment(repo_full_name: str, pr_number: int, body: str):
    url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
    r = requests.post(url, headers=GITHUB_HEADERS, json={"body": body})
    return r.status_code


def set_pr_review(repo_full_name: str, pr_number: int, action: str, body: str):
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/reviews"
    r = requests.post(url, headers=GITHUB_HEADERS, json={
                      "event": action, "body": body})
    return r.status_code


# ── Chains ────────────────────────────────────────────────
analyst_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Code Analyst reviewing a GitHub PR diff.
Return ONLY this JSON — no extra text:
{{
  "summary": "one sentence describing what this PR does",
  "files_changed": ["file1.py"],
  "logic_issues": ["issue or empty list"]
}}"""),
    ("human", "PR Diff:\n{diff}")
])
code_analyst_chain = analyst_prompt | llm | StrOutputParser()

risk_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Security Risk Scorer.
Scan for: hardcoded secrets, SQL injection, missing auth, eval/exec usage.
Return ONLY this JSON — no extra text:
{{
  "risk_score": <0-100>,
  "confidence": <0-100>,
  "findings": ["finding1"],
  "action": "APPROVE"
}}
Set action to BLOCK if risk_score>75, REQUEST_CHANGES if 30-75, APPROVE if <30."""),
    ("human", "PR Diff:\n{diff}")
])
risk_scorer_chain = risk_prompt | llm | StrOutputParser()

doc_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Documentation Auditor.
Check if new functions in the diff have README docs.
Return ONLY this JSON — no extra text:
{{
  "docs_missing": true,
  "missing_for": ["function_name"],
  "draft_documentation": "## New Feature\\n\\ndescription here"
}}"""),
    ("human", "PR Diff:\n{diff}\n\nCurrent README:\n{readme}")
])
doc_auditor_chain = doc_prompt | llm | StrOutputParser()

# ── Self Correction ───────────────────────────────────────


def clean_and_parse(raw: str) -> dict:
    cleaned = re.sub(r"```json|```", "", raw).strip()
    return json.loads(cleaned)


def run_with_self_correction(chain, inputs: dict, chain_name: str = "Chain") -> dict:
    raw = ""
    last_error = None
    for attempt in range(3):
        if last_error:
            print(f"  ⚠️  [{chain_name}] Correcting error: {last_error}")
            correction_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Your previous response caused JSON error: {error}. Return valid JSON only."),
                ("human",
                 "Bad output:\n{bad_output}\n\nReturn corrected JSON:")
            ])
            correction_chain = correction_prompt | llm | StrOutputParser()
            raw = correction_chain.invoke(
                {"error": last_error, "bad_output": raw})
        else:
            raw = chain.invoke(inputs)
        try:
            result = clean_and_parse(raw)
            if attempt > 0:
                print(f"  ✅ [{chain_name}] Self-corrected!")
            return result
        except json.JSONDecodeError as e:
            last_error = str(e)
    print(f"  ❌ [{chain_name}] Failed — using fallback")
    return {"error": "parse_failed", "action": "REQUEST_CHANGES"}

# ── Comment Builder ───────────────────────────────────────


def build_github_comment(result: dict) -> str:
    emoji = {"APPROVE": "✅", "REQUEST_CHANGES": "⚠️",
             "BLOCK": "🚨"}.get(result["action"], "⚠️")
    comment = f"""## {emoji} PR Gatekeeper Report

**Decision:** `{result['action']}`
**Risk Score:** {result.get('risk_score', 0)}/100
**Confidence:** {result.get('confidence', 0)}%

### 📋 Code Summary
{result.get('code_summary', 'N/A')}
"""
    if result.get("findings"):
        comment += "\n### 🔒 Security Findings\n"
        for f in result["findings"]:
            comment += f"- {f}\n"
    if result.get("logic_issues"):
        comment += "\n### 🐛 Logic Issues\n"
        for i in result["logic_issues"]:
            comment += f"- {i}\n"
    if result.get("docs_missing"):
        comment += f"\n### 📝 Documentation Gap\nHere is a draft:\n\n{result.get('draft_documentation', '')}\n"
    comment += "\n---\n*Posted by PR Gatekeeper 🤖 — Powered by LangChain + Groq*"
    return comment

# ── Manager ───────────────────────────────────────────────


def run_manager(diff: str, readme: str, repo: str, pr_number: int):
    print("\n" + "="*50)
    print("🤖 PR GATEKEEPER STARTING")
    print("="*50)

    print("\n[Manager] → Code Analyst...")
    analysis = run_with_self_correction(
        code_analyst_chain, {"diff": diff}, "Code Analyst")
    print(f"[Code Analyst] ✅ {analysis.get('summary', '')[:60]}")

    print("\n[Manager] → Risk Scorer...")
    risk = run_with_self_correction(
        risk_scorer_chain, {"diff": diff}, "Risk Scorer")
    print(f"[Risk Scorer] ✅ Score: {risk.get('risk_score', 0)}/100")

    print("\n[Manager] → Doc Auditor...")
    docs = run_with_self_correction(
        doc_auditor_chain, {"diff": diff, "readme": readme}, "Doc Auditor")
    print(f"[Doc Auditor] ✅ Docs missing: {docs.get('docs_missing', False)}")

    action = risk.get("action", "REQUEST_CHANGES")
    if risk.get("risk_score", 0) > 80:
        action = "BLOCK"

    result = {
        "action": action,
        "risk_score": risk.get("risk_score", 0),
        "confidence": risk.get("confidence", 0),
        "findings": risk.get("findings", []),
        "code_summary": analysis.get("summary", ""),
        "logic_issues": analysis.get("logic_issues", []),
        "docs_missing": docs.get("docs_missing", False),
        "draft_documentation": docs.get("draft_documentation", ""),
    }

    print(
        f"\n[Manager] ⚖️  Decision: {action} | Risk: {result['risk_score']}/100")

    comment = build_github_comment(result)
    status = post_github_comment(repo, pr_number, comment)
    print(f"[Manager] GitHub comment posted — HTTP {status}")

    review_event = "APPROVE" if action == "APPROVE" else "REQUEST_CHANGES"
    set_pr_review(repo, pr_number, review_event, f"PR Gatekeeper: {action}")

    print("\n" + "="*50)
    print("✅ PR GATEKEEPER DONE")
    print("="*50)
    return result


# ── FastAPI Server ────────────────────────────────────────
app = FastAPI()


@app.get("/")
def health():
    return {"status": "PR Gatekeeper is running ✅"}


@app.post("/webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    action = payload.get("action", "No action found (might be a ping!)")

    # Let's print exactly what GitHub sends us!
    print(f"\n[Webhook Debug] Received event action: {action}")

    if action not in ["opened", "synchronize"]:
        print(f"[Webhook] Ignored event: {action}")
        return {"status": "ignored"}

    pr = payload.get("pull_request")
    if not pr:
        print("[Webhook] Error: No pull request data found in payload.")
        return {"status": "error"}

    pr_number = pr["number"]
    pr_url = pr["url"]
    print(f"\n📥 New PR #{pr_number} received!")

    diff = get_pr_diff(pr_url)
    readme = get_file_content(REPO, "README.md")
    background_tasks.add_task(run_manager, diff, readme, REPO, pr_number)
    return {"status": "processing", "pr": pr_number}

# ── Start ─────────────────────────────────────────────────
if __name__ == "__main__":
    ngrok.set_auth_token(os.getenv("NGROK_TOKEN"))
    ngrok.kill()
    public_url = ngrok.connect(8000)
    print("="*55)
    print("🌐 YOUR WEBHOOK URL:")
    print(f"{public_url.public_url}/webhook")
    print("="*55)

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
