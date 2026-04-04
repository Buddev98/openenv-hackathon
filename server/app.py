from fastapi import FastAPI, Body, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from customer_support_env.models import (
    Observation, Action, RewardOutput,
    ClassifyAction, ReplyAction, EscalateAction, ArchiveAction
)
from customer_support_env.env import CustomerSupportEnv
from typing import Union
import os

env = CustomerSupportEnv()

app = FastAPI(title="OpenEnv: Customer Support Triage")

# Allow all origins for CORS (needed for HF Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenEnv Explorer — Customer Support Triage</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  *{margin:0;padding:0;box-sizing:border-box;}
  body{font-family:'Inter',sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh;}
  .header{background:linear-gradient(135deg,#161b22,#1f2937);border-bottom:1px solid #30363d;padding:18px 32px;display:flex;align-items:center;gap:12px;}
  .header h1{font-size:20px;font-weight:600;background:linear-gradient(90deg,#58a6ff,#bc8cff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .badge{background:#238636;color:#fff;font-size:11px;padding:2px 8px;border-radius:20px;font-weight:500;}
  .container{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:24px 32px;max-width:1200px;margin:0 auto;}
  .card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;}
  .card h2{font-size:14px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;margin-bottom:16px;}
  label{display:block;font-size:12px;color:#8b949e;margin-bottom:5px;font-weight:500;}
  select,input,textarea{width:100%;background:#0d1117;border:1px solid #30363d;border-radius:8px;color:#e6edf3;padding:9px 12px;font-size:13px;font-family:inherit;outline:none;transition:border-color .2s;}
  select:focus,input:focus,textarea:focus{border-color:#58a6ff;}
  .field{margin-bottom:14px;}
  .radio-group{display:flex;gap:8px;flex-wrap:wrap;}
  .radio-btn{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:7px 14px;font-size:12px;cursor:pointer;transition:all .2s;color:#8b949e;}
  .radio-btn.active{border-color:#58a6ff;color:#58a6ff;background:#1f2d4a;}
  .btn{width:100%;padding:10px;border-radius:8px;border:none;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;font-family:inherit;}
  .btn-primary{background:linear-gradient(135deg,#238636,#2ea043);color:#fff;}
  .btn-primary:hover{opacity:.9;transform:translateY(-1px);}
  .btn-secondary{background:linear-gradient(135deg,#1f6feb,#388bfd);color:#fff;margin-top:10px;}
  .btn-secondary:hover{opacity:.9;transform:translateY(-1px);}
  .btn:disabled{opacity:.5;cursor:not-allowed;transform:none;}
  .metrics{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;}
  .metric{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px;text-align:center;}
  .metric .val{font-size:22px;font-weight:700;color:#58a6ff;}
  .metric .lbl{font-size:11px;color:#8b949e;margin-top:2px;}
  .done-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;}
  .done-yes{background:#238636;color:#fff;}
  .done-no{background:#1f2937;color:#8b949e;}
  pre{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px;font-size:12px;overflow:auto;max-height:360px;line-height:1.6;color:#79c0ff;}
  .status{font-size:11px;color:#8b949e;margin-top:8px;text-align:center;}
  .log{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:10px 14px;font-size:11px;color:#8b949e;max-height:100px;overflow-y:auto;margin-top:12px;font-family:monospace;}
  .log p{margin:2px 0;}
  .log .ok{color:#3fb950;}
  .log .err{color:#f85149;}
  @media(max-width:768px){.container{grid-template-columns:1fr;}}
</style>
</head>
<body>
<div class="header">
  <span style="font-size:24px;">📧</span>
  <h1>OpenEnv Explorer — Customer Support Triage</h1>
  <span class="badge">RUNNING</span>
</div>
<div class="container">
  <div>
    <div class="card" style="margin-bottom:20px">
      <h2>⚙️ Task Configuration</h2>
      <div class="field">
        <label>Select Task Difficulty</label>
        <select id="taskSelect">
          <option value="easy">Easy — Spam vs Important Classification</option>
          <option value="medium">Medium — Classification + Reply Validation</option>
          <option value="hard">Hard — Multi-step Escalation Workflow</option>
        </select>
      </div>
      <button class="btn btn-primary" onclick="resetEnv()">🔄 Reset Environment</button>
      <div class="status" id="resetStatus">No environment loaded yet.</div>
    </div>
    <div class="card">
      <h2>🎮 Execute Action</h2>
      <div class="field">
        <label>Action Type</label>
        <div class="radio-group" id="actionGroup">
          <div class="radio-btn active" onclick="selectAction(this,'classify')">classify</div>
          <div class="radio-btn" onclick="selectAction(this,'reply')">reply</div>
          <div class="radio-btn" onclick="selectAction(this,'escalate')">escalate</div>
          <div class="radio-btn" onclick="selectAction(this,'archive')">archive</div>
        </div>
      </div>
      <div class="field">
        <label>Email ID</label>
        <input type="text" id="emailId" placeholder="e.g. email_1">
      </div>
      <div class="field" id="extraField">
        <label id="extraLabel">Category</label>
        <input type="text" id="extraVal" placeholder="e.g. spam, important, billing, technical">
      </div>
      <button class="btn btn-secondary" onclick="sendStep()">⚡ Execute Action</button>
      <div class="log" id="actionLog"><p>Action log will appear here...</p></div>
    </div>
  </div>
  <div>
    <div class="card" style="margin-bottom:20px">
      <h2>📊 Performance Metrics</h2>
      <div class="metrics">
        <div class="metric"><div class="val" id="stepReward">—</div><div class="lbl">Step Reward</div></div>
        <div class="metric"><div class="val" id="totalReward">—</div><div class="lbl">Total Reward</div></div>
      </div>
      <div style="text-align:center">
        <span class="done-badge done-no" id="doneBadge">⏳ In Progress</span>
      </div>
    </div>
    <div class="card">
      <h2>🔍 Environment Observation</h2>
      <pre id="obsView">Click "Reset Environment" to start.</pre>
    </div>
  </div>
</div>
<script>
  let selectedAction = 'classify';
  let totalAccum = 0;

  function selectAction(el, action) {
    document.querySelectorAll('.radio-btn').forEach(b => b.classList.remove('active'));
    el.classList.add('active');
    selectedAction = action;
    const label = document.getElementById('extraLabel');
    const field = document.getElementById('extraField');
    if (action === 'classify') { label.textContent = 'Category'; field.style.display='block'; }
    else if (action === 'reply') { label.textContent = 'Reply Content'; field.style.display='block'; }
    else { field.style.display='none'; }
  }

  function log(msg, ok=true) {
    const el = document.getElementById('actionLog');
    const p = document.createElement('p');
    p.className = ok ? 'ok' : 'err';
    p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    el.appendChild(p);
    el.scrollTop = el.scrollHeight;
  }

  async function resetEnv() {
    const task = document.getElementById('taskSelect').value;
    document.getElementById('resetStatus').textContent = 'Resetting...';
    totalAccum = 0;
    try {
      const r = await fetch(`/reset?task=${task}`, {method:'POST'});
      const data = await r.json();
      document.getElementById('obsView').textContent = JSON.stringify(data, null, 2);
      document.getElementById('resetStatus').textContent = `✅ Environment reset for task: ${task}`;
      document.getElementById('stepReward').textContent = '0.00';
      document.getElementById('totalReward').textContent = '0.00';
      document.getElementById('doneBadge').textContent = '⏳ In Progress';
      document.getElementById('doneBadge').className = 'done-badge done-no';
      log(`Environment reset for task: ${task}`);
    } catch(e) {
      document.getElementById('resetStatus').textContent = '❌ Reset failed.';
      log('Reset failed: ' + e.message, false);
    }
  }

  async function sendStep() {
    const eid = document.getElementById('emailId').value.trim();
    const val = document.getElementById('extraVal').value.trim();
    if (!eid) { log('Please enter an Email ID.', false); return; }
    
    const payload = { action_type: selectedAction, email_id: eid };
    if (selectedAction === 'classify') payload.category = val;
    if (selectedAction === 'reply') payload.content = val;

    try {
      const r = await fetch('/step', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
      const data = await r.json();
      if (!r.ok) { log('Step error: ' + JSON.stringify(data), false); return; }
      
      const reward = (data.reward || 0).toFixed(2);
      totalAccum += data.reward || 0;
      
      document.getElementById('stepReward').textContent = reward;
      document.getElementById('totalReward').textContent = totalAccum.toFixed(2);
      document.getElementById('obsView').textContent = JSON.stringify(data.observation, null, 2);
      
      if (data.done) {
        document.getElementById('doneBadge').textContent = '✅ Task Complete!';
        document.getElementById('doneBadge').className = 'done-badge done-yes';
      }
      log(`Action: ${selectedAction} | Email: ${eid} | Reward: ${reward}`);
    } catch(e) {
      log('Step failed: ' + e.message, false);
    }
  }
</script>
</body>
</html>"""


# Serve the UI at both / and /web (HF Spaces uses /web via base_path)
@app.get("/", response_class=HTMLResponse)
@app.get("/web", response_class=HTMLResponse)
@app.get("/web/", response_class=HTMLResponse)
async def ui():
    return UI_HTML

# OpenEnv API routes — also accessible at /web prefix for HF proxy compatibility
@app.get("/health")
@app.get("/web/health")
def health():
    return {"status": "ok", "env": "customer-support"}

@app.post("/reset", response_model=Observation)
@app.post("/web/reset")
async def reset(task: str = "easy"):
    return await env.reset(task_name=task)

@app.post("/step", response_model=RewardOutput)
@app.post("/web/step")
async def step(action: Action = Body(...)):
    return await env.step(action)


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
