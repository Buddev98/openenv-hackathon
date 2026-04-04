from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from typing import Dict, Any, Union, Optional
import os
import gradio as gr

# Import from the absolute package structure
from customer_support_env.models import (
    Observation, Action, RewardOutput, 
    ClassifyAction, ReplyAction, EscalateAction, ArchiveAction
)
from customer_support_env.env import CustomerSupportEnv

# Initialize FastAPI and Environment
app = FastAPI(title="OpenEnv: Customer Support Triage")
env = CustomerSupportEnv()

@app.get("/health")
def health():
    return {"status": "ok", "env": "customer-support"}

@app.post("/reset", response_model=Observation)
async def reset(task: str = "easy"):
    return await env.reset(task_name=task)

@app.post("/step", response_model=RewardOutput)
async def step(action: Union[ClassifyAction, ReplyAction, EscalateAction, ArchiveAction] = Body(...)):
    # Standard OpenEnv Step route
    return await env.step(action)

# --- ROBUST REDIRECT HANDLER (Must be registered before mounting) ---
@app.get("/", response_class=HTMLResponse)
def root_page():
    return """
    <html>
        <head><meta http-equiv="refresh" content="0; url='/web/'" /></head>
        <body style="background: #0b0f19; color: white; display: flex; justify-content: center; align-items: center; height: 100vh; font-family: sans-serif;">
            <div style="text-align: center;">
                <h2>📧 Launching OpenEnv UI...</h2>
                <p>Redirecting to <a href="/web/" style="color: #3498db;">/web/</a></p>
            </div>
        </body>
    </html>
    """

# --- PREMIUM WEB INTERFACE ---
def create_web_ui():
    with gr.Blocks(title="OpenEnv Explorer", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 📧 OpenEnv: Customer Support Triage")
        gr.Markdown("Agentic environment explorer for customer support triage.")
        
        with gr.Row():
            task_select = gr.Dropdown(["easy", "medium", "hard"], label="Select Task", value="easy")
            reset_btn = gr.Button("Reset Environment", variant="primary")
        
        with gr.Row():
            obs_json = gr.JSON(label="Observation View")
        
        with gr.Row():
            with gr.Column():
                atype = gr.Radio(["classify", "reply", "escalate", "archive"], label="Action", value="classify")
                eid = gr.Textbox(label="Email ID")
                val = gr.Textbox(label="Parameter (e.g., Category)")
                step_btn = gr.Button("Execute", variant="secondary")
            
            with gr.Column():
                rew = gr.Number(label="Reward")
                total = gr.Number(label="Total Reward")
                done = gr.Checkbox(label="Done")

        async def ui_reset(t):
            res = await env.reset(t)
            return res.model_dump(), 0, 0, False

        async def ui_step(at, id, v):
            data = {"action_type": at, "email_id": id}
            if at == "classify": data["category"] = v
            if at =="reply": data["content"] = v
            res = await env.step(Action(**data))
            return res.observation.model_dump(), res.reward, res.info.get("total_reward", 0), res.done

        reset_btn.click(ui_reset, inputs=[task_select], outputs=[obs_json, rew, total, done])
        step_btn.click(ui_step, inputs=[atype, eid, val], outputs=[obs_json, rew, total, done])
        
    return demo

# Mount Gradio safely at /web
demo = create_web_ui()
app = gr.mount_gradio_app(app, demo, path="/web")

def main():
    import uvicorn
    # Use standard 7860 or environment variable for HF Spaces
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
