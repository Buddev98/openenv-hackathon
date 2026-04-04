from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict, Any, Union
import gradio as gr
import os

# 1. Environment Logic (Models/Env)
from customer_support_env.models import (
    Observation, Action, RewardOutput, 
    ClassifyAction, ReplyAction, EscalateAction, ArchiveAction
)
from customer_support_env.env import CustomerSupportEnv
env = CustomerSupportEnv()

# 2. Initialize FastAPI
app = FastAPI(title="OpenEnv: Customer Support Triage")

@app.get("/health")
def health():
    return {"status": "ok", "env": "customer-support"}

@app.post("/reset", response_model=Observation)
async def reset(task: str = "easy"):
    return await env.reset(task_name=task)

@app.post("/step", response_model=RewardOutput)
async def step(action: Action = Body(...)):
    return await env.step(action)

# --- ROBUST UI LOADER at Root ---
@app.get("/", response_class=HTMLResponse)
def root_page():
    """Serves the UI inside a full-screen iframe to avoid proxy and mixed-content issues."""
    return """
    <!DOCTYPE html>
    <html style="margin: 0; padding: 0; height: 100%; overflow: hidden;">
    <head>
        <title>OpenEnv Explorer</title>
        <style>iframe { width: 100%; height: 100%; border: none; margin: 0; padding: 0; }</style>
    </head>
    <body style="margin: 0; padding: 0; height: 100%;">
        <iframe src="./web/"></iframe>
    </body>
    </html>
    """

# 3. Create Gradio UI
def create_web_ui():
    with gr.Blocks(title="OpenEnv Explorer", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 📧 OpenEnv: Customer Support Triage")
        with gr.Row():
            task_select = gr.Dropdown(["easy", "medium", "hard"], label="Task", value="easy")
            reset_btn = gr.Button("Reset", variant="primary")
        obs_json = gr.JSON(label="Observation View")
        with gr.Row():
            with gr.Column():
                atype = gr.Radio(["classify", "reply", "escalate", "archive"], label="Action", value="classify")
                eid = gr.Textbox(label="Email ID")
                val = gr.Textbox(label="Parameter")
                step_btn = gr.Button("Send", variant="secondary")
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
            if at == "reply": data["content"] = v
            res = await env.step(Action(**data))
            return res.observation.model_dump(), res.reward, res.info.get("total_reward", 0), res.done

        reset_btn.click(ui_reset, inputs=[task_select], outputs=[obs_json, rew, total, done])
        step_btn.click(ui_step, inputs=[atype, eid, val], outputs=[obs_json, rew, total, done])
        
    return demo

# 4. Mount Gradio to /web
demo = create_web_ui()
app = gr.mount_gradio_app(app, demo, path="/web")

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
