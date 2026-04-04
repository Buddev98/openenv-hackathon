from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from customer_support_env.models import (
    Observation, Action, RewardOutput, 
    ClassifyAction, ReplyAction, EscalateAction, ArchiveAction
)
from customer_support_env.env import CustomerSupportEnv
from typing import Dict, Any, Union
import gradio as gr

# Initialize FastAPI
app = FastAPI(title="OpenEnv API")
env = CustomerSupportEnv()

@app.get("/health")
def health():
    return {"status": "ok", "env": "customer-support"}

@app.post("/reset", response_model=Observation)
async def reset(task: str = "easy"):
    return await env.reset(task_name=task)

@app.post("/step", response_model=RewardOutput)
async def step(action: Union[ClassifyAction, ReplyAction, EscalateAction, ArchiveAction] = Body(...)):
    # Applies an action and returns the reward/new observation
    return await env.step(action)

# --- PREMIUM WEB INTERFACE ---
def create_web_ui():
    with gr.Blocks(title="OpenEnv Explorer", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 📧 OpenEnv: Customer Support Triage")
        gr.Markdown("Directly interact with the environment through this interactive panel.")
        
        with gr.Row():
            task_select = gr.Dropdown(["easy", "medium", "hard"], label="Select Task", value="easy")
            reset_btn = gr.Button("Reset Environment", variant="primary")
        
        with gr.Row():
            obs_json = gr.JSON(label="Observation View")
        
        with gr.Row():
            with gr.Column():
                action_type = gr.Radio(["classify", "reply", "escalate", "archive"], label="Action", value="classify")
                email_id = gr.Textbox(label="Email ID")
                extra = gr.Textbox(label="Value (Category or Reply Content)")
                step_btn = gr.Button("Execute", variant="secondary")
            
            with gr.Column():
                reward_out = gr.Number(label="Step Reward")
                total_reward_out = gr.Number(label="Total Reward")
                done_out = gr.Checkbox(label="Is Task Done")

        async def ui_reset(t):
            res = await env.reset(t)
            return res.model_dump(), 0, 0, False

        async def ui_step(atype, eid, val):
            data = {"action_type": atype, "email_id": eid}
            if atype == "classify": data["category"] = val
            if atype == "reply": data["content"] = val
            
            res = await env.step(Action(**data))
            return res.observation.model_dump(), res.reward, res.info.get("total_reward", 0), res.done

        reset_btn.click(ui_reset, inputs=[task_select], outputs=[obs_json, reward_out, total_reward_out, done_out])
        step_btn.click(ui_step, inputs=[action_type, email_id, extra], outputs=[obs_json, reward_out, total_reward_out, done_out])
        
    return demo

# Mount Gradio safely at /web
web_ui = create_web_ui()
app = gr.mount_gradio_app(app, web_ui, path="/web")

# --- ROBUST ROOT HANDLER ---
@app.get("/", response_class=HTMLResponse)
def root_page():
    """Returns a robust HTML page that redirects to /web/ to avoid HF Space 404s."""
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url='/web/'" />
            <title>Loading OpenEnv UI...</title>
            <style>
                body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background: #0b0f19; color: white; }
                .loader { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 2s linear infinite; }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            </style>
        </head>
        <body>
            <div style="text-align: center;">
                <div class="loader" style="margin: 0 auto 20px;"></div>
                <h2>Launching OpenEnv Explorer...</h2>
                <p>If not redirected, <a href="/web/" style="color: #3498db;">click here</a>.</p>
            </div>
        </body>
    </html>
    """

def main():
    import uvicorn, os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
