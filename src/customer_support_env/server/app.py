from fastapi import FastAPI, HTTPException, Body
from customer_support_env.models import (
    Observation, Action, RewardOutput, 
    ClassifyAction, ReplyAction, EscalateAction, ArchiveAction
)
from customer_support_env.env import CustomerSupportEnv
from typing import Dict, Any, Union
import gradio as gr

# Initialize FastAPI for the API endpoints
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
    if hasattr(action, 'action_type'):
        # Validates and applies action
        return await env.step(action)
    raise HTTPException(status_code=400, detail="Invalid action format")

# --- CONSOLIDATED WEB INTERFACE ---
def create_web_ui():
    with gr.Blocks(title="OpenEnv Explorer", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 📧 OpenEnv: Customer Support Triage")
        gr.Markdown("Interactive explorer for the agentic execution environment.")
        
        with gr.Row():
            with gr.Column(scale=1):
                task_select = gr.Dropdown(["easy", "medium", "hard"], label="Select Task", value="easy")
                reset_btn = gr.Button("Reset Environment", variant="primary")
                
                gr.Markdown("### 🔧 Execute Action")
                action_type = gr.Radio(["classify", "reply", "escalate", "archive"], label="Action Type", value="classify")
                email_id = gr.Textbox(label="Email ID (e.g., email_1)")
                extra_input = gr.Textbox(label="Category / Content (optional)")
                step_btn = gr.Button("Send Action", variant="secondary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 🔍 Environment State")
                obs_json = gr.JSON(label="Observation")
                
                with gr.Row():
                    reward_out = gr.Number(label="Last Step Reward", precision=2)
                    total_reward_out = gr.Number(label="Total Task Reward", precision=2)
                    done_out = gr.Checkbox(label="Task Complete")

        async def ui_reset(t):
            res = await env.reset(t)
            return res.model_dump(), 0, 0, False

        async def ui_step(atype, eid, extra):
            # Map UI to Environment Action
            data = {"action_type": atype, "email_id": eid}
            if atype == "classify": data["category"] = extra
            if atype == "reply": data["content"] = extra
            
            try:
                res = await env.step(Action(**data))
                return res.observation.model_dump(), res.reward, res.info.get("total_reward", 0), res.done
            except Exception as e:
                return {"error": str(e)}, 0, 0, False

        reset_btn.click(ui_reset, inputs=[task_select], outputs=[obs_json, reward_out, total_reward_out, done_out])
        step_btn.click(ui_step, inputs=[action_type, email_id, extra_input], outputs=[obs_json, reward_out, total_reward_out, done_out])
        
    return demo

# Mount Gradio directly at / and ensure it is the main application
web_ui = create_web_ui()
app = gr.mount_gradio_app(app, web_ui, path="/")

def main():
    import uvicorn
    # Use 0.0.0.0 and the PORT environment variable (standard for HF)
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
