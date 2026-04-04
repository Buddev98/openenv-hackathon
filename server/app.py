import gradio as gr
from fastapi import FastAPI, Body, HTTPException
from customer_support_env.models import (
    Observation, Action, RewardOutput, 
    ClassifyAction, ReplyAction, EscalateAction, ArchiveAction
)
from customer_support_env.env import CustomerSupportEnv
import os

# 1. Initialize Environment
env = CustomerSupportEnv()

# 2. Define the Premium Web UI
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
                rew = gr.Number(label="Step Reward")
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

# 3. Create the Main App (Gradio first)
demo = create_web_ui()

# 4. Initialize FastAPI for the OpenEnv API
api = FastAPI(title="OpenEnv API")

@api.get("/health")
def health():
    return {"status": "ok", "env": "customer-support"}

@api.post("/reset", response_model=Observation)
async def reset(task: str = "easy"):
    return await env.reset(task_name=task)

@api.post("/step", response_model=RewardOutput)
async def step(action: Action = Body(...)):
    # Standard OpenEnv Step route
    return await env.step(action)

# 5. Mount the API into the Gradio app at /api prefix for stability
# Note: We will keep the root (/) for the UI to avoid Mixed Content errors
app = gr.mount_fastapi_app(demo, api, path="/api")

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    # Run the combined app
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
