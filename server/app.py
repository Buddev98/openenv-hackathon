import gradio as gr
from fastapi import FastAPI, Body
from customer_support_env.models import (
    Observation, Action, RewardOutput, 
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

# 3. Initialize FastAPI for the OpenEnv API
api_app = FastAPI(title="OpenEnv API")

@api_app.get("/health")
def health():
    return {"status": "ok", "env": "customer-support"}

@api_app.post("/reset", response_model=Observation)
async def reset(task: str = "easy"):
    return await env.reset(task_name=task)

@api_app.post("/step", response_model=RewardOutput)
async def step(action: Action = Body(...)):
    return await env.step(action)

# 4. Create the Main FastAPI app and mount everything
app = FastAPI()

# Mount the API app at /api
app.mount("/api", api_app)

# Mount the Gradio UI at the root /
# This uses the correct Gradio function: mount_gradio_app
demo = create_web_ui()
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
