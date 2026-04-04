import gradio as gr
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from customer_support_env.models import (
    Observation, Action, RewardOutput,
    ClassifyAction, ReplyAction, EscalateAction, ArchiveAction
)
from customer_support_env.env import CustomerSupportEnv
import os

env = CustomerSupportEnv()

# Create the main FastAPI app
app = FastAPI(title="OpenEnv: Customer Support Triage")

# --- OpenEnv API routes ---
@app.get("/health")
def health():
    return {"status": "ok", "env": "customer-support"}

@app.get("/")
def root():
    return {"status": "ok", "env": "customer-support", "ui": "/web/"}

@app.post("/reset", response_model=Observation)
async def reset(task: str = "easy"):
    return await env.reset(task_name=task)

@app.post("/step", response_model=RewardOutput)
async def step(action: Action = Body(...)):
    return await env.step(action)

# --- Gradio UI ---
def create_web_ui():
    with gr.Blocks(title="OpenEnv Explorer") as demo:
        gr.Markdown("# 📧 OpenEnv: Customer Support Triage")
        gr.Markdown("**Agentic execution environment for customer support email triage.**")

        with gr.Row():
            task_select = gr.Dropdown(["easy", "medium", "hard"], label="Select Task", value="easy")
            reset_btn = gr.Button("Reset Environment", variant="primary")

        obs_json = gr.JSON(label="Current Observation")

        with gr.Row():
            with gr.Column():
                atype = gr.Radio(["classify", "reply", "escalate", "archive"], label="Action Type", value="classify")
                eid = gr.Textbox(label="Email ID (e.g. email_1)")
                val = gr.Textbox(label="Category / Reply Content")
                step_btn = gr.Button("Execute Action", variant="secondary")
            with gr.Column():
                rew = gr.Number(label="Step Reward")
                total = gr.Number(label="Total Reward")
                done = gr.Checkbox(label="Task Done")

        async def ui_reset(t):
            res = await env.reset(t)
            return res.model_dump(), 0, 0, False

        async def ui_step(at, id, v):
            data = {"action_type": at, "email_id": id}
            if at == "classify":
                data["category"] = v
            elif at == "reply":
                data["content"] = v
            res = await env.step(Action(**data))
            return res.observation.model_dump(), res.reward, res.info.get("total_reward", 0), res.done

        reset_btn.click(ui_reset, inputs=[task_select], outputs=[obs_json, rew, total, done])
        step_btn.click(ui_step, inputs=[atype, eid, val], outputs=[obs_json, rew, total, done])

    return demo

# Mount Gradio at /web — This matches the base_path set in README.md by openenv push
demo = create_web_ui()
app = gr.mount_gradio_app(app, demo, path="/web", root_path="/web")

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, root_path="")

if __name__ == "__main__":
    main()
