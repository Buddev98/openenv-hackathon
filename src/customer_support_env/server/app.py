from fastapi import FastAPI, HTTPException, Body
from customer_support_env.models import (
    Observation, Action, RewardOutput, 
    ClassifyAction, ReplyAction, EscalateAction, ArchiveAction
)
from customer_support_env.env import CustomerSupportEnv
from typing import Dict, Any, Union
import gradio as gr

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
    return await env.step(action)

# --- PREMIUM WEB INTERFACE ---
def create_web_ui():
    with gr.Blocks(title="OpenEnv Explorer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📧 Customer Support Email Triage\nAgentic Execution Environment for OpenEnv.")
        
        with gr.Row():
            task_select = gr.Dropdown(["easy", "medium", "hard"], label="Select Task", value="easy")
            reset_btn = gr.Button("Reset Environment", variant="primary")
        
        with gr.Row():
            obs_json = gr.JSON(label="Current Observation")
        
        with gr.Row():
            with gr.Column():
                action_type = gr.Radio(["classify", "reply", "escalate", "archive"], label="Action Type", value="classify")
                email_id = gr.Textbox(label="Email ID")
                cat_input = gr.Textbox(label="Category (optional)")
                content_input = gr.Textbox(label="Content (optional)")
                step_btn = gr.Button("Execute Step", variant="secondary")
            
            with gr.Column():
                reward_out = gr.Number(label="Last Step Reward")
                total_reward_out = gr.Number(label="Total Reward")
                done_out = gr.Checkbox(label="Done")

        async def ui_reset(t):
            res = await env.reset(t)
            return res.model_dump(), 0, 0, False

        async def ui_step(atype, eid, cat, cont):
            # Formulate action
            data = {"action_type": atype, "email_id": eid}
            if atype == "classify": data["category"] = cat
            if atype == "reply": data["content"] = cat # Content for reply, using cat input as proxy
            if atype == "escalate": pass
            
            # Step
            res = await env.step(Action(**data))
            return res.observation.model_dump(), res.reward, res.info.get("total_reward", 0), res.done

        reset_btn.click(ui_reset, inputs=[task_select], outputs=[obs_json, reward_out, total_reward_out, done_out])
        step_btn.click(ui_step, inputs=[action_type, email_id, cat_input, content_input], outputs=[obs_json, reward_out, total_reward_out, done_out])
        
    return demo

# Mount Gradio to root / (This ensures it is the default page for the Space)
web_ui = create_web_ui()
app = gr.mount_gradio_app(app, web_ui, path="/")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
