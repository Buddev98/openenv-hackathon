from fastapi import FastAPI, HTTPException, Body
from customer_support_env.models import (
    Observation, Action, RewardOutput, 
    ClassifyAction, ReplyAction, EscalateAction, ArchiveAction
)
from customer_support_env.env import CustomerSupportEnv
from typing import Dict, Any, Union

app = FastAPI(title="OpenEnv: Customer Support Triage")
env = CustomerSupportEnv()

@app.post("/reset")
async def reset(task: str = "easy") -> Observation:
    try:
        return await env.reset(task_name=task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
async def step(action: Dict[str, Any] = Body(...)) -> RewardOutput:
    """Take a step in the environment."""
    action_type = action.get("action_type")
    
    # Parse action based on type
    try:
        if action_type == "classify":
            action_obj = ClassifyAction(**action)
        elif action_type == "reply":
            action_obj = ReplyAction(**action)
        elif action_type == "escalate":
            action_obj = EscalateAction(**action)
        elif action_type == "archive":
            action_obj = ArchiveAction(**action)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported action type: {action_type}")
            
        return await env.step(action_obj)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Action validation error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
