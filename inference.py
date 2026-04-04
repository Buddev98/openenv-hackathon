import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration (STRICT)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)

def get_action_from_llm(observation):
    """Query the LLM to decide the next action based on the observation."""
    system_prompt = (
        "You are a Customer Support AI agent. Your goal is to triage incoming emails.\n"
        "Actions available:\n"
        "- classify: email_id, category (spam, refund, technical, important, general)\n"
        "- reply: email_id, content\n"
        "- escalate: email_id\n"
        "- archive: email_id\n\n"
        "Rules:\n"
        "1. For spam, classify as spam and then archive.\n"
        "2. For refund or technical issues, classify correctly and then escalate.\n"
        "3. For important emails, classify and then reply.\n\n"
        "Return your choice as a JSON object."
    )
    
    user_prompt = f"Observation: {json.dumps(observation)}\nNext Action:"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def run_evaluation(task_name):
    """Run a full evaluation loop for a specific task."""
    env_name = "openenv-customer-support"
    
    # [START] Logging
    print(f"[START] task={task_name} env={env_name} model={MODEL_NAME}")
    
    # 1. Reset Environment
    try:
        reset_url = f"{API_BASE_URL}/reset?task={task_name}"
        response = requests.post(reset_url, timeout=30)
        response.raise_for_status()
        observation = response.json()
    except Exception as e:
        print(f"[END] success=False steps=0 score=0.0 rewards=[] error='Reset failed: {str(e)}'")
        return

    done = False
    step_count = 0
    rewards = []
    
    # 2. Main Step Loop
    while not done and step_count < 15:
        step_count += 1
        
        # Get Action
        action = get_action_from_llm(observation)
        
        if "error" in action:
            print(f"[STEP] step={step_count} action='error' reward=0.0 done=True error='{action['error']}'")
            break
            
        # Execute Step
        try:
            step_url = f"{API_BASE_URL}/step"
            step_response = requests.post(step_url, json=action, timeout=30)
            step_response.raise_for_status()
            result = step_response.json()
            
            observation = result["observation"]
            reward = float(result["reward"])
            done = bool(result["done"])
            rewards.append(reward)
            
            # [STEP] Logging (STRICT)
            action_str = json.dumps(action)
            print(f"[STEP] step={step_count} action='{action_str}' reward={reward} done={done} error=None")
            
        except Exception as e:
            print(f"[STEP] step={step_count} action='{json.dumps(action)}' reward=0.0 done=True error='{str(e)}'")
            break
            
    # 3. Final Summary
    # Score is already normalized (0.0 - 1.0) by the environment grader
    score = result["info"].get("final_score", 0.0) if 'result' in locals() else 0.0
    success = score >= 0.8 # Threshold for success in logs
    
    # [END] Logging (STRICT)
    print(f"[END] success={success} steps={step_count} score={float(score)} rewards={rewards}")

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("CRITICAL: OPENAI_API_KEY is not set.")
    else:
        # Evaluate mandatory 3 tasks
        for task in ["easy", "medium", "hard"]:
            run_evaluation(task)
            print() # Spacer between task logs
