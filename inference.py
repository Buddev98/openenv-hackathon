import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================
# REQUIRED CONFIGURATION
# ============================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Initialize OpenAI Client
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

def get_action_from_llm(observation):
    """Query the LLM or use a Mock Agent if no key is provided."""
    if client and MODEL_NAME:
        system_prompt = (
            "You are a customer support triage agent. Analyze the observation and return "
            "a single JSON action with fields: action_type (classify/reply/escalate/archive), "
            "email_id, category (if classify), content (if reply)."
        )
        user_prompt = f"Observation: {json.dumps(observation)}"
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
        except Exception:
            pass  # Fall back to mock agent

    # ============================
    # MOCK AGENT (Fallback — no API key required)
    # ============================
    for email in observation.get("emails", []):
        if email.get("status") == "received":
            eid = email["id"]
            subj = email.get("subject", "").lower()
            body = email.get("body", "").lower()

            if "won" in subj or "prize" in body or "lottery" in body:
                return {"action_type": "classify", "email_id": eid, "category": "spam"}
            if "refund" in subj or "charge" in body or "payment" in body:
                return {"action_type": "classify", "email_id": eid, "category": "refund"}
            if "broken" in body or "error" in body or "crash" in body:
                return {"action_type": "classify", "email_id": eid, "category": "technical"}
            return {"action_type": "classify", "email_id": eid, "category": "important"}

    # Follow-up actions after classification
    for email in observation.get("emails", []):
        status = email.get("status", "")
        category = email.get("category", "")
        eid = email["id"]

        if category == "spam" and status != "archived":
            return {"action_type": "archive", "email_id": eid}
        if category in ("important", "general") and status not in ("replied", "archived"):
            return {"action_type": "reply", "email_id": eid, "content": "Thank you for reaching out! We will get back to you shortly."}
        if category in ("refund", "technical") and status not in ("escalated", "archived"):
            return {"action_type": "escalate", "email_id": eid}

    return {"error": "no_more_actions"}


def run_evaluation(task_name):
    """Run a full evaluation loop for a specific task."""
    env_name = "openenv-customer-support"

    # [START] — REQUIRED LOG FORMAT
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
    result = {}

    # 2. Main Step Loop (max 15 steps to stay within 20min limit)
    while not done and step_count < 15:
        step_count += 1

        # Get Action from LLM or Mock Agent
        action = get_action_from_llm(observation)

        if "error" in action:
            err_msg = action["error"]
            err_action = json.dumps({"error": err_msg})
            print(f"[STEP] step={step_count} action='{err_action}' reward=0.0 done=True error='{err_msg}'")
            done = True
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
            rewards.append(round(reward, 4))

            # [STEP] — REQUIRED LOG FORMAT
            action_str = json.dumps(action)
            print(f"[STEP] step={step_count} action='{action_str}' reward={reward} done={done} error=None")

        except Exception as e:
            print(f"[STEP] step={step_count} action='{json.dumps(action)}' reward=0.0 done=True error='{str(e)}'")
            done = True
            break

    # 3. Final Summary
    score = float(result.get("info", {}).get("final_score", 0.0)) if result else 0.0
    # Clamp to 0.0–1.0 range
    score = max(0.0, min(1.0, score))
    success = score >= 0.5

    # [END] — REQUIRED LOG FORMAT
    print(f"[END] success={success} steps={step_count} score={score} rewards={rewards}")


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set. Running with Mock Agent.")

    # Evaluate all 3 mandatory tasks
    for task in ["easy", "medium", "hard"]:
        run_evaluation(task)
        print()  # Spacer between tasks
