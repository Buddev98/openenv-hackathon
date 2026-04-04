---
title: Customer Support Env
emoji: mail
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# OpenEnv: Customer Support Email Triage

[OpenEnv Compatible](https://github.com/meta-pytorch/OpenEnv)
[Hugging Face Spaces](https://huggingface.co/spaces)

A high-fidelity agentic execution environment designed to simulate real-world customer support email triage workflows. This environment is strictly compliant with the OpenEnv specification.

---

## Environment Overview & Motivation

In modern enterprise environments, customer support teams are overwhelmed by high ticket volumes. This environment simulates the critical first layer of support: Triage. 

AI agents must process incoming emails, categorize them accurately (e.g., distinguishing a simple query from a critical security breach), and execute the correct workflow action-whether it's a direct reply, an internal escalation, or archiving spam.

### Key Challenges for Agents:
- Accuracy: Misclassifying a refund request as spam leads to poor customer experience.
- Multi-step Reasoning: Agents must classify an issue before taking a terminal action.
- Efficiency: Resolving issues in the minimum number of steps is prioritized.

---

## Action Space

The agent interacts via the /step endpoint using the following actions:

| Action | Arguments | Description |
| :--- | :--- | :--- |
| classify | email_id, category | Assign a category: spam, refund, technical, important, general. |
| reply | email_id, content | Send a response to the customer. |
| escalate | email_id | Hand off to a human specialist (required for refunds and technical bugs). |
| archive | email_id | Close the ticket (recommended for spam and resolved issues). |

---

## Observation Space

The environment state is provided as a JSON object:

- emails: A list of current email objects.
    - id: Unique identifier.
    - subject: Email subject line.
    - body: Full email content.
    - status: One of received, replied, escalated, archived.
    - category: Current assigned category.
- step_count / max_steps: Progression tracking (limit is 15 steps per task).

---

## Task Descriptions

| Task ID | Level | Objective |
| :--- | :--- | :--- |
| easy | Easy | Correctly identify and archive spam while flagging important messages. |
| medium | Medium | Triage and generate high-quality replies for important and general queries. |
| hard | Hard | Complex multi-step reasoning: detect refund or security issues and escalate them to the correct department. |

---

## Reward Design

The environment uses a dense reward function to provide continuous signals for agent learning:

- +0.25: Correct Classification (Major milestone).
- +0.35: Correct Workflow Completion (e.g., Escalating a refund request).
- +0.05: Progress Bonus (Awarded the first time an email is interacted with).
- -0.01: Efficiency Penalty (Applied to every step to encourage speed).
- -0.10: Useless Action Penalty (Repeatedly performing the same action).
- -0.50: Invalid Action (Targeting a non-existent email).

Final Score: A deterministic value between 0.0 and 1.0 based on the accuracy of categories and terminal statuses across all emails in the task.

---

## Setup & Execution

### 1. Local Development
Ensure you have Python 3.10+ and uv installed:
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the server
python server/app.py
```

### 2. Docker Deployment
The environment is optimized for a lightweight footprint:
```bash
# Build the image
docker build -t openenv-support .

# Run the container
docker run -p 7860:7860 openenv-support
```

---

## Running Inference

Use the inference.py script to evaluate an AI agent (GPT-4o required by default).

```bash
export OPENAI_API_KEY="your-key"
export API_BASE_URL="http://localhost:7860"
python inference.py
```

### Example Logs (STRICT FORMAT):
```text
[START] task=easy env=openenv-customer-support model=gpt-4o
[STEP] step=1 action='{"action_type": "classify", "email_id": "email_1", "category": "spam"}' reward=0.29 done=False error=None
[STEP] step=2 action='{"action_type": "archive", "email_id": "email_1"}' reward=0.34 done=False error=None
...
[END] success=True steps=6 score=1.0 rewards=[0.29, 0.34, 0.29, 0.34, 0.29, 0.34]
```

---

## OpenEnv Compliance
This environment has been validated with openenv-core and is ready for multi-mode deployment:
- docker
- openenv_serve
- uv_run
- python_module
