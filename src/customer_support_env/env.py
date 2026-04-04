import asyncio
import copy
from typing import List, Optional, Dict, Any
from customer_support_env.models import (
    Email, EmailStatus, EmailCategory, Observation, 
    Action, RewardOutput, ClassifyAction, ReplyAction,
    EscalateAction, ArchiveAction
)
from customer_support_env.tasks.grader import DefaultGrader
import customer_support_env.tasks.easy as easy
import customer_support_env.tasks.medium as medium
import customer_support_env.tasks.hard as hard

class CustomerSupportEnv:
    def __init__(self):
        self.emails: List[Email] = []
        self.step_count = 0
        self.max_steps = 15
        self.grader = DefaultGrader()
        self.task_name = "easy"
        self.total_reward = 0.0

    async def reset(self, task_name: str = "easy") -> Observation:
        self.task_name = task_name
        if task_name == "easy":
            self.emails = copy.deepcopy(easy.get_task_data())
        elif task_name == "medium":
            self.emails = copy.deepcopy(medium.get_task_data())
        elif task_name == "hard":
            self.emails = copy.deepcopy(hard.get_task_data())
        else:
            raise ValueError(f"Unknown task: {task_name}")
        
        self.step_count = 0
        self.total_reward = 0.0
        return self.state()

    def state(self, message: Optional[str] = None) -> Observation:
        # Return observation (with expected_category hidden via Pydantic exclude)
        return Observation(
            emails=self.emails,
            step_count=self.step_count,
            max_steps=self.max_steps,
            message=message
        )

    def get_state(self) -> Dict[str, Any]:
        """Returns the current environment state as a plain dict (for /state endpoint)."""
        return {
            "task": self.task_name,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "total_reward": round(self.total_reward, 3),
            "emails_total": len(self.emails),
            "emails_handled": sum(1 for e in self.emails if e.status.value != "received"),
        }

    async def step(self, action: Action) -> RewardOutput:
        self.step_count += 1
        
        if self.step_count > self.max_steps:
            return RewardOutput(
                observation=self.state("Max steps reached."),
                reward=0.0,
                done=True,
                info={"error": "max_steps_reached", "final_score": self.grader.get_final_score(self.emails)}
            )

        email_id = action.email_id
        target_email = next((e for e in self.emails if e.id == email_id), None)
        
        if not target_email:
            return RewardOutput(
                observation=self.state(f"Email {email_id} not found."),
                reward=-0.5,
                done=False,
                info={"error": "email_not_found"}
            )

        old_email = copy.deepcopy(target_email)
        action_type = action.action_type
        
        # Apply action
        if action_type == "classify":
            target_email.category = action.category
        elif action_type == "reply":
            target_email.status = EmailStatus.REPLIED
        elif action_type == "escalate":
            target_email.status = EmailStatus.ESCALATED
        elif action_type == "archive":
            target_email.status = EmailStatus.ARCHIVED

        # Calculate reward from grader
        reward = self.grader.get_reward(
            old_email=old_email,
            new_email=target_email,
            action_type=action_type,
            action_data=action.model_dump()
        )
        
        # Step efficiency penalty (-0.01 per step)
        reward -= 0.01
        
        self.total_reward += reward
        
        # Check if all emails handled
        done = all(e.status != EmailStatus.RECEIVED for e in self.emails) or self.step_count >= self.max_steps
        
        final_score = self.grader.get_final_score(self.emails) if done else 0.0

        return RewardOutput(
            observation=self.state(),
            reward=round(reward, 3),
            done=done,
            info={"total_reward": round(self.total_reward, 3), "final_score": final_score}
        )
