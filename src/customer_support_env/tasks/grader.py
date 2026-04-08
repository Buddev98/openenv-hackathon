from abc import ABC, abstractmethod
from typing import List, Dict, Any
from customer_support_env.models import Email, EmailStatus, EmailCategory

class BaseGrader(ABC):
    @abstractmethod
    def get_reward(self, old_email: Email, new_email: Email, action_type: str, action_data: Dict[str, Any]) -> float:
        """Calculate step reward for an action."""
        pass

    @abstractmethod
    def get_final_score(self, emails: List[Email]) -> float:
        """Calculate final score (0.0 to 1.0) for the task."""
        pass

class DefaultGrader(BaseGrader):
    def __init__(self, 
                 correct_class_reward: float = 0.25, 
                 wrong_class_penalty: float = -0.15,
                 correct_workflow_reward: float = 0.35,
                 wrong_workflow_penalty: float = -0.25,
                 useless_action_penalty: float = -0.1,
                 invalid_action_penalty: float = -0.5):
        self.correct_class_reward = correct_class_reward
        self.wrong_class_penalty = wrong_class_penalty
        self.correct_workflow_reward = correct_workflow_reward
        self.wrong_workflow_penalty = wrong_workflow_penalty
        self.useless_action_penalty = useless_action_penalty
        self.invalid_action_penalty = invalid_action_penalty

    def get_reward(self, old_email: Email, new_email: Email, action_type: str, action_data: Dict[str, Any]) -> float:
        reward = 0.0
        
        # 1. Detect Repeated/Useless Actions (No state change)
        if action_type == "classify" and old_email.category == new_email.category and old_email.category != EmailCategory.UNCATEGORIZED:
            return self.useless_action_penalty
        if action_type == "reply" and old_email.status == EmailStatus.REPLIED:
            return self.useless_action_penalty
        if action_type == "escalate" and old_email.status == EmailStatus.ESCALATED:
            return self.useless_action_penalty
        if action_type == "archive" and old_email.status == EmailStatus.ARCHIVED:
            return self.useless_action_penalty

        # 2. Progress Reward (Dense signal for touching a new email)
        if old_email.status == EmailStatus.RECEIVED and action_type == "classify":
            reward += 0.05 # Small "touch" reward

        # 3. Main Business Logic
        if action_type == "classify":
            if new_email.category == old_email.expected_category:
                reward += self.correct_class_reward
            else:
                reward += self.wrong_class_penalty
        
        elif action_type == "reply":
            is_good = self._is_good_reply(new_email, action_data.get("content", ""))
            is_correct_cat = new_email.category == old_email.expected_category
            
            if is_good and is_correct_cat:
                reward += self.correct_workflow_reward
            elif is_good:
                reward += 0.1 # Partial reward
            else:
                reward += self.wrong_workflow_penalty
        
        elif action_type == "escalate":
            is_correct_cat = new_email.category == old_email.expected_category
            needs_escalation = old_email.expected_category in [EmailCategory.REFUND, EmailCategory.TECHNICAL]
            
            if is_correct_cat and needs_escalation:
                reward += self.correct_workflow_reward
            elif needs_escalation:
                reward += 0.1
            else:
                reward += self.wrong_workflow_penalty
                
        elif action_type == "archive":
            is_correct_cat = new_email.category == old_email.expected_category
            is_spam = old_email.expected_category == EmailCategory.SPAM
            
            if is_correct_cat and is_spam:
                reward += self.correct_workflow_reward
            elif is_spam:
                reward += 0.1
            else:
                reward += self.wrong_workflow_penalty
                
        return round(reward, 3)

    def _is_good_reply(self, email: Email, content: str) -> bool:
        content = content.lower()
        if len(content) < 15:
            return False
            
        category = email.expected_category
        if category == EmailCategory.REFUND:
            keywords = ["refund", "process", "order", "money"]
        elif category == EmailCategory.TECHNICAL:
            keywords = ["investigate", "fix", "logs", "access", "support"]
        elif category == EmailCategory.IMPORTANT:
            keywords = ["thank", "help", "assistant", "contact"]
        else:
            keywords = ["hello", "support", "team"]
            
        return any(k in content for k in keywords)

    def get_final_score(self, emails: List[Email]) -> float:
        if not emails:
            # Clamp: never return exactly 0.0 — validator requires strictly > 0
            return 0.01
        
        total_score = 0.0
        for email in emails:
            email_score = 0.0
            if email.category == email.expected_category:
                email_score += 0.4
            
            cat = email.expected_category
            stat = email.status
            
            if cat == EmailCategory.SPAM and stat == EmailStatus.ARCHIVED:
                email_score += 0.6
            elif cat in [EmailCategory.REFUND, EmailCategory.TECHNICAL] and stat == EmailStatus.ESCALATED:
                email_score += 0.6
            elif cat == EmailCategory.IMPORTANT and stat == EmailStatus.REPLIED:
                email_score += 0.6
            elif cat == EmailCategory.GENERAL and stat in [EmailStatus.REPLIED, EmailStatus.ARCHIVED]:
                email_score += 0.6
                
            total_score += email_score
        
        raw = total_score / len(emails)
        # Clamp strictly to open interval (0, 1) — validator rejects 0.0 and 1.0 exactly
        clamped = max(0.01, min(0.99, raw))
        return round(clamped, 4)
