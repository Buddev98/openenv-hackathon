from enum import Enum
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

class EmailCategory(str, Enum):
    SPAM = "spam"
    REFUND = "refund"
    TECHNICAL = "technical"
    IMPORTANT = "important"
    GENERAL = "general"
    UNCATEGORIZED = "uncategorized"

class EmailStatus(str, Enum):
    RECEIVED = "received"
    REPLIED = "replied"
    ESCALATED = "escalated"
    ARCHIVED = "archived"

class Email(BaseModel):
    id: str
    subject: str
    body: str
    category: EmailCategory = EmailCategory.UNCATEGORIZED
    status: EmailStatus = EmailStatus.RECEIVED
    expected_category: Optional[EmailCategory] = Field(None, exclude=True) # Hidden from agent observation

class Observation(BaseModel):
    emails: List[Email]
    step_count: int
    max_steps: int
    message: Optional[str] = None

class ClassifyAction(BaseModel):
    action_type: str = Field("classify", pattern="^classify$")
    email_id: str
    category: EmailCategory

class ReplyAction(BaseModel):
    action_type: str = Field("reply", pattern="^reply$")
    email_id: str
    content: str

class EscalateAction(BaseModel):
    action_type: str = Field("escalate", pattern="^escalate$")
    email_id: str

class ArchiveAction(BaseModel):
    action_type: str = Field("archive", pattern="^archive$")
    email_id: str

Action = Union[ClassifyAction, ReplyAction, EscalateAction, ArchiveAction]

class RewardOutput(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
