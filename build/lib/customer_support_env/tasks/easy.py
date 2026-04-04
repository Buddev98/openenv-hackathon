from customer_support_env.models import Email, EmailCategory, EmailStatus

TASK_EMAILS = [
    Email(
        id="email_1",
        subject="Claim your prize!",
        body="Congratulations! You won $10,000. Click here to claim your reward before it expires.",
        expected_category=EmailCategory.SPAM
    ),
    Email(
        id="email_2",
        subject="Project Update: Monday",
        body="Hi Team, just a quick reminder that the sync is scheduled for 2 PM on Monday.",
        expected_category=EmailCategory.IMPORTANT
    ),
    Email(
        id="email_3",
        subject="Quick question",
        body="Hey, where can I find the latest documentation and roadmap for the project?",
        expected_category=EmailCategory.IMPORTANT
    ),
    Email(
        id="email_4",
        subject="Discount Code INSIDE",
        body="Don't miss our summer sale! Use code SUMMER24 for 50% off all items.",
        expected_category=EmailCategory.SPAM
    )
]

def get_task_data():
    """Objectives: Classify Spam vs Important emails and handle accordingly."""
    return TASK_EMAILS
