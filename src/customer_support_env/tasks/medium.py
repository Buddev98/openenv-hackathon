from customer_support_env.models import Email, EmailCategory, EmailStatus

TASK_EMAILS = [
    Email(
        id="email_5",
        subject="Refund for Order #10023",
        body="I would like to request a refund for the broken item I received. I've been waiting for weeks and no response.",
        expected_category=EmailCategory.REFUND
    ),
    Email(
        id="email_6",
        subject="Login Issues - Support Needed",
        body="I am locked out of my account and I need to reset my password, but the email never arrives.",
        expected_category=EmailCategory.TECHNICAL
    ),
    Email(
        id="email_7",
        subject="Great service!",
        body="Just wanted to thank you for the quick response to my last ticket. You're making my life easy!",
        expected_category=EmailCategory.IMPORTANT
    ),
    Email(
        id="email_8",
        subject="Question about Billing Cycle",
        body="When will my next payment be processed? I need to make sure I have enough funds in my account.",
        expected_category=EmailCategory.GENERAL
    )
]

def get_task_data():
    """Objectives: Classify, reply to important/general, and handle refund/tech requests."""
    return TASK_EMAILS
