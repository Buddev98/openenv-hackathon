from customer_support_env.models import Email, EmailCategory, EmailStatus

TASK_EMAILS = [
    Email(
        id="email_9",
        subject="URGENT: Billing Overcharge X2",
        body="I was charged for two subscriptions this month! This is unacceptable. Please refund the extra charge immediately.",
        expected_category=EmailCategory.REFUND
    ),
    Email(
        id="email_10",
        subject="Account Security Breach - Help!",
        body="I have strange transactions on my account and I cannot log in. Please lock my account and help me recover it.",
        expected_category=EmailCategory.TECHNICAL
    ),
    Email(
        id="email_11",
        subject="You have won a free iPhone 15 Pro",
        body="Click this official link to claim your reward! No sign-up required, just $1 shipping.",
        expected_category=EmailCategory.SPAM
    ),
    Email(
        id="email_12",
        subject="Strategic Partnership Inquiry",
        body="I am writing from the CEO's office. We are interested in exploring a potential partnership with your firm.",
        expected_category=EmailCategory.IMPORTANT
    ),
    Email(
        id="email_13",
        subject="Technical: API Key Renewal",
        body="Our current API key is set to expire in 48 hours. We need to know how to rotate it securely before the service is interrupted.",
        expected_category=EmailCategory.TECHNICAL
    )
]

def get_task_data():
    """Objectives: Multi-step triage - Verify Refund/Security issues and escalate accordingly."""
    return TASK_EMAILS
