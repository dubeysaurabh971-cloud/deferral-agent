"""Synthetic substitute for the Kaggle ticket datasets (no Kaggle credentials available).

Hand-written tickets themed around Wix (the WixQA KB's product) so retrieval against
that KB is meaningful, loosely matching the schema of the Kaggle "Multilingual Customer
Support Tickets" set: queue, priority, subject, body.
"""
import json

from src import config

QUEUES = [
    "Billing and Payments",
    "Technical Support",
    "Account Management",
    "Domains",
    "Design and Editor",
    "Marketing and SEO",
    "Events and Bookings",
    "Email",
]

TICKETS = [
    {"queue": "Billing and Payments", "priority": "high",
     "subject": "Wix Payments account stuck in verification",
     "body": "I set up Wix Payments two weeks ago and it still shows as under verification. "
             "Can I accept payments before it's fully verified, and what's holding it up?"},
    {"queue": "Billing and Payments", "priority": "medium",
     "subject": "Double charged for premium plan",
     "body": "My card was charged twice this month for the same Wix premium plan. Please refund the duplicate charge."},
    {"queue": "Billing and Payments", "priority": "low",
     "subject": "How do I switch from monthly to yearly billing?",
     "body": "I'm on a monthly premium plan and want to switch to yearly to save money. How do I do that?"},
    {"queue": "Technical Support", "priority": "medium",
     "subject": "Site editor won't load",
     "body": "Every time I try to open the Wix Editor for my site it spins forever and never loads. "
             "I've tried two different browsers."},
    {"queue": "Technical Support", "priority": "high",
     "subject": "Site is down for all visitors",
     "body": "My published site is returning a 500 error for every visitor since this morning. This is costing me sales."},
    {"queue": "Account Management", "priority": "medium",
     "subject": "Can't log into my Wix account",
     "body": "I'm getting 'incorrect password' even after resetting it three times. I need access to fix my storefront."},
    {"queue": "Account Management", "priority": "low",
     "subject": "How to transfer site ownership to a new email",
     "body": "I'm handing my site off to a business partner. How do we transfer ownership to their Wix account?"},
    {"queue": "Domains", "priority": "medium",
     "subject": "Connected domain not working",
     "body": "I connected my GoDaddy domain to my Wix site following the instructions, but it still shows the old placeholder page."},
    {"queue": "Domains", "priority": "low",
     "subject": "How long does domain propagation take?",
     "body": "I just pointed my domain's nameservers to Wix. How long before it actually starts working?"},
    {"queue": "Design and Editor", "priority": "low",
     "subject": "How do I add a hover effect to a button?",
     "body": "I want my call-to-action button to change color when a visitor hovers over it. Is that possible in the Wix Editor?"},
    {"queue": "Design and Editor", "priority": "medium",
     "subject": "Mobile view looks broken",
     "body": "My site looks fine on desktop but on mobile the images overlap the text. How do I fix the mobile layout separately?"},
    {"queue": "Marketing and SEO", "priority": "low",
     "subject": "Site not showing up on Google",
     "body": "I published my site three weeks ago and searching my business name on Google still doesn't show it. What am I missing?"},
    {"queue": "Marketing and SEO", "priority": "medium",
     "subject": "How to set up an email marketing campaign",
     "body": "I want to send a promotional email to my customer list for a holiday sale. What's the best way to do this in Wix?"},
    {"queue": "Events and Bookings", "priority": "medium",
     "subject": "Guests can't see ticket prices on event page",
     "body": "I created a ticketed event but visitors say they only see an RSVP button, not the ticket prices or purchase option."},
    {"queue": "Events and Bookings", "priority": "low",
     "subject": "Can I disable the event details page?",
     "body": "My events don't use tickets, and I'd rather send visitors straight to registration. Can I skip the details page?"},
    {"queue": "Email", "priority": "low",
     "subject": "Custom email address not receiving mail",
     "body": "I set up name@mydomain.com through Wix but emails sent to it never arrive. Is there a setup step I'm missing?"},
    {"queue": "Billing and Payments", "priority": "high",
     "subject": "Refund request for annual plan",
     "body": "I upgraded to an annual plan by mistake three days ago. I'd like a full refund and to go back to my free plan."},
    {"queue": "Technical Support", "priority": "low",
     "subject": "How do I duplicate my site as a backup?",
     "body": "Before I make big changes I want a backup copy of my current site. Is there a duplicate or clone feature?"},
    {"queue": "Account Management", "priority": "medium",
     "subject": "Two-factor authentication not sending codes",
     "body": "I enabled 2FA on my account and now the SMS codes never arrive, so I can't finish logging in."},
    {"queue": "Design and Editor", "priority": "low",
     "subject": "How to add a countdown timer to my homepage",
     "body": "I'm launching a product next month and want a countdown timer on my homepage. Does Wix have a built-in element for that?"},
]


def load_tickets() -> list[dict]:
    return [{"ticket_id": f"T{idx:03d}", **t} for idx, t in enumerate(TICKETS, start=1)]


def write_tickets_jsonl() -> None:
    config.DATA_DIR.joinpath("tickets").mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / "tickets" / "synthetic_tickets.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ticket in load_tickets():
            f.write(json.dumps(ticket) + "\n")
    print(f"Wrote {len(TICKETS)} synthetic tickets to {path}")


if __name__ == "__main__":
    write_tickets_jsonl()
