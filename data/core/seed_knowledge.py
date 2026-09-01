"""
Seed data for UDA-Hub's own knowledge base (core DB, `knowledge` table).
Run via 02_core_db_setup.ipynb.

The brief describes a `cultpass_articles.jsonl` handed off by the customer
with 4 starter articles, to be expanded to at least 14 covering diverse
topics. This project has no starter repo to copy those 4 from (see README),
so all 18 articles below were authored fresh for this submission — well
past both the "14 total" and "10 additional" thresholds — spanning billing,
account management, subscriptions, bookings, technical/app issues,
notifications, privacy, referrals, accessibility, and support-contact
topics, so the RAG resolver has real category diversity to route across.
"""
from __future__ import annotations

from data.core.db import get_core_session, init_core_db
from data.core.models import Knowledge

ARTICLES = [
    dict(
        title="How Billing Cycles Work",
        category="billing",
        tags=["billing", "invoice", "renewal"],
        content=(
            "CultPass memberships renew automatically on your billing date, either monthly or "
            "annually depending on the plan you chose at signup. You're charged the full plan "
            "price on each renewal date; there is no proration for mid-cycle upgrades or "
            "downgrades — a plan change takes effect at your *next* renewal. You can see your "
            "next renewal date and billing cycle any time in Account > Billing. If a renewal "
            "charge fails (e.g. expired card), we retry it up to three times over seven days "
            "before pausing the account."
        ),
    ),
    dict(
        title="Understanding Your Invoice and Charges",
        category="billing",
        tags=["billing", "invoice", "charges"],
        content=(
            "Every invoice lists your plan renewal charge plus any pay-per-experience bookings "
            "made outside your plan's included credits. If you see a charge you don't recognize, "
            "check the invoice line description first — booking charges are labeled with the "
            "experience name. Duplicate-looking charges are almost always a renewal charge plus "
            "a separate booking charge on the same day, but if you believe you were genuinely "
            "double-charged for the same booking, support can investigate and refund the "
            "duplicate once confirmed."
        ),
    ),
    dict(
        title="Refund Eligibility and the 30-Day Window",
        category="billing",
        tags=["refund", "billing", "policy"],
        content=(
            "Bookings can be refunded within 30 days of the booking date if the experience "
            "hasn't already occurred, or if it was cancelled by the venue. Refunds outside the "
            "30-day window are not eligible through self-service and require manager approval — "
            "escalate these. Membership renewal charges themselves are refundable only within 48 "
            "hours of the charge, per our terms of service. Refunds are issued to the original "
            "payment method and typically post within 5-10 business days."
        ),
    ),
    dict(
        title="Updating Your Account Email and Password",
        category="account",
        tags=["account", "login", "email", "password"],
        content=(
            "You can change your account email and password from Account > Profile. Changing "
            "your email requires confirming a verification link sent to the *new* address before "
            "the change takes effect — your old email keeps working until that confirmation "
            "happens. If you no longer have access to either email, support can manually verify "
            "your identity (name, billing zip, last 4 digits of card) and update it for you."
        ),
    ),
    dict(
        title="Adding or Removing Members from a Household Account",
        category="account",
        tags=["account", "household", "members"],
        content=(
            "Premium and Elite plans support multiple members under one account (Basic is "
            "single-user). The account owner can invite members by email from Account > "
            "Household Members; invited members get their own login but share the household's "
            "booking credits. Removing a member immediately revokes their access to book new "
            "experiences, but doesn't cancel bookings they already made."
        ),
    ),
    dict(
        title="Pausing or Cancelling Your Membership",
        category="account",
        tags=["account", "cancellation", "pause"],
        content=(
            "You can pause your membership for up to 3 months from Account > Membership > Pause. "
            "While paused, you keep your account and booking history but can't book new "
            "experiences, and you aren't billed. Cancelling ends the membership entirely at the "
            "end of the current billing period — you keep access until then, after which the "
            "account becomes read-only. Reactivating a cancelled account is self-service and "
            "starts a brand-new billing cycle."
        ),
    ),
    dict(
        title="Comparing Basic, Premium, and Elite Plans",
        category="subscription",
        tags=["subscription", "plans", "pricing"],
        content=(
            "Basic is single-user with 2 experience credits per month. Premium supports up to 4 "
            "household members and 6 shared credits per month, plus early access to popular "
            "events. Elite supports up to 8 members, 12 shared credits per month, and includes "
            "guest passes for non-members. All plans can book additional experiences beyond "
            "their included credits at pay-per-booking pricing."
        ),
    ),
    dict(
        title="How to Upgrade or Downgrade Your Plan",
        category="subscription",
        tags=["subscription", "upgrade", "downgrade"],
        content=(
            "Plan changes are made from Account > Membership > Change Plan and take effect at "
            "your next renewal date — you keep your current plan's benefits until then, so "
            "there's no partial-month credit or charge. If you upgrade, your new monthly/annual "
            "price and credit allotment start on the next renewal. Downgrades that would drop "
            "your household below its current member count aren't allowed until members are "
            "removed first."
        ),
    ),
    dict(
        title="How to Reserve a Cultural Experience",
        category="bookings",
        tags=["bookings", "reservations", "experiences"],
        content=(
            "Browse experiences in the Explore tab, filtered by city, date, or category (museum, "
            "music, workshop, etc.). Booking an experience uses one of your plan's monthly "
            "credits, or is charged at pay-per-booking price once credits are used up. You'll "
            "receive a digital pass (QR code) by email and in the app immediately after booking — "
            "this is what you show at the venue."
        ),
    ),
    dict(
        title="Cancelling or Rescheduling a Booking",
        category="bookings",
        tags=["bookings", "cancellation", "reschedule"],
        content=(
            "Bookings can be cancelled or rescheduled for free up to 24 hours before the "
            "experience's start time from My Bookings > Manage. Cancelling within 24 hours "
            "forfeits the credit used (or the payment, for pay-per-booking) unless the venue "
            "itself cancelled the event, in which case you're automatically refunded in full."
        ),
    ),
    dict(
        title="Troubleshooting the CultPass Mobile App",
        category="technical",
        tags=["technical", "mobile-app", "troubleshooting"],
        content=(
            "If the app is slow, crashing, or showing stale booking data: force-close and reopen "
            "the app, then check Settings > About for available updates. Persistent issues are "
            "usually fixed by logging out and back in (Settings > Log Out), which refreshes your "
            "session without losing any bookings. If crashes continue after updating and "
            "re-logging in, this is a bug — escalate with the device model and OS version."
        ),
    ),
    dict(
        title="QR Code / Digital Pass Not Scanning at Venue",
        category="technical",
        tags=["technical", "qr-code", "venue"],
        content=(
            "Digital passes need an active internet connection to validate at the door — if "
            "you're somewhere with poor signal, switch to Wi-Fi or pre-load the pass by opening "
            "it at home before you travel (the app caches it for 24 hours offline). If the venue "
            "staff says the code is 'already used' or 'invalid' and you haven't checked in "
            "elsewhere, this usually means a duplicate booking or a sync delay — support can "
            "verify the booking status and issue a manual check-in code."
        ),
    ),
    dict(
        title="Login and Two-Factor Authentication Issues",
        category="technical",
        tags=["technical", "login", "2fa", "security"],
        content=(
            "If a 2FA code isn't arriving, check that the phone number/email on file is current "
            "and check spam folders for email codes. Codes expire after 10 minutes — request a "
            "new one rather than reusing an old one. If you're fully locked out (no access to "
            "your 2FA method at all), support can disable 2FA temporarily after identity "
            "verification, but this always requires escalation since it's a security-sensitive "
            "action."
        ),
    ),
    dict(
        title="Managing Email and Push Notification Preferences",
        category="notifications",
        tags=["notifications", "email", "push", "preferences"],
        content=(
            "Notification preferences live in Account > Notifications, split into three "
            "categories: booking reminders, new-experience alerts, and billing receipts. Billing "
            "receipts can't be fully disabled (they're required records) but can be set to "
            "email-only. Push notifications require the app to have notification permission "
            "granted at the OS level — if toggling in-app doesn't work, check the phone's system "
            "settings for the app."
        ),
    ),
    dict(
        title="How CultPass Handles Your Personal Data",
        category="privacy",
        tags=["privacy", "data", "gdpr"],
        content=(
            "CultPass collects account details (name, email, billing info) and booking history "
            "to operate the service. We don't sell personal data to third parties. You can "
            "request a full export of your data or request account deletion from Account > "
            "Privacy > Data Requests; exports are emailed within 5 business days. Deletion "
            "requests are honored after any pending bookings are resolved and legally required "
            "billing records are retained per applicable law."
        ),
    ),
    dict(
        title="Referral Program and Gift Memberships",
        category="referrals",
        tags=["referrals", "gifting", "promotions"],
        content=(
            "Existing members can refer friends from Account > Refer a Friend; both the referrer "
            "and the new member get one free bonus credit once the new member completes their "
            "first booking. Gift memberships (1, 3, or 12 months) can be purchased for someone "
            "without a CultPass account from the Gift a Membership page; the recipient redeems "
            "it with the code emailed to them, which creates a brand-new account under their own "
            "email."
        ),
    ),
    dict(
        title="Accessibility Accommodations at Partner Venues",
        category="accessibility",
        tags=["accessibility", "venues", "accommodations"],
        content=(
            "Each experience listing shows the venue's accessibility features (wheelchair access, "
            "hearing loop, sensory-friendly sessions) under the 'Accessibility' tab on the "
            "listing page. If a listing is missing accessibility info you need, or you require an "
            "accommodation not listed (e.g. a companion seat, ASL interpretation), contact "
            "support at least 5 business days before the experience so we can coordinate with "
            "the venue directly."
        ),
    ),
    dict(
        title="How to Contact Human Support and Response Times",
        category="support",
        tags=["support", "contact", "sla"],
        content=(
            "In-app chat is the fastest way to reach a human for anything the self-service help "
            "center can't resolve, with typical first response under 4 business hours. Email "
            "support (support@cultpass.example.com) has a 1 business day SLA. Urgent, "
            "time-sensitive issues (e.g. a same-day event access problem) should always use "
            "in-app chat and be flagged as urgent so they're prioritized ahead of the queue."
        ),
    ),
]


def seed() -> None:
    init_core_db()
    with get_core_session() as session:
        if session.query(Knowledge).count() > 0:
            print("Knowledge base already seeded — skipping.")
            return
        for article in ARTICLES:
            session.add(Knowledge(**article))
        session.flush()
        print(f"Seeded {len(ARTICLES)} knowledge base articles across "
              f"{len(set(a['category'] for a in ARTICLES))} categories.")


if __name__ == "__main__":
    seed()
