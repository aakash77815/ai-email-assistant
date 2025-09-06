# coding_challenge.py
import argparse
import pandas as pd
from transformers import pipeline

# ---------- Utilities ----------
def find_column(df, candidates):
    """Return the first column name from candidates that exists in df, else None"""
    for c in candidates:
        if c in df.columns:
            return c
        # case-insensitive match
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

def load_emails(csv_path):
    df = pd.read_csv(csv_path)
    # find likely columns
    sender_col = find_column(df, ["sender", "from", "email"])
    subject_col = find_column(df, ["subject", "title", "heading"])
    body_col = find_column(df, ["body", "message", "content", "text"])
    date_col = find_column(df, ["date", "timestamp", "time"])
    # fallback: if no body, try using subject as body
    if body_col is None:
        df["body_fallback"] = df[subject_col].astype(str) if subject_col else ""
        body_col = "body_fallback"
    return df, sender_col, subject_col, body_col, date_col

# ---------- Filtering + NLP ----------
SUPPORT_KEYWORDS = ["support", "query", "request", "help", "issue", "complaint"]

def is_support_email(subject, body):
    s = (str(subject) + " " + str(body)).lower()
    return any(k in s for k in SUPPORT_KEYWORDS)

def is_urgent(text):
    urgent_keywords = ["urgent", "immediately", "asap", "critical", "cannot", "can't", "cannot access", "error", "failed"]
    t = str(text).lower()
    return any(k in t for k in urgent_keywords)

def generate_reply(subject, body, sentiment_label):
    # Simple rule-based reply using sentiment
    subj = subject if subject else "your request"
    if sentiment_label.upper().startswith("NEG"):
        return (
            f"Dear Customer,\n\n"
            f"We're very sorry to hear about the issue regarding '{subj}'. "
            "Our team is investigating this right away and we will update you as soon as possible.\n\n"
            "Best regards,\nSupport Team"
        )
    elif sentiment_label.upper().startswith("POS"):
        return (
            f"Dear Customer,\n\n"
            f"Thank you for your message about '{subj}'. We appreciate the feedback and are happy to help.\n\n"
            "Best regards,\nSupport Team"
        )
    else:
        return (
            f"Dear Customer,\n\n"
            f"Thank you for contacting us regarding '{subj}'. We will review and get back to you shortly.\n\n"
            "Best regards,\nSupport Team"
        )

# ---------- Main ----------
def main(csv_path):
    print("Loading dataset:", csv_path)
    df, sender_col, subject_col, body_col, date_col = load_emails(csv_path)

    if subject_col is None and body_col is None:
        print("Could not find subject or body columns in CSV. Columns found:", list(df.columns))
        return

    # Filter support-related emails
    df["is_support"] = df.apply(lambda r: is_support_email(r.get(subject_col, ""), r.get(body_col, "")), axis=1)
    df_filtered = df[df["is_support"]].copy()
    if df_filtered.empty:
        print("No support-related emails found with keywords.")
        return

    # Load sentiment pipeline (this will download first time)
    print("Loading sentiment model (may take a moment on first run)...")
    sentiment = pipeline("sentiment-analysis")

    # Process and print
    for idx, row in df_filtered.iterrows():
        subj = row.get(subject_col, "")
        body = row.get(body_col, "")
        txt = f"{subj} {body}"
        sres = sentiment(txt)[0]  # {'label': 'NEGATIVE', 'score': 0.99}
        label, score = sres["label"], sres.get("score", 0.0)
        urgent = "Yes" if is_urgent(txt) else "No"
        reply = generate_reply(subj, body, label)

        sender = row.get(sender_col, "unknown") if sender_col else "unknown"
        date = row.get(date_col, "")

        print("="*80)
        print(f"From    : {sender}")
        print(f"Date    : {date}")
        print(f"Subject : {subj}")
        print(f"Body    : {str(body)[:200]}{'...' if len(str(body))>200 else ''}")
        print(f"Sentiment: {label} (conf {score:.2f})")
        print(f"Urgent   : {urgent}")
        print("\nAI Draft Reply:\n")
        print(reply)
        print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/Sample_Support_Emails_Dataset(1).csv", help="Path to CSV file")
    args = parser.parse_args()
    main(args.csv)
