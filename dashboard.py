# dashboard.py
import streamlit as st
import pandas as pd
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

def is_support_email(subject, body):
    keywords = ["support", "query", "request", "help", "issue", "complaint"]
    s = (str(subject) + " " + str(body)).lower()
    return any(k in s for k in keywords)

def is_urgent(text):
    urgent_keywords = ["urgent", "immediately", "asap", "critical", "cannot", "error", "failed"]
    return any(k in str(text).lower() for k in urgent_keywords)

def generate_reply(subject, body, sentiment):
    if sentiment.upper().startswith("NEG"):
        return f"Dear Customer,\n\nWe're sorry to hear about '{subject}'. Our team is working on it.\n\nBest,\nSupport"
    if sentiment.upper().startswith("POS"):
        return f"Dear Customer,\n\nThanks for your message about '{subject}'. Happy to help!\n\nBest,\nSupport"
    return f"Dear Customer,\n\nThanks for contacting us regarding '{subject}'. We'll get back shortly.\n\nBest,\nSupport"

st.title("📧 AI-Powered Email Assistant (Demo)")

uploaded = st.file_uploader("Upload Support CSV (or use default data file)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("data/Sample_Support_Emails_Dataset(1).csv")

sender_col = find_column(df, ["sender", "from", "email"])
subject_col = find_column(df, ["subject", "title"])
body_col = find_column(df, ["body", "message", "content"])
date_col = find_column(df, ["date", "timestamp", "time"])
if body_col is None:
    df["body_fallback"] = df[subject_col].astype(str) if subject_col else ""
    body_col = "body_fallback"

df["is_support"] = df.apply(lambda r: is_support_email(r.get(subject_col, ""), r.get(body_col, "")), axis=1)
df_support = df[df["is_support"]].copy()
if df_support.empty:
    st.warning("No support-like emails found.")
    st.stop()

sentiment_model = load_model()
results = []
for _, r in df_support.iterrows():
    subj = r.get(subject_col, "")
    body = r.get(body_col, "")
    txt = f"{subj} {body}"
    sres = sentiment_model(txt)[0]
    label = sres["label"]
    urgent = "Yes" if is_urgent(txt) else "No"
    reply = generate_reply(subj, body, label)
    results.append({
        "Sender": r.get(sender_col, "unknown"),
        "Subject": subj,
        "Body": body,
        "Date": r.get(date_col, ""),
        "Sentiment": label,
        "Urgent": urgent,
        "Reply": reply
    })

df_res = pd.DataFrame(results)

st.sidebar.header("View Options")
show_urgent = st.sidebar.checkbox("Only urgent", False)
if show_urgent:
    df_res = df_res[df_res["Urgent"] == "Yes"]

st.metric("Total Support Emails", len(df_support))
st.metric("Urgent Emails", len(df_res[df_res["Urgent"]=="Yes"]))

st.subheader("Emails")
st.dataframe(df_res[["Sender","Subject","Sentiment","Urgent","Date"]], use_container_width=True)

st.subheader("Select an email to view/edit reply")
selected = st.selectbox("Choose subject", df_res["Subject"].tolist())
sel_row = df_res[df_res["Subject"] == selected].iloc[0]
st.write("**From:**", sel_row["Sender"])
st.write("**Date:**", sel_row["Date"])
st.write("**Body:**", sel_row["Body"])
edited = st.text_area("Draft Reply (editable)", sel_row["Reply"], height=200)
if st.button("Mark Resolved (demo)"):
    st.success("Marked resolved (demo). In production you'd update DB / send email.")
