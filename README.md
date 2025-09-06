# AI-Powered Email Assistant — Unstop Challenge

## What this project does
- Loads a support emails CSV
- Filters support-related emails
- Detects sentiment (Hugging Face)
- Detects urgency using keywords
- Generates a simple AI draft reply
- CLI demo + Streamlit dashboard demo

## How to run
1. Create virtualenv and activate:
   python3 -m venv venv
   source venv/bin/activate

2. Install:
   pip install -r requirements.txt

3. CLI:
   python coding_challenge.py --csv data/Sample_Support_Emails_Dataset(1).csv

4. Dashboard:
   streamlit run dashboard.py

## Design notes / extension ideas
- Replace rule-based replies with OpenAI + RAG to include product FAQs
- Integrate Gmail/Outlook APIs for live emails
- Store processed emails in DB and add authentication to dashboard
