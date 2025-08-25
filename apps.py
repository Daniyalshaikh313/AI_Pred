# app.py
import os
import re
import json
import pandas as pd
import plotly.express as px
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# ==========================
# Setup
# ==========================
load_dotenv("ev.env")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    st.error("Set GOOGLE_API_KEY in ev.env (GOOGLE_API_KEY=xxx)")
    st.stop()

genai.configure(api_key=API_KEY)
GEM_MODEL = genai.GenerativeModel("gemini-1.5-flash")

st.set_page_config(page_title="AI Data Analyst (Business-Ready)", layout="wide")
st.title("📊 AI-powered Data Analyst & Business Assistant APP")

# --------------------------
# Helpers
# --------------------------
def gemini_generate(prompt: str, max_tokens: int = 800) -> str:
    try:
        resp = GEM_MODEL.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.2},
        )
        return (resp.text or "").strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {e}"

def basic_eda(df):
    """Performs basic EDA and returns summary as dictionary"""
    try:
        describe_df = df.describe(include="all", datetime_is_numeric=True)
    except TypeError:
        describe_df = df.describe(include="all")

    missing_data = (
        (df.isnull().sum() / len(df) * 100)
        .round(2)
        .to_dict()
    )

    ed = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns.astype(str)),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": {col: int(val) for col, val in missing_data.items()},
        "describe_small": df.describe(include='all').fillna("").astype(str).to_dict()
    }
    return ed

def auto_plot_from_df(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) >= 2:
        return px.scatter(df, x=num_cols[0], y=num_cols[1], title=f"{num_cols[0]} vs {num_cols[1]}")
    if len(num_cols) == 1:
        return px.histogram(df, x=num_cols[0], title=f"Distribution of {num_cols[0]}")
    cat_cols = df.select_dtypes(exclude="number").columns
    if len(cat_cols) >= 1:
        tmp = df[cat_cols[0]].value_counts().reset_index()
        tmp.columns = [cat_cols[0], "count"]
        return px.bar(tmp, x=cat_cols[0], y="count", title=f"Category counts: {cat_cols[0]}")
    return None

# --------------------------
# Business-style insights
# --------------------------
def generate_professional_insights(df: pd.DataFrame):
    ed = basic_eda(df)
    prompt = f"""
You are a senior business analyst writing for executives (no technical jargon).
Use the dataset summary to produce crisp, decision-focused output.

DATASET SUMMARY (JSON):
{json.dumps(ed, ensure_ascii=False)}

Write:

### Executive Summary
- 3–4 sentences on size, coverage, and what the data enables

### Key Findings (5 bullets)
- Plain language, each bullet starts with a bold headline and a concrete metric or trend

### Risks & Data Quality
- Brief list of gaps, missing data, or caveats, with business impact

Keep it concise and professional.
"""
    return gemini_generate(prompt, max_tokens=900)

# --------------------------
# NL → Pandas (Safe) → Business Answer
# --------------------------
FORBIDDEN = re.compile(
    r"(import|open\(|__|os\.|sys\.|subprocess|eval\(|exec\(|pickle|pathlib|write|read\(|save|to_csv|to_excel|requests|urllib|pip|!|%|\.\s*to_parquet)",
    re.IGNORECASE,
)

def nl_to_pandas_code(df: pd.DataFrame, question: str) -> str:
    prompt = f"""
Convert the user's business question into **pure Pandas code** that runs on a
DataFrame called `df`. Return **only code**, no backticks, no prose.

Rules:
- df is already defined (the uploaded dataset)
- Use only pandas & built-ins; NO imports, file I/O, plotting, or network calls
- End by assigning the final result to a variable named: answer
- If the question is ambiguous, make a reasonable assumption and proceed
- If the user asks for top/bottom items without a metric, prefer the first numeric column
- Prefer readable code

User question: {question}
Columns: {list(df.columns)}
Dtypes: {df.dtypes.astype(str).to_dict()}
"""
    code = gemini_generate(prompt, max_tokens=600)
    code = code.strip().strip("```").replace("python", "")
    return code

def execute_pandas_code(df: pd.DataFrame, code: str):
    if FORBIDDEN.search(code):
        raise RuntimeError("Blocked unsafe code.")
    safe_globals = {"pd": pd}
    safe_locals = {"df": df}
    exec(code, safe_globals, safe_locals)  # noqa: S102
    if "answer" not in safe_locals:
        raise RuntimeError("Model did not set `answer` variable.")
    return safe_locals["answer"]

def summarize_for_business(question: str, result) -> str:
    if isinstance(result, pd.DataFrame):
        preview = result.head(20).to_csv(index=False)
    elif isinstance(result, (pd.Series,)):
        preview = result.head(20).to_csv(index=True)
    else:
        preview = str(result)
    prompt = f"""
You are a BI analyst speaking to business stakeholders.
Question: {question}

RESULT (CSV or text preview):
{preview}

Write:
- A plain-English, decision-ready answer in 3–6 sentences
- Include exact numbers and time frames where possible
- End with 2 short recommended actions
"""
    return gemini_generate(prompt, max_tokens=500)

# ==========================
# UI
# ==========================
uploaded = st.file_uploader("📂 Upload CSV or XLSX", type=["csv", "xlsx"])
if not uploaded:
    st.info("Upload a dataset to begin.")
    st.stop()

# ✅ Reset chat history when new dataset uploaded
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None
if "chat" not in st.session_state:
    st.session_state.chat = []

if st.session_state.last_uploaded != uploaded.name:
    st.session_state.chat = []   # clear history
    st.session_state.last_uploaded = uploaded.name

df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)

st.subheader("🔍 Data Preview")
st.dataframe(df.head(10), use_container_width=True)

with st.expander("Automatic EDA"):
    ed = basic_eda(df)
    st.write(f"Rows: **{ed['rows']}**, Columns: **{ed['cols']}**")
    st.json(ed["dtypes"])
    st.json(ed["missing"])

fig = auto_plot_from_df(df)
if fig:
    st.plotly_chart(fig, use_container_width=True)

if st.button("✨ Generate Executive Insights"):
    with st.spinner("Analyzing like a business analyst..."):
        insights = generate_professional_insights(df)
    st.markdown(insights)

st.markdown("---")

# ==========================
# 💬 Business Chatbot
# ==========================
st.subheader("💬 Dataset Chatbot (Business Answers)")

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

user_msg = st.chat_input("Ask a business question (e.g., 'Top 5 products by revenue last quarter')")
if user_msg:
    st.session_state.chat.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Working on your answer..."):
            try:
                code = nl_to_pandas_code(df, user_msg)
                st.code(code, language="python")

                result = execute_pandas_code(df, code)

                if isinstance(result, pd.DataFrame):
                    st.dataframe(result, use_container_width=True)
                else:
                    st.write(result)

                summary = summarize_for_business(user_msg, result)
                st.markdown(summary)

                st.session_state.chat.append(("assistant", summary))
            except Exception as e:
                err = f"Sorry — I couldn’t compute that safely. ({e})"
                st.error(err)
                st.session_state.chat.append(("assistant", err))















