import os, re, json
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
import gradio as gr

# ----------------------------
# Storage (HF Space)
# ----------------------------
REPORTS = "./reports"
LOGS = "./logs"
os.makedirs(REPORTS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# SEC-konformer User-Agent als Secret setzen:
# Settings → Variables and secrets → SEC_USER_AGENT
SEC_UA = os.getenv("SEC_USER_AGENT", "BuchhaltungAI/0.1 (contact: missing)")
HEADERS = {"User-Agent": SEC_UA}

# ----------------------------
# MVP: Apple (Demo). Next: beliebige Firma via EDGAR-Suche
# ----------------------------
APPLE_10K = {
    "2024": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
    "2023": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
    "2022": "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220924.htm",
}
SUPPORTED = {"apple": APPLE_10K, "aapl": APPLE_10K, "apple inc": APPLE_10K}

# ----------------------------
# Fetch + Extract
# ----------------------------
def fetch_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def pick_balance_sheet_table(html: str):
    tables = pd.read_html(StringIO(html))
    best, best_score = None, -1
    for t in tables:
        s = t.astype(str).apply(lambda col: col.str.lower())
        joined = " ".join(s.fillna("").values.flatten().tolist())
        score = sum(k in joined for k in [
            "total assets", "total liabilities",
            "shareholders' equity", "stockholders' equity",
            "total stockholders", "total shareholders",
        ])
        if score > best_score:
            best, best_score = t, score
    return best, best_score

def clean_money(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("$", "").replace(",", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = "".join(c for c in s if c.isdigit() or c in ".-")
    if s in ("", "-", "--"):
        return np.nan
    v = float(s)
    return -v if neg else v

def extract_totals(balance_table: pd.DataFrame) -> dict:
    if balance_table is None or balance_table.empty:
        return {}
    t = balance_table.copy()
    t.columns = [str(c) for c in t.columns]
    label_col = t.columns[0]
    t[label_col] = t[label_col].astype(str).str.lower()

    def find_row_contains(*keywords):
        mask = False
        for kw in keywords:
            mask = mask | t[label_col].str.contains(kw, na=False)
        if mask.any():
            return t[mask].iloc[0]
        return None

    row_assets = find_row_contains("total assets")
    row_liab   = find_row_contains("total liabilities")
    row_eq     = find_row_contains(
        "total shareholders", "shareholders' equity",
        "total stockholders", "stockholders' equity"
    )

    vals = {}
    for c in t.columns[1:]:
        vals[c] = {
            "total_assets": clean_money(row_assets[c]) if row_assets is not None else np.nan,
            "total_liabilities": clean_money(row_liab[c]) if row_liab is not None else np.nan,
            "total_equity": clean_money(row_eq[c]) if row_eq is not None else np.nan,
        }
    return vals

def analyze_rows(rows):
    df = pd.DataFrame(rows).sort_values("fiscal_year")
    df["equity_ratio"] = df["total_equity"] / df["total_assets"]
    df["debt_to_equity"] = df["total_liabilities"] / df["total_equity"]
    df["assets_yoy_pct"] = df["total_assets"].pct_change() * 100
    df["equity_yoy_pct"] = df["total_equity"].pct_change() * 100

    flags = []
    if len(df) >= 2:
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if (last["total_equity"] < prev["total_equity"]) and (last["total_liabilities"] > prev["total_liabilities"]):
            flags.append("⚠️ Equity ↓ und Liabilities ↑ (möglicher Leverage-/Risikoanstieg)")
    return df, flags

# ----------------------------
# Dialog / State
# ----------------------------
def init_state():
    return {"pending_intent": None, "companies": None, "years": None}

def detect_intent(text: str):
    t = text.lower()
    if ("analys" in t) and ("bilanz" in t or "balance sheet" in t):
        return "ANALYZE_BALANCE"
    return None

def parse_companies(text: str):
    return [p.strip() for p in text.split(",") if p.strip()]

def parse_years(text: str):
    years = re.findall(r"(20\d{2})", text)
    return sorted(set(years))

def company_urls(company: str, years):
    key = company.lower().strip()
    if key in SUPPORTED:
        src = SUPPORTED[key]
        return {y: src[y] for y in years if y in src}
    return {}

def save_report(company, df, flags):
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", company.strip().lower())
    csv_path = f"{REPORTS}/{safe}_{stamp}.csv"
    md_path  = f"{REPORTS}/{safe}_{stamp}.md"

    df.to_csv(csv_path, index=False)

    md = []
    md.append(f"# Bilanzvergleich – {company}")
    md.append(f"Zeit: {stamp}")
    md.append("")
    md.append("## Kennzahlen & Totals")
    md.append(df.to_markdown(index=False))
    md.append("")
    md.append("## Flags")
    md.append("\n".join(flags) if flags else "Keine Flags ausgelöst.")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return csv_path, md_path

def log_turn(state, user_text, assistant_text, extra=None):
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = f"{LOGS}/chatlog_{stamp}.json"
    payload = {
        "time": stamp,
        "state": state,
        "user": user_text,
        "assistant": assistant_text,
        "extra": extra or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

def handle_message(message, history, state):
    text = (message or "").strip()
    history = history or []

    if not text:
        return history, state

    if state["pending_intent"] == "ANALYZE_BALANCE":
        if state["companies"] is None:
            comps = parse_companies(text)
            if not comps:
                reply = "Welche Firmen? (z. B. `Apple` oder `Apple, Microsoft`)"
                history.append((message, reply))
                log_turn(state, message, reply)
                return history, state
            state["companies"] = comps
            reply = "Welche Jahre? (z. B. `2022 2023 2024`)"
            history.append((message, reply))
            log_turn(state, message, reply)
            return history, state

        if state["years"] is None:
            years = parse_years(text)
            if not years:
                reply = "Welche Jahre? (z. B. `2022 2023 2024`)"
                history.append((message, reply))
                log_turn(state, message, reply)
                return history, state
            state["years"] = years

            results_texts = []
            extra = {"reports": []}

            for comp in state["companies"]:
                urls = company_urls(comp, state["years"])
                if not urls:
                    results_texts.append(f"**{comp}:** MVP unterstützt aktuell nur Apple/AAPL (SEC).")
                    continue

                rows = []
                for y, url in urls.items():
                    html = fetch_html(url)
                    bs, _ = pick_balance_sheet_table(html)
                    totals = extract_totals(bs)
                    if not totals:
                        rows.append({"fiscal_year": y, "total_assets": np.nan, "total_liabilities": np.nan, "total_equity": np.nan})
                        continue
                    best_col = max(totals, key=lambda c: sum(pd.notna(v) for v in totals[c].values()))
                    d = totals[best_col]
                    rows.append({"fiscal_year": y, **d})

                df, flags = analyze_rows(rows)
                csv_path, md_path = save_report(comp, df, flags)

                results_texts.append(
                    f"## {comp}\n"
                    f"- gespeichert im Space:\n"
                    f"  - CSV: `{csv_path}`\n"
                    f"  - Report: `{md_path}`\n"
                    f"- Flags: " + ("; ".join(flags) if flags else "keine")
                )
                extra["reports"].append({"company": comp, "csv": csv_path, "md": md_path})

            reply = "\n\n".join(results_texts) if results_texts else "Keine Ergebnisse."
            history.append((message, reply))

            state = init_state()
            log_turn(state, message, reply, extra=extra)
            return history, state

    intent = detect_intent(text)
    if intent == "ANALYZE_BALANCE":
        state["pending_intent"] = "ANALYZE_BALANCE"
        reply = "Welche Firmen soll ich analysieren? (z. B. `Apple` oder `Apple, Microsoft`)"
        history.append((message, reply))
        log_turn(state, message, reply)
        return history, state

    reply = "Bereit ✅ Tipp z. B. `Analysiere die Bilanzen`."
    history.append((message, reply))
    log_turn(state, message, reply)
    return history, state

# ----------------------------
# UI
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        "## 💬 BuchhaltungAI – Bilanz-Chat (MVP)\n"
        "Beispiel:\n"
        "1) **Analysiere die Bilanzen**\n"
        "2) **Apple**\n"
        "3) **2022 2023 2024**\n"
    )

    chatbot = gr.Chatbot(height=380, allow_tags=False)
    chatbot.value = [("System", "Bereit ✅ Sag: **Analysiere die Bilanzen**")]

    msg = gr.Textbox(label="Deine Anfrage", placeholder="z. B. Analysiere die Bilanzen")
    state = gr.State(init_state())

    def on_send(message, history, state):
        history, state = handle_message(message, history, state)
        return history, state

    msg.submit(on_send, [msg, chatbot, state], [chatbot, state]).then(lambda: "", None, msg)

demo.launch()
