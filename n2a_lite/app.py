import json
import random
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

st.set_page_config(page_title="N2A Lab", page_icon="🧪", layout="wide")

# N2A Lite v0.1: deterministic teaching simulator, NOT an LLM benchmark.
CASES = {
    "Arithmetic": {
        "question": "Calculate 17 × 19.",
        "truth": "323", "tool": "calculator", "tool_result": "17 × 19 = 323",
        "no_tool": "323", "tool_answer": "323", "p0": 0.97, "p1": 0.995,
        "necessary": False,
        "why": "A capable model can solve this internally. The calculator can verify it, but is unnecessary here."
    },
    "Private record lookup": {
        "question": "Revenue was $8.0M in 2024 and $10.0M in 2025. What was year-over-year growth?",
        "truth": "25.0%", "tool": "company_facts_lookup",
        "tool_result": "2024 = $8.0M; 2025 = $10.0M; growth = (10 - 8) / 8 × 100 = 25.0%.",
        "no_tool": "I cannot reliably infer the figures without the records.",
        "tool_answer": "25.0%", "p0": 0.18, "p1": 0.98, "necessary": True,
        "why": "The answer depends on task-specific records that are not present in the prompt. Retrieval is required."
    },
    "Multi-step retrieval + calculation": {
        "question": "The dataset says 1,200 units were sold in Q1 and 1,500 in Q2. What was the percentage increase?",
        "truth": "25.0%", "tool": "sales_records_lookup + calculator",
        "tool_result": "Retrieved Q1 = 1,200 and Q2 = 1,500; increase = (1,500 - 1,200) / 1,200 × 100 = 25.0%.",
        "no_tool": "I do not have the dataset values available in the prompt context.",
        "tool_answer": "25.0%", "p0": 0.12, "p1": 0.96, "necessary": True,
        "why": "The agent must retrieve private task data and then calculate from it."
    },
    "Stable general knowledge": {
        "question": "What is the capital of France?",
        "truth": "Paris", "tool": "mock_search",
        "tool_result": "Local mock index: Paris is the capital of France.",
        "no_tool": "Paris", "tool_answer": "Paris", "p0": 0.99, "p1": 0.995,
        "necessary": False,
        "why": "This is stable common knowledge. Search can confirm it but adds little value."
    },
}

POLICIES = [
    "P0 · No tool",
    "P1 · Always tool",
    "P2 · Confidence threshold",
    "P3 · Self-assessment",
    "P6 · N2A utility-aware",
]

def execute_tool(case_name):
    """Execute one of the sandbox's local, read-only tools."""
    if case_name == "Arithmetic":
        result = 17 * 19
        return str(result), f"Calculator executed 17 × 19 = {result}."
    if case_name == "Private record lookup":
        records = {"2024_revenue_millions": 8.0, "2025_revenue_millions": 10.0}
        growth = (records["2025_revenue_millions"] - records["2024_revenue_millions"]) / records["2024_revenue_millions"] * 100
        return f"{growth:.1f}%", f"Lookup retrieved {records}; calculator computed growth = {growth:.1f}%."
    if case_name == "Multi-step retrieval + calculation":
        sales = {"Q1_units": 1200, "Q2_units": 1500}
        increase = (sales["Q2_units"] - sales["Q1_units"]) / sales["Q1_units"] * 100
        return f"{increase:.1f}%", f"Lookup retrieved {sales}; calculator computed increase = {increase:.1f}%."
    if case_name == "Stable general knowledge":
        mock_index = {"capital_of_france": "Paris"}
        return mock_index["capital_of_france"], "Mock search queried the local fixed index."
    raise ValueError("Unknown sandbox tool scenario.")


def tool_ok(case_name, reliability):
    if reliability >= 100:
        return True
    if reliability <= 0:
        return False
    seed = sum(ord(c) for c in case_name) + 9173
    return random.Random(seed).random() < reliability / 100

def run(policy, name, reliability, cost, risk, value, threshold, bias, authorized):
    c = CASES[name]
    p0 = c["p0"]
    p1 = p0 + (c["p1"] - p0) * reliability / 100
    gain = p1 - p0
    net = gain * value - cost - risk

    if policy.startswith("P0"):
        call, reason = False, "No-tool baseline."
    elif policy.startswith("P1"):
        call, reason = True, "Calls whenever the tool is available."
    elif policy.startswith("P2"):
        call = p0 < threshold
        reason = f"No-tool success estimate {p0:.2f} vs threshold {threshold:.2f}."
    elif policy.startswith("P3"):
        self_signal = min(1, max(0, 1 - p0 + bias))
        call = self_signal >= threshold
        reason = f"Self-assessment signal {self_signal:.2f}; not a calibrated probability."
    else:
        call = net > 0 and authorized
        reason = f"Estimated net utility {net:+.3f}; call only if positive and authorized."

    if call and not authorized:
        call, reason = False, reason + " Authorization gate blocked the call."

    trace = [
        ("Task", c["question"]),
        ("Necessity estimate", f"Estimated no-tool success = {p0:.2f}"),
        ("Utility estimate", f"Expected gain = {gain:+.3f}; estimated net utility = {net:+.3f}"),
        ("Action", ("CALL" if call else "NO-CALL") + " — " + reason),
    ]

    tool_success = None
    if call:
        tool_success = tool_ok(name, reliability)
        if tool_success:
            answer, tool_message = execute_tool(name)
            trace.append(("Tool", tool_message))
        else:
            answer = "Tool failed; no verified answer."
            trace.append(("Tool", "Tool returned an unreliable result."))
    else:
        answer = c["no_tool"]
        trace.append(("Tool", "No tool was invoked."))

    success = answer.strip() == c["truth"]
    unnecessary = call and not c["necessary"]
    missed = (not call) and c["necessary"]
    realized = (value if success else 0) - ((cost + risk) if call else 0)
    trace += [
        ("Outcome", ("SUCCESS" if success else "FAILURE") + f" — expected answer: {c['truth']}"),
        ("Realized utility", f"{realized:+.3f} = success value − call cost − risk penalty"),
    ]
    return {
        "policy": policy, "call": call, "success": success,
        "unnecessary": unnecessary, "missed": missed,
        "p0": p0, "p1": p1, "gain": gain, "net": net,
        "realized": realized, "answer": answer, "trace": trace,
    }

st.markdown("<span style='color:#16a34a;font-weight:700;letter-spacing:.14em'>PAPER RESEARCH PROJECT</span>", unsafe_allow_html=True)
st.title("🧪 N2A Lab")
st.subheader("Tool Necessity · Decision Bias · Calibration · Utility")
st.caption("N2A Lite v0.1 — interactive research sandbox")

st.warning(
    "Scientific status: this release uses hand-authored synthetic scenarios and a deterministic policy simulator. "
    "It is NOT an LLM benchmark reproduction, does NOT extract hidden states, and does NOT establish that N2A outperforms other methods."
)

with st.sidebar:
    st.header("Experiment controls")
    case_name = st.selectbox("Scenario", list(CASES))
    st.divider()
    reliability = st.slider("Tool reliability", 0, 100, 95, 5)
    cost = st.slider("Tool-call cost", 0.0, 1.0, 0.10, 0.05)
    risk = st.slider("Risk / privacy penalty", 0.0, 1.0, 0.05, 0.05)
    value = st.slider("Value of correct answer", 0.1, 2.0, 1.0, 0.1)
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.60, 0.05)
    bias = st.slider("Self-assessment bias", -0.5, 0.5, 0.10, 0.05)
    authorized = st.checkbox("Tool use is authorized", True)

case = CASES[case_name]
a, b = st.columns([1.5, 1])
with a:
    st.markdown("### Task")
    st.info(case["question"])
    st.markdown("**Available tool**")
    st.code(case["tool"])
with b:
    st.markdown("### Ground truth")
    st.metric("Expected answer", case["truth"])
    st.metric("Necessity label", "Necessary" if case["necessary"] else "Not necessary")
    st.caption(case["why"])

st.divider()
st.markdown("### Compare decision policies")
selected = st.multiselect("Policies", POLICIES, default=POLICIES)
if st.button("▶ Run experiment", type="primary", use_container_width=True, disabled=not selected):
    rows = [run(p, case_name, reliability, cost, risk, value, threshold, bias, authorized) for p in selected]
    st.session_state["n2a_rows"] = rows
    st.session_state["n2a_case"] = case_name
    st.session_state["n2a_config"] = {
        "tool_reliability": reliability, "tool_cost": cost, "risk_penalty": risk,
        "value_per_success": value, "threshold": threshold, "self_assessment_bias": bias,
        "authorized": authorized,
    }

if "n2a_rows" in st.session_state:
    rows = st.session_state["n2a_rows"]
    df = pd.DataFrame([{
        "Policy": r["policy"],
        "Tool call": "Yes" if r["call"] else "No",
        "Task success": "Yes" if r["success"] else "No",
        "Unnecessary call": int(r["unnecessary"]),
        "Missed necessary call": int(r["missed"]),
        "Expected gain": round(r["gain"], 3),
        "Estimated net utility": round(r["net"], 3),
        "Realized utility": round(r["realized"], 3),
    } for r in rows])

    st.markdown("### Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Successful tasks", f"{sum(r['success'] for r in rows)}/{len(rows)}")
    m2.metric("Tool calls", f"{sum(r['call'] for r in rows)}/{len(rows)}")
    m3.metric("Unnecessary calls", sum(r["unnecessary"] for r in rows))
    m4.metric("Missed necessary calls", sum(r["missed"] for r in rows))
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("#### Utility comparison")
    st.bar_chart(df.set_index("Policy")[["Realized utility", "Estimated net utility"]])

    st.markdown("#### Execution traces")
    for r in rows:
        with st.expander(f"{r['policy']} — {'CALL' if r['call'] else 'NO-CALL'} — {'SUCCESS' if r['success'] else 'FAILURE'}"):
            st.write("**Answer:** " + r["answer"])
            st.write(f"**P(no-tool success):** {r['p0']:.2f} · **P(with-tool success):** {r['p1']:.2f}")
            for stage, event in r["trace"]:
                st.markdown(f"- **{stage}:** {event}")

    export = {
        "app": "N2A Lite", "version": "0.1",
        "evidence_status": "SYNTHETIC_DEMO_NOT_BENCHMARK",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": st.session_state["n2a_case"],
        "config": st.session_state["n2a_config"],
        "results": rows,
    }
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       "n2a_lite_results.csv", "text/csv", use_container_width=True)
    c2.download_button("⬇ Download trace JSON", json.dumps(export, ensure_ascii=False, indent=2),
                       "n2a_lite_trace.json", "application/json", use_container_width=True)

with st.expander("How to interpret the experiment"):
    st.markdown("""
    - **Necessity:** whether this model/task can meet the stated success criterion without the tool.
    - **Utility:** whether the tool's incremental benefit justifies its cost and risk.
    - **Over-calling:** a tool call in a scenario labeled not necessary.
    - **Under-calling:** failing to call a tool in a scenario labeled necessary.
    - **Self-assessment:** a behavioral signal, not ground truth or calibrated probability.
    - **Estimated net utility** is the policy estimate; **realized utility** is computed from the simulator outcome.

    The scenarios are hand-authored. Replace them with official benchmark splits and independently verified model rollouts before making research claims.
    """)

with st.expander("Research roadmap"):
    st.markdown("""
    1. v0.1 — synthetic policy sandbox (this release).
    2. v0.2 — official When2Tool dataset integration and baseline reproduction.
    3. v0.3 — real black-box LLM adapter with safe key management and rate limits.
    4. v0.4 — open-weight white-box runner for hidden-state extraction and probe training.
    5. v1.0 — reproducible evaluation with pinned models, datasets, prompts, traces, and statistical analysis.
    """)

st.caption("PAPER · N2A Lab · v0.1 · Synthetic demo only")
