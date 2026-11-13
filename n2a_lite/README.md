# N2A Lab Lite

A free-to-host Streamlit prototype for exploring tool necessity, over-calling, under-calling, and utility-aware decisions.

## Scientific status

**v0.1 is a deterministic teaching simulator using hand-authored synthetic scenarios.** It is not an LLM benchmark reproduction, does not extract hidden states, and does not establish that N2A outperforms other methods.

## Deploy free with Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select repository: abdelbasethayam/ThinkGen
3. Select branch: n2a-lite
4. Set app file: n2a_lite/app.py
5. Deploy.

Streamlit Community Cloud hosts the app for free, subject to its published resource limits.

## Features

- Five policies: no-tool, always-tool, confidence threshold, self-assessment, N2A utility-aware.
- Four hand-authored scenarios.
- Tool reliability, cost, risk, value, threshold, bias, and authorization controls.
- Policy comparison, execution traces, CSV and JSON export.

## Next research steps

- Import official When2Tool splits and reproduce its baselines.
- Add a black-box LLM adapter with safe key management and rate limits.
- Add a white-box runner for open-weight models, hidden-state extraction, and probe training.
- Record pinned dataset/model revisions, prompts, seeds, tool versions, and raw traces.
