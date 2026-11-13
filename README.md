# N2A Lab Lite

Interactive sandbox for exploring when an LLM agent should call a tool, when a call is unnecessary, and how cost and risk change the decision.

> **Scientific status:** v0.1 is a deterministic teaching simulator with hand-authored synthetic scenarios. It is not an LLM benchmark reproduction, does not extract hidden states, and does not establish that N2A outperforms existing methods.

## Run locally

```bash
pip install -r n2a_lite/requirements.txt
streamlit run n2a_lite/app.py
```

## Deploy for free

This branch is ready for Streamlit Community Cloud:

- Repository: `abdelbasethayam/ThinkGen`
- Branch: `n2a-lite`
- Entrypoint: `n2a_lite/app.py`

Open https://share.streamlit.io/, sign in with GitHub, create an app, enter the repository, branch, and entrypoint above, then deploy.

## What is included

- Five decision policies: no-tool, always-tool, confidence threshold, self-assessment, and N2A utility-aware.
- Four hand-authored scenarios.
- Controls for tool reliability, call cost, risk penalty, value of success, decision threshold, self-assessment bias, and authorization.
- Side-by-side results, execution traces, CSV export, and JSON trace export.
- GitHub Actions smoke check for Python syntax and dependencies.

## Research roadmap

1. Integrate official When2Tool benchmark data and reproduce its baselines.
2. Add a black-box LLM adapter with explicit key management and rate limits.
3. Add a white-box open-weight runner for hidden-state extraction and probe training.
4. Evaluate necessity, calibration, over/under-calling, and realized net utility on fixed benchmark splits.

The original `main` branch is unchanged; N2A Lite lives on the isolated `n2a-lite` branch.
