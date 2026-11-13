---
title: ThinkGen
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: src/streamlit_app.py
pinned: false
---

# ThinkGen — Adaptive Socratic Learning Assistant

**Group 45 | CSAI 810**

## Setup
1. Create a Groq API key from https://console.groq.com
2. On Hugging Face Spaces: Settings → Repository secrets → add `GROQ_API_KEY`
3. That's it!

## How it works
- Student selects a GenAI concept
- Model asks 7 progressive Socratic questions
- Each answer is evaluated strictly (gibberish = score 1 automatically)
- Progress bar updates with weighted scoring (harder questions worth more)
- Conceptual gap explanation appears with every answer
- Final report summarizes knowledge gaps and next steps
