import streamlit as st
import streamlit.components.v1 as components
import json
import re
from groq import Groq

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL = "llama-3.3-70b-versatile"

WEIGHT_MAPS = {
    5:  [1, 1, 2, 3, 4],
    6:  [1, 1, 2, 2, 3, 4],
    7:  [1, 1, 1, 2, 2, 3, 4],
    8:  [1, 1, 1, 2, 2, 3, 3, 4],
    9:  [1, 1, 1, 2, 2, 2, 3, 3, 4],
    10: [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
    12: [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
}

CONCEPTS = {
    "🔄 Transformers": {
        "desc": "Transformer architecture and how it works",
        "key_ideas": ["encoder / decoder", "self-attention", "positional encoding", "feed-forward sublayers"],
        "difficulty": "Intermediate",
        "questions": 10,
    },
    "👁️ Attention Mechanism": {
        "desc": "Self-Attention and Multi-Head Attention",
        "key_ideas": ["query / key / value", "scaled dot-product", "multi-head parallelism", "attention scores"],
        "difficulty": "Advanced",
        "questions": 10,
    },
    "✍️ Prompt Engineering": {
        "desc": "Prompt design and its effect on outputs",
        "key_ideas": ["few-shot examples", "chain-of-thought", "system vs user roles", "temperature effect"],
        "difficulty": "Foundational",
        "questions": 7,
    },
    "🎯 Fine-tuning": {
        "desc": "Adapting models on custom datasets",
        "key_ideas": ["pre-training vs fine-tuning", "catastrophic forgetting", "LoRA / PEFT", "task-specific data"],
        "difficulty": "Intermediate",
        "questions": 9,
    },
    "📚 RAG": {
        "desc": "Retrieval Augmented Generation",
        "key_ideas": ["vector store", "retrieval pipeline", "context injection", "hallucination reduction"],
        "difficulty": "Applied",
        "questions": 8,
    },
    "🎨 Diffusion Models": {
        "desc": "Diffusion-based image generation models",
        "key_ideas": ["forward noise process", "denoising U-Net", "DDPM / DDIM", "classifier-free guidance"],
        "difficulty": "Advanced",
        "questions": 10,
    },
    "🧮 Embeddings": {
        "desc": "Mathematical representation of meaning",
        "key_ideas": ["vector space", "cosine similarity", "semantic clustering", "dimensionality"],
        "difficulty": "Foundational",
        "questions": 7,
    },
    "🌀 Hallucination": {
        "desc": "The hallucination phenomenon in LLMs",
        "key_ideas": ["training data gaps", "overconfident decoding", "grounding vs generation", "mitigation strategies"],
        "difficulty": "Applied",
        "questions": 8,
    },
    "🏆 RLHF": {
        "desc": "Reinforcement Learning from Human Feedback",
        "key_ideas": ["reward model", "PPO fine-tuning", "preference data", "alignment tax"],
        "difficulty": "Advanced",
        "questions": 10,
    },
    "✂️ Tokenization": {
        "desc": "Splitting text into tokens",
        "key_ideas": ["BPE / WordPiece", "vocabulary size", "out-of-vocabulary handling", "token vs word"],
        "difficulty": "Foundational",
        "questions": 7,
    },
}

# ─── NEW HARMONIOUS COLOR PALETTE ─────────────────────────────────────────────
# Primary:   #5B6EF5  (indigo-blue)
# Secondary: #1ABFA3  (teal-green)
# Warning:   #F5A524  (amber)
# Danger:    #EF4B5E  (coral-red)
# Neutral:   #6B7A90  (slate)
# Difficulty badge colors — all from the same palette family
DIFF_CLR = {
    "Foundational": "#1ABFA3",   # teal
    "Intermediate": "#5B6EF5",   # indigo
    "Applied":      "#A07AF5",   # violet
    "Advanced":     "#EF4B5E",   # coral-red
}

SYSTEM_PROMPT = """You are ThinkGen, a strict Socratic AI tutor for Generative AI concepts.
STRICT SCORING RULES — follow exactly, no exceptions:
- Score 1: Gibberish, random letters, "idk", empty meaning, irrelevant text
- Score 2: Mostly wrong or extremely vague — just buzzwords, no real understanding
- Score 3: Partially correct but missing key mechanism or causal logic
- Score 4: Mostly correct, minor gap or imprecision
- Score 5: Accurate, clear, demonstrates real understanding of how/why
WARNING: If you give score 4 or 5 to a wrong answer, you fail your job. Be strict.
ADAPTIVE STRATEGY — calibrate to current mastery level "{mastery_label}" (avg {mastery_avg:.1f}/5):
- Beginner / Developing (avg <= 2.6): Simplify completely. Use everyday analogies, smallest steps,
  avoid jargon. Ask recall/definition questions. Meet them where they are.
- Intermediate (avg ~3): Balance depth with accessibility. Bridge from what they know to what
  they are missing. Use concrete examples.
- Proficient / Expert (avg >= 4): Push deeper. Ask about edge cases, tradeoffs, "why does this
  matter", cross-concept connections. Challenge them with application scenarios.
- If last 3 scores are all <= 2: CHANGE ANGLE completely — switch question type entirely.
- If score >= 4 for 2 consecutive turns: pivot to the NEXT uncovered key idea.
Key ideas to cover (visit all across the session):
{key_ideas}
Score history so far: {score_history}
RESPONSE STRUCTURE — follow this 3-part pattern every reply:
  1. MIRROR  — Paraphrase what the student said in 1-2 sentences
  2. INSIGHT — Acknowledge what is correct, name ONE specific conceptual gap
  3. QUESTION — ONE Socratic question calibrated to their mastery level
For gap_explanation: identify the mental model or framework missing. Teach it at their level.
{clarify_context}
Respond ONLY in this JSON format (no extra text, no markdown):
{{
  "score": <integer 1-5>,
  "feedback": "<MIRROR + INSIGHT — 2-3 sentences>",
  "gap_explanation": "<conceptual framework missing — teach at student level>",
  "next_question": "<ONE Socratic question calibrated to mastery — empty string if last question>"
}}
Concept: {concept} | Question {q_num} of {total_q}
Progression: early = define/recall, middle = mechanism/causal, late = application/edge case.
"""

FIRST_Q_PROMPT = """You are ThinkGen. Generate the FIRST Socratic question for: "{concept}"
Ask the student to explain what {concept} is in their own words.
Keep it open-ended and non-leading.
Return ONLY the question text, nothing else."""

HINT_PROMPT = """You are ThinkGen helping a student think through a question without giving the answer.
Concept: "{concept}"
Question: "{current_q}"
Give ONE guiding nudge (1-2 sentences) that:
1. Uses a simple real-world analogy or relatable comparison
2. Does NOT reveal the answer or key technical terms directly
3. Sparks their own thinking — start with "Think of it like..." or "Imagine..."
Return ONLY the nudge sentence(s)."""

CLARIFY_PROMPT = """You are ThinkGen, a Socratic AI tutor. Student is studying "{concept}".
Current question: "{current_q}"
RULES:
1. Only answer if directly about the current question or concept.
2. Off-topic: respond ONLY with "I can only help clarify questions related to this topic."
3. Do NOT give away the answer. Guide thinking.
4. 2-4 sentences max.
5. Do NOT score or generate new questions.
Student question: "{student_q}"
"""

REPORT_PROMPT = """Based on this student session on "{concept}":
{qa_block}
Write a structured learning report with these exact sections:
## ✅ What You Understood Well
## ⚠️ Conceptual Gaps
## 🧠 The Missing Mental Framework
## 🗺️ Next Steps
Be specific. Reference their actual answers. Write in English."""

TOPIC_SPLIT_PROMPT = """You are an expert curriculum designer. A student wants to learn: "{topic}"
Break into 3-5 key concepts in sequence (foundational to advanced).
Choose question count per concept (5, 6, 7, 8, 9, or 10). Simpler = fewer, complex = more.
Return ONLY a valid JSON array (no extra text, no markdown):
[
  {{
    "name": "Concept Name",
    "desc": "One sentence description",
    "key_ideas": ["idea 1", "idea 2", "idea 3", "idea 4"],
    "difficulty": "Foundational",
    "questions": 7
  }}
]
Difficulty: Foundational | Intermediate | Applied | Advanced. Order: foundational first."""


def get_client():
    key = st.secrets.get("GROQ_API_KEY", "")
    if not key:
        st.error("❌ GROQ_API_KEY is required in secrets.toml")
        st.stop()
    return Groq(api_key=key)


def get_weights(n: int) -> list:
    if n in WEIGHT_MAPS:
        return WEIGHT_MAPS[n]
    closest = min(WEIGHT_MAPS.keys(), key=lambda x: abs(x - n))
    w = list(WEIGHT_MAPS[closest])
    if len(w) > n:
        return w[:n]
    while len(w) < n:
        w.append(4)
    return w


def is_gibberish(text: str) -> bool:
    text = text.strip()
    if len(text) < 4:
        return True
    return len(re.findall(r"[a-zA-Z\u0600-\u06FF]{3,}", text)) < 2


def get_mastery(scores: list) -> tuple:
    if not scores:
        return ("Not started", 0, "#DCE3EC", 0.0)
    avg = sum(scores) / len(scores)
    if avg < 1.8:
        return ("Beginner",     max(4, int(avg / 5 * 100)), "#EF4B5E", avg)   # coral-red
    elif avg < 2.6:
        return ("Developing",   int(avg / 5 * 100), "#F5A524", avg)           # amber
    elif avg < 3.4:
        return ("Intermediate", int(avg / 5 * 100), "#F5C842", avg)           # yellow-amber
    elif avg < 4.2:
        return ("Proficient",   int(avg / 5 * 100), "#5B6EF5", avg)           # indigo
    else:
        return ("Expert",       int(avg / 5 * 100), "#1ABFA3", avg)           # teal


def mastery_bar_html(scores: list, compact: bool = False) -> str:
    label, pct, color, _ = get_mastery(scores)
    height = "4px" if compact else "5px"
    font   = "0.6rem" if compact else "0.65rem"
    margin = "8px 0 2px 0" if compact else "10px 0 4px 0"
    return (
        f'<div style="margin:{margin};">'
        f'  <div style="position:relative;background:#DCE3EC;border-radius:99px;height:{height};width:100%;">'
        f'    <div style="position:absolute;left:0;top:0;height:100%;width:{pct}%;'
        f'         background:{color};border-radius:99px;transition:width .5s ease;"></div>'
        f'  </div>'
        f'  <div style="text-align:right;font-size:{font};font-weight:700;letter-spacing:.1em;'
        f'       text-transform:uppercase;color:{color};margin-top:3px;">{label}</div>'
        f'</div>'
    )


def get_first_question(concept: str) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": FIRST_Q_PROMPT.format(concept=concept)}],
        temperature=0.7, max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def get_hint(concept: str, current_q: str) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": HINT_PROMPT.format(
            concept=concept, current_q=current_q)}],
        temperature=0.6, max_tokens=120,
    )
    return resp.choices[0].message.content.strip()


def clarify(concept: str, current_q: str, student_q: str) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": CLARIFY_PROMPT.format(
            concept=concept, current_q=current_q, student_q=student_q)}],
        temperature=0.3, max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def split_topic_into_concepts(topic: str) -> list:
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": TOPIC_SPLIT_PROMPT.format(topic=topic)}],
        temperature=0.5, max_tokens=800,
    )
    raw = re.sub(r"```(?:json)?", "", resp.choices[0].message.content.strip()).strip()
    try:
        concepts = json.loads(raw)
        valid = []
        for c in concepts:
            q_count = int(c.get("questions", 8))
            if q_count not in WEIGHT_MAPS:
                q_count = min(WEIGHT_MAPS.keys(), key=lambda x: abs(x - q_count))
            valid.append({
                "name": c.get("name", "Concept"), "desc": c.get("desc", ""),
                "key_ideas": c.get("key_ideas", []),
                "difficulty": c.get("difficulty", "Intermediate"),
                "questions": q_count,
            })
        return valid
    except Exception:
        return [{"name": topic, "desc": f"Core concepts of {topic}",
                 "key_ideas": ["foundations", "core mechanism", "applications", "limitations"],
                 "difficulty": "Intermediate", "questions": 8}]


def evaluate(concept, q_num, total_q, question, answer, history, scores, key_ideas, clarify_qs=None):
    client = get_client()
    clarify_context = "\n"
    if clarify_qs:
        qs_fmt = "\n".join(f'  - "{q}"' for q in clarify_qs)
        clarify_context = (
            f"\nSTUDENT CONFUSION SIGNALS — student asked before answering:\n{qs_fmt}\n"
            f"Use these to tailor gap_explanation and next_question.\n"
        )
    mastery_label, _, _, mastery_avg = get_mastery(scores)
    system = SYSTEM_PROMPT.format(
        concept=concept, q_num=q_num, total_q=total_q,
        clarify_context=clarify_context,
        score_history=", ".join(str(s) for s in scores) or "none yet",
        key_ideas=", ".join(key_ideas),
        mastery_label=mastery_label, mastery_avg=mastery_avg,
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"Question: {question}\nStudent answer: {answer}"})
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2, max_tokens=600)
    raw = re.sub(r"```(?:json)?", "", resp.choices[0].message.content.strip()).strip()
    try:
        result = json.loads(raw)
        result["score"] = max(1, min(5, int(result.get("score", 2))))
        return result
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                result["score"] = max(1, min(5, int(result.get("score", 2))))
                return result
            except Exception:
                pass
    return {"score": 2, "feedback": "Evaluation error.", "gap_explanation": "", "next_question": ""}


def generate_report(concept, questions, answers, scores, gaps):
    client = get_client()
    lines = [f"Q{i+1} (score {s}/5): {q}\nAnswer: {a}\nGap noted: {g}"
             for i, (q, a, s, g) in enumerate(zip(questions, answers, scores, gaps))]
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": REPORT_PROMPT.format(
            concept=concept, qa_block="\n\n".join(lines))}],
        temperature=0.4, max_tokens=900,
    )
    return resp.choices[0].message.content.strip()


def build_markdown_download(concept, questions, answers, scores, feedbacks, gaps, report, custom_topic=None):
    lines = []
    header = f"# ThinkGen Session — {custom_topic}\n**Concept:** {concept}" if custom_topic else f"# ThinkGen Session — {concept}"
    lines.append(header + "\n")
    avg = sum(scores) / len(scores) if scores else 0
    lines.append(f"**Questions:** {len(scores)}  |  **Average:** {avg:.1f}/5\n\n---\n\n## Q&A Transcript\n")
    for i, (q, a, s, f, g) in enumerate(zip(questions, answers, scores, feedbacks, gaps)):
        lines.append(f"### Question {i+1}  [{s}/5]\n**Q:** {q}\n**A:** {a}\n**Feedback:** {f}")
        if g:
            lines.append(f"**Gap:** {g}")
        lines.append("")
    lines.append("---\n\n## Full Analysis\n")
    lines.append(report)
    return "\n".join(lines)


def init():
    defaults = {
        "stage": "select", "mode": "preset",
        "concept": None, "total_q": 10, "weights": WEIGHT_MAPS[10], "key_ideas": [],
        "q_num": 1, "questions": [], "answers": [], "scores": [],
        "feedbacks": [], "gaps": [], "history": [], "current_q": "",
        "clarify_open": False, "clarify_reply": "", "clarify_questions": [],
        "hint_text": "", "loading_concept": None,
        "custom_topic": "", "custom_concepts": [], "custom_concept_idx": 0,
        "custom_results": [], "report_md": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_quiz_state():
    for k, v in [("q_num", 1), ("questions", []), ("answers", []), ("scores", []),
                 ("feedbacks", []), ("gaps", []), ("history", []), ("current_q", ""),
                 ("clarify_open", False), ("clarify_reply", ""), ("clarify_questions", []),
                 ("hint_text", ""), ("report_md", None)]:
        st.session_state[k] = v


def load_concept(info, name):
    n = info.get("questions", 10)
    st.session_state.concept   = name
    st.session_state.total_q   = n
    st.session_state.weights   = get_weights(n)
    st.session_state.key_ideas = info.get("key_ideas", [])


# ─── PAGE CONFIG & CSS ────────────────────────────────────────────────────────
st.set_page_config(page_title="ThinkGen", page_icon="🧠", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

/* ── DESIGN TOKENS ── */
:root {
    --bg:         #F0F4F8;
    --surface:    #FFFFFF;
    --border:     #DCE3EC;
    --accent:     #5B6EF5;   /* indigo-blue  */
    --accent2:    #1ABFA3;   /* teal-green   */
    --violet:     #A07AF5;   /* violet       */
    --amber:      #F5A524;   /* amber        */
    --danger:     #EF4B5E;   /* coral-red    */
    --text:       #1E2A3A;
    --muted:      #6B7A90;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: 'Inter', sans-serif; }
h1,h2,h3 { font-family: 'Space Mono', monospace; color: var(--accent2) !important; }

/* ── QUESTION CARD — blue ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background:   #EEF2FF !important;
    border:       1px solid #C7D2FE !important;
    border-radius: 14px !important;
    box-shadow:   0 1px 6px rgba(91,110,245,0.07) !important;
}
/* ── ANSWER CARD — deep mauve/plum ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background:   #6B2D5E !important;
    border:       1px solid #56244C !important;
    border-radius: 14px !important;
    box-shadow:   0 2px 10px rgba(107,45,94,0.25) !important;
    color:        #F5E6F2 !important;
}

/* ── CONCEPT BUTTONS ── */
.stButton>button {
    background:    var(--surface) !important;
    border:        1px solid var(--border) !important;
    color:         var(--text) !important;
    border-radius: 12px !important;
    padding:       1.2rem !important;
    font-size:     0.95rem !important;
    text-align:    left !important;
    height:        auto !important;
    white-space:   pre-wrap !important;
    box-shadow:    0 1px 4px rgba(0,0,0,0.05) !important;
    transition:    all 0.18s ease !important;
}
.stButton>button:hover {
    border-color: var(--accent) !important;
    background:   #ECEFFE !important;
    box-shadow:   0 4px 16px rgba(91,110,245,0.14) !important;
    transform:    translateY(-1px) !important;
}

/* ── PROGRESS BAR ── */
.stProgress>div>div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 99px !important;
}
.stProgress>div { background: var(--border) !important; border-radius: 99px !important; }

.stChatMessage { border-radius: 14px !important; }

/* ── SCORE BADGES ── */
.score-badge { display:inline-block; padding:2px 12px; border-radius:99px;
    font-family:'Space Mono',monospace; font-size:0.85rem; font-weight:700; }
.score-5 { background:#D4F7F2; color:#0D7A69; border:1px solid #1ABFA3; }   /* teal */
.score-4 { background:#E8EAFE; color:#3040C0; border:1px solid #5B6EF5; }   /* indigo */
.score-3 { background:#FEF4DC; color:#A06800; border:1px solid #F5A524; }   /* amber */
.score-2, .score-1 { background:#FDE8EB; color:#A01828; border:1px solid #EF4B5E; }  /* coral */

/* ── GAP BOX ── */
.gap-box {
    background:    #EDF6FF;
    border-left:   3px solid var(--accent2);
    border-radius: 0 8px 8px 0;
    padding:       0.8rem 1rem;
    margin-top:    0.5rem;
    font-size:     0.93rem;
    color:         #1A3458;
}

/* ── HINT BOX ── */
.hint-box {
    background:    #FFFBEC;
    border-left:   3px solid var(--amber);
    border-radius: 0 8px 8px 0;
    padding:       0.75rem 1rem;
    margin-top:    0.5rem;
    font-size:     0.9rem;
    color:         #5A3800;
}

/* ── METRIC CARDS ── */
.metric-card {
    background:    var(--surface);
    border:        1px solid var(--border);
    border-radius: 12px;
    padding:       1rem;
    text-align:    center;
    box-shadow:    0 1px 4px rgba(0,0,0,0.05);
}
.metric-val   { font-family:'Space Mono',monospace; font-size:2rem; color: var(--accent2); }
.metric-label { color: var(--muted); font-size:0.85rem; margin-top:4px; }

/* ── SIDEBAR ── */
.css-1cypcdb, [data-testid="stSidebar"] {
    background:   var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
.streamlit-expanderHeader { background: var(--surface) !important; color: var(--text) !important; }
input, textarea {
    background: var(--surface) !important;
    color:      var(--text) !important;
    border:     1px solid var(--border) !important;
}

/* ── END SESSION BUTTON ── */
.end-btn>button {
    background:   #FDE8EB !important;
    border-color: var(--danger) !important;
    color:        #A01828 !important;
}
.end-btn>button:hover {
    background:  #FAC8CE !important;
    transform:   translateY(-1px) !important;
}

/* ── DOWNLOAD / ACTION BUTTONS ── */
[data-testid="stDownloadButton"] button,
[data-testid="stBaseButton-secondary"] {
    min-height:      46px !important;
    height:          46px !important;
    width:           100% !important;
    border-radius:   10px !important;
    padding:         0 1rem !important;
    font-weight:     600 !important;
    font-size:       0.92rem !important;
    transition:      all 0.2s ease !important;
    white-space:     nowrap !important;
    display:         flex !important;
    align-items:     center !important;
    justify-content: center !important;
}
[data-testid="stDownloadButton"]:nth-of-type(1) button {
    background:  var(--accent2) !important;
    color:       #fff !important;
    border:      1.5px solid var(--accent2) !important;
    box-shadow:  0 2px 8px rgba(26,191,163,0.2) !important;
}
[data-testid="stDownloadButton"]:nth-of-type(1) button:hover {
    background:  #139E88 !important;
    transform:   translateY(-2px) !important;
    box-shadow:  0 6px 18px rgba(26,191,163,0.3) !important;
}
[data-testid="stDownloadButton"]:nth-of-type(2) button {
    background:  var(--accent) !important;
    color:       #fff !important;
    border:      1.5px solid var(--accent) !important;
    box-shadow:  0 2px 8px rgba(91,110,245,0.2) !important;
}
[data-testid="stDownloadButton"]:nth-of-type(2) button:hover {
    background:  #4758D4 !important;
    transform:   translateY(-2px) !important;
    box-shadow:  0 6px 18px rgba(91,110,245,0.3) !important;
}
[data-testid="stBaseButton-secondary"] {
    background:  #fff !important;
    color:       var(--text) !important;
    border:      1.5px solid var(--border) !important;
    box-shadow:  0 2px 6px rgba(0,0,0,0.06) !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--accent) !important;
    color:        var(--accent) !important;
    transform:    translateY(-2px) !important;
    box-shadow:   0 6px 18px rgba(91,110,245,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

init()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — CONCEPT SELECTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "select":
    st.title("🧠 ThinkGen")
    st.markdown("<p style='color:#6B7A90;font-size:1.1rem;'>Adaptive Socratic Tutor — Choose a concept or enter your own topic</p>", unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.get("loading_concept"):
        name = st.session_state.loading_concept
        st.markdown(f"<h3 style='color:#1E2A3A;'>Loading: {name}</h3>", unsafe_allow_html=True)
        with st.spinner("⏳ Generating first question..."):
            q = get_first_question(name)
            st.session_state.mode = "preset"
            load_concept(CONCEPTS[name], name)
            reset_quiz_state()
            st.session_state.stage = "quiz"
            st.session_state.q_num = 1
            st.session_state.current_q = q
            st.session_state.questions.append(q)
            st.session_state.loading_concept = None
        st.rerun()
    else:
        col_custom, _ = st.columns([1, 3])
        with col_custom:
            if st.button("🎯 Learn any topic...", key="custom_topic_btn", use_container_width=True):
                st.session_state.stage = "topic_input"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (name, info) in enumerate(CONCEPTS.items()):
            diff_color = DIFF_CLR[info["difficulty"]]
            with cols[i % 2]:
                st.markdown(
                    f'<div style="margin-bottom:6px;display:flex;gap:6px;align-items:center;">'
                    f'<span style="font-size:.65rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;'
                    f'padding:2px 8px;border-radius:99px;background:{diff_color};color:#fff;">{info["difficulty"]}</span>'
                    f'<span style="font-size:.75rem;color:#6B7A90;">{info["questions"]} questions</span></div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"{name}\n{info['desc']}", key=name, use_container_width=True):
                    st.session_state.loading_concept = name
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1b — CUSTOM TOPIC
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "topic_input":
    st.title("🎯 Custom Topic")
    st.markdown("<p style='color:#6B7A90;font-size:1.1rem;'>Enter any topic — ThinkGen builds a learning path for you</p>", unsafe_allow_html=True)
    st.markdown("---")
    topic_input = st.text_input("What do you want to learn?",
        placeholder="e.g. Quantum Computing, Blockchain, CRISPR...", key="topic_text_input")
    col_start, col_back, _ = st.columns([1, 1, 3])
    with col_start:
        start_clicked = st.button("🚀 Build Learning Path", type="primary", use_container_width=True)
    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.stage = "select"
            st.rerun()
    if start_clicked and topic_input.strip():
        with st.spinner(f"⏳ Building path for **{topic_input.strip()}**..."):
            concepts = split_topic_into_concepts(topic_input.strip())
            st.session_state.mode = "custom"
            st.session_state.custom_topic = topic_input.strip()
            st.session_state.custom_concepts = concepts
            st.session_state.custom_concept_idx = 0
            st.session_state.custom_results = []
            first = concepts[0]
            load_concept(first, first["name"])
            reset_quiz_state()
            q = get_first_question(first["name"])
            st.session_state.stage = "quiz"
            st.session_state.q_num = 1
            st.session_state.current_q = q
            st.session_state.questions.append(q)
        st.rerun()
    elif start_clicked:
        st.warning("Please enter a topic first.")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — QUIZ
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "quiz":
    concept   = st.session_state.concept
    q_num     = st.session_state.q_num
    total_q   = st.session_state.total_q
    scores    = st.session_state.scores
    weights   = st.session_state.weights
    key_ideas = st.session_state.key_ideas

    if st.session_state.mode == "custom":
        idx = st.session_state.custom_concept_idx
        st.markdown(
            f"<p style='color:#6B7A90;font-size:0.9rem;'>📚 <b>{st.session_state.custom_topic}</b> "
            f"— Concept {idx+1} of {len(st.session_state.custom_concepts)}</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"### {concept}")
        st.markdown(f"<p style='color:#6B7A90;'>Question <b style='color:#1E2A3A'>{q_num}</b> of {total_q}</p>", unsafe_allow_html=True)
        st.markdown(mastery_bar_html(scores), unsafe_allow_html=True)
        st.markdown("---")
        if scores:
            st.markdown('<div class="end-btn">', unsafe_allow_html=True)
            if st.button("⏹ End Session & Get Report", key="end_session", use_container_width=True):
                st.session_state.stage = "report"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("")
        if st.button("← Back to Topics", key="back_from_quiz", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.title(concept)
    st.markdown("---")

    for i in range(len(st.session_state.answers)):
        # Question bubble — blue
        st.markdown(
            f'<div style="background:#EEF2FF;border:1px solid #C7D2FE;border-radius:14px;'
            f'padding:1rem 1.2rem;margin-bottom:8px;">' 
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">' 
            f'<span style="font-size:1.2rem;">🧠</span>' 
            f'<b style="color:#1E2A3A;font-size:0.9rem;">Question {i+1}</b></div>' 
            f'<div style="color:#1E2A3A;font-size:0.95rem;line-height:1.6;">{st.session_state.questions[i]}</div>' 
            f'</div>',
            unsafe_allow_html=True
        )
        # Answer bubble — green
        st.markdown(
            f'<div style="background:#EDFAF5;border:1px solid #B2EDD8;border-radius:14px;'
            f'padding:1rem 1.2rem;margin-bottom:16px;">' 
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">' 
            f'<span style="font-size:1.2rem;">🎓</span>' 
            f'<b style="color:#0D7A69;font-size:0.9rem;">Your Answer</b></div>' 
            f'<div style="color:#1E2A3A;font-size:0.95rem;line-height:1.6;">{st.session_state.answers[i]}</div>' 
            f'{mastery_bar_html(st.session_state.scores[:i+1], compact=True)}' 
            f'</div>',
            unsafe_allow_html=True
        )

    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(f"**Question {q_num}**", unsafe_allow_html=True)
        st.markdown(st.session_state.current_q)

        col_btn, _ = st.columns([1, 4])
        with col_btn:
            if st.button(
                "🙋 Need a hint?" if not st.session_state.clarify_open else "✖ Close",
                key="clarify_toggle", use_container_width=True,
            ):
                st.session_state.clarify_open  = not st.session_state.clarify_open
                st.session_state.clarify_reply = ""
                st.session_state.hint_text     = ""
                st.rerun()

        if st.session_state.clarify_open:
            if not st.session_state.hint_text:
                with st.spinner("💡 Getting a nudge..."):
                    st.session_state.hint_text = get_hint(concept, st.session_state.current_q)
                st.rerun()

            if st.session_state.hint_text:
                st.markdown(
                    f"<div class='hint-box'>💡 <b>Think about it this way:</b><br>"
                    f"{st.session_state.hint_text}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<p style='color:#6B7A90;font-size:0.85rem;margin-top:0.8rem;'>"
                "Still stuck? Ask a specific question:</p>", unsafe_allow_html=True,
            )
            cq = st.text_input("Your question:", key="clarify_input",
                placeholder="e.g. What exactly does 'attention' mean here?",
                label_visibility="collapsed")
            if st.button("Ask →", key="clarify_send"):
                if cq and cq.strip():
                    with st.spinner("💭 Thinking..."):
                        reply = clarify(concept, st.session_state.current_q, cq.strip())
                    st.session_state.clarify_questions.append(cq.strip())
                    st.session_state.clarify_reply = reply
                    st.rerun()

            if st.session_state.clarify_reply:
                st.markdown(
                    f"<div class='gap-box'>🗣 <b>Clarification:</b><br>{st.session_state.clarify_reply}</div>",
                    unsafe_allow_html=True,
                )

    user_answer = st.chat_input("Your answer here...")

    # ── JS: apply card colors on load ────────────────────────────────────────
    components.html("""
<script>
(function(){
  var doc = window.parent.document;

  function colorCard(msg){
    if (msg.querySelector('[data-testid="chatAvatarIcon-user"]')) {
      msg.style.background   = '#6B2D5E';
      msg.style.border       = '1px solid #56244C';
      msg.style.borderRadius = '14px';
      msg.style.boxShadow    = '0 2px 10px rgba(107,45,94,0.25)';
      msg.style.color        = '#F5E6F2';
      msg.querySelectorAll('p,span,div').forEach(function(el){
        if(!el.children.length) el.style.color = '#F5E6F2';
      });
    } else {
      msg.style.background   = '#EEF2FF';
      msg.style.border       = '1px solid #C7D2FE';
      msg.style.borderRadius = '14px';
      msg.style.boxShadow    = '0 1px 6px rgba(91,110,245,0.07)';
    }
  }

  function applyAll(){
    doc.querySelectorAll('[data-testid="stChatMessage"]').forEach(colorCard);
  }

  [100, 400, 900, 1800, 3000].forEach(function(ms){ setTimeout(applyAll, ms); });

  var observer = new MutationObserver(function(){ applyAll(); });
  setTimeout(function(){
    var root = doc.querySelector('.stApp') || doc.body;
    observer.observe(root, {childList:true, subtree:true});
  }, 500);
})();
</script>
""", height=0)

    if user_answer:
        if is_gibberish(user_answer):
            st.warning("⚠️ Please write a real answer — at least 2 meaningful words. Express what you know, even if unsure.")
            st.stop()

        with st.spinner("⏳ Evaluating..."):
            result = evaluate(concept, q_num, total_q,
                              st.session_state.current_q, user_answer,
                              st.session_state.history, scores, key_ideas,
                              st.session_state.clarify_questions)

            score  = result["score"]
            gap    = result.get("gap_explanation", "")
            next_q = result.get("next_question", "").strip()

            st.session_state.answers.append(user_answer)
            st.session_state.scores.append(score)
            st.session_state.feedbacks.append(result.get("feedback", ""))
            st.session_state.gaps.append(gap)

            st.session_state.history.append({"role": "user", "content": f"Q: {st.session_state.current_q}\nA: {user_answer}"})
            st.session_state.history.append({"role": "assistant", "content": f"Score: {score}. Gap: {gap}"})
            if len(st.session_state.history) > 12:
                st.session_state.history = st.session_state.history[-12:]

            if q_num >= total_q:
                if st.session_state.mode == "custom":
                    st.session_state.custom_results.append({
                        "concept": concept,
                        "questions": st.session_state.questions[:],
                        "answers":   st.session_state.answers[:],
                        "scores":    st.session_state.scores[:],
                        "feedbacks": st.session_state.feedbacks[:],
                        "gaps":      st.session_state.gaps[:],
                    })
                    st.session_state.stage = "concept_done"
                else:
                    st.session_state.stage = "report"
            else:
                st.session_state.q_num += 1
                if not next_q:
                    next_q = get_first_question(concept)
                st.session_state.current_q = next_q
                st.session_state.questions.append(next_q)
                st.session_state.clarify_open  = False
                st.session_state.clarify_reply = ""
                st.session_state.hint_text     = ""
                st.session_state.clarify_questions = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2b — CONCEPT DONE (Custom)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "concept_done":
    idx      = st.session_state.custom_concept_idx
    concepts = st.session_state.custom_concepts
    rd       = st.session_state.custom_results[-1]
    concept  = rd["concept"]
    scores   = rd["scores"]
    weights  = get_weights(len(scores))
    pct      = sum(s*w for s,w in zip(scores,weights)) / sum(w*5 for w in weights) * 100 if scores else 0
    avg      = sum(scores)/len(scores) if scores else 0

    st.title(f"✅ Concept Complete: {concept}")
    st.markdown(f"<p style='color:#6B7A90;'>Topic: <b>{st.session_state.custom_topic}</b> — Concept {idx+1} of {len(concepts)}</p>", unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    for col, val, label in [(c1,f"{pct:.0f}%","Score"),(c2,f"{avg:.1f}/5","Avg Answer"),(c3,f"{sum(1 for s in scores if s>=4)}/{len(scores)}","Strong Answers")]:
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-val'>{val}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(pct/100)

    next_idx = idx + 1
    if next_idx < len(concepts):
        nc = concepts[next_idx]
        st.markdown("---")
        st.markdown(f"**Next up:** {nc['name']} — {nc['desc']}")
        col_next, col_stop, _ = st.columns([1, 1, 3])
        with col_next:
            if st.button(f"▶ Start: {nc['name']}", type="primary", use_container_width=True):
                st.session_state.custom_concept_idx += 1
                load_concept(nc, nc["name"])
                reset_quiz_state()
                with st.spinner("⏳ Generating first question..."):
                    q = get_first_question(nc["name"])
                    st.session_state.stage = "quiz"
                    st.session_state.q_num = 1
                    st.session_state.current_q = q
                    st.session_state.questions.append(q)
                st.rerun()
        with col_stop:
            if st.button("📊 Get Full Report", use_container_width=True):
                st.session_state.stage = "report"
                st.rerun()
    else:
        st.success("🎉 You've completed all concepts!")
        if st.button("📊 View Full Report", type="primary"):
            st.session_state.stage = "report"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "report":
    if st.session_state.mode == "custom" and st.session_state.custom_results:
        rd = st.session_state.custom_results[-1]
        concept, questions, answers = rd["concept"], rd["questions"], rd["answers"]
        scores, feedbacks, gaps = rd["scores"], rd["feedbacks"], rd["gaps"]
        custom_topic = st.session_state.custom_topic
    else:
        concept, questions, answers = st.session_state.concept, st.session_state.questions, st.session_state.answers
        scores, feedbacks, gaps = st.session_state.scores, st.session_state.feedbacks, st.session_state.gaps
        custom_topic = None

    weights      = get_weights(len(scores)) if scores else WEIGHT_MAPS[10]
    total_earned = sum(s*w for s,w in zip(scores,weights))
    max_possible = sum(w*5 for w in weights)
    overall_pct  = total_earned/max_possible*100 if max_possible else 0
    avg_score    = sum(scores)/len(scores) if scores else 0
    strong_count = sum(1 for s in scores if s >= 4)
    n            = len(scores)

    with st.sidebar:
        st.markdown(f"### {concept}")
        st.markdown("<p style='color:#6B7A90;font-size:0.9rem;'>Session complete ✅</p>", unsafe_allow_html=True)
        st.markdown(mastery_bar_html(scores), unsafe_allow_html=True)
        st.markdown("---")
        if st.button("← Back to Topics", key="back_from_report", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.title("📊 Final Report")
    if custom_topic:
        st.markdown(f"<p style='color:#6B7A90;'>Topic: <b>{custom_topic}</b> — {concept}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:#6B7A90;'>{concept}</p>", unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [(c1,f"{overall_pct:.0f}%","Overall Score"),(c2,f"{avg_score:.1f}/5","Average Answer"),
                             (c3,f"{strong_count}/{n}","Strong (4+)"),(c4,f"{n-strong_count}/{n}","To Review")]:
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-val'>{val}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗂️ Question Breakdown")
    for i in range(len(answers)):
        s    = scores[i]
        icon = "🟢" if s >= 4 else "🟡" if s == 3 else "🔴"
        with st.expander(f"{icon} Question {i+1}  —  {s}/5"):
            st.markdown(f"**Question:** {questions[i]}")
            st.markdown(f"**Your answer:** {answers[i]}")
            st.markdown(f"**Feedback:** {feedbacks[i]}")
            if gaps[i]:
                st.markdown(f"<div class='gap-box'>📖 {gaps[i]}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Full Analysis")
    if st.session_state.report_md is None:
        with st.spinner("⏳ Generating report..."):
            st.session_state.report_md = generate_report(concept, questions, answers, scores, gaps)
    report_md = st.session_state.report_md
    st.markdown(report_md)

    st.markdown("---")
    md_content = build_markdown_download(concept, questions, answers, scores, feedbacks, gaps, report_md, custom_topic)
    safe_name  = re.sub(r"[^\w]", "_", concept)[:30]


    import base64
    md_b64  = base64.b64encode(md_content.encode()).decode()
    txt_b64 = base64.b64encode(md_content.encode()).decode()

    st.markdown(f'''
    <div style="display:flex;gap:12px;margin-top:8px;">
      <a href="data:text/markdown;base64,{md_b64}" download="thinkgen_{safe_name}.md"
         style="flex:1;height:52px;display:flex;align-items:center;justify-content:center;
                background:#1ABFA3;color:#fff;border-radius:10px;font-weight:600;
                font-size:0.92rem;text-decoration:none;border:1.5px solid #1ABFA3;
                box-shadow:0 2px 8px rgba(26,191,163,0.2);">
        ⬇️&nbsp; Save as .md
      </a>
      <a href="data:text/plain;base64,{txt_b64}" download="thinkgen_{safe_name}.txt"
         style="flex:1;height:52px;display:flex;align-items:center;justify-content:center;
                background:#5B6EF5;color:#fff;border-radius:10px;font-weight:600;
                font-size:0.92rem;text-decoration:none;border:1.5px solid #5B6EF5;
                box-shadow:0 2px 8px rgba(91,110,245,0.2);">
        📄&nbsp; Save as .txt
      </a>
      <a href="?"
         style="flex:1;height:52px;display:flex;align-items:center;justify-content:center;
                background:#fff;color:#1E2A3A;border-radius:10px;font-weight:600;
                font-size:0.92rem;text-decoration:none;border:1.5px solid #DCE3EC;
                box-shadow:0 2px 6px rgba(0,0,0,0.06);">
        🔄&nbsp; New Session
      </a>
    </div>
    ''', unsafe_allow_html=True)