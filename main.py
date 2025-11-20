from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json

# Optional OpenAI client (lazy import to avoid crashing if not installed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        _client = None

app = FastAPI(title="MindMap AI API")

# CORS for local dev and preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Models -----
class GenerateRequest(BaseModel):
    subject: str
    style: Optional[str] = None
    depth: int = 2
    template: Optional[str] = None

class Node(BaseModel):
    id: str
    label: str
    children: List['Node'] = []

Node.model_rebuild()

class GenerateResponse(BaseModel):
    root: Node
    theme: str
    template: Optional[str] = None

# ----- Data -----
THEMES = [
    {
        "id": "dark",
        "name": "Dark Gray",
        "background": "#0b0f19",
        "primary": "#8b5cf6",
        "muted": "#1f2937",
        "text": "#e5e7eb",
    },
    {
        "id": "light",
        "name": "Minimal Light",
        "background": "#ffffff",
        "primary": "#111827",
        "muted": "#e5e7eb",
        "text": "#111827",
    },
    {
        "id": "neon",
        "name": "Neon Pulse",
        "background": "#050b1b",
        "primary": "#22d3ee",
        "muted": "#0b132b",
        "text": "#e2f7ff",
    },
]

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "classic": {
        "id": "classic",
        "name": "Classic Mind Map",
        "branches": [
            "Definition",
            "Key Elements",
            "Use Cases",
            "Challenges",
            "Trends",
        ],
    },
    "logo_brief": {
        "id": "logo_brief",
        "name": "Logo Design Brief",
        "branches": [
            "Brand Values",
            "Target Audience",
            "Visual Style",
            "Color + Typography",
            "Usage Contexts",
            "Competitors",
        ],
    },
    "brainstorm": {
        "id": "brainstorm",
        "name": "Divergent Brainstorm",
        "branches": [
            "Problem Statement",
            "Wild Ideas",
            "Constraints",
            "Inspiration",
            "Next Actions",
        ],
    },
}

# ----- Heuristic Generator (fallback) -----

def expand_topic(topic: str, depth: int, idx_prefix: str = "1", branches: Optional[List[str]] = None) -> Node:
    node = Node(id=idx_prefix, label=topic, children=[])
    if depth <= 0:
        return node
    if branches is None:
        branches = [
            f"Definition of {topic}",
            f"Key elements of {topic}",
            f"Use cases of {topic}",
            f"Challenges in {topic}",
            f"Trends for {topic}",
        ]
    for i, sub in enumerate(branches, start=1):
        child = expand_topic(sub, depth - 1, f"{idx_prefix}.{i}")
        node.children.append(child)
    return node

# ----- AI Helpers -----

def _ai_suggest_branches(topic: str, count: int = 5, context: Optional[str] = None) -> List[str]:
    """Ask the LLM for N concise branch labels for a given topic.
    Returns a list of strings. On error or if client unavailable, returns empty list.
    """
    if _client is None:
        return []
    system = (
        "You are a branding strategist and mind-mapping assistant. "
        "Return only a JSON array of short, punchy branch labels (3-5 words each)."
    )
    user = (
        f"Topic: {topic}.\n"
        f"Goal: Generate {count} concise sub-branches for a logo design mind map.\n"
        + (f"Context: {context}\n" if context else "")
        + "Output strictly a JSON array of strings, no extra commentary."
    )
    try:
        resp = _client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        text = resp.choices[0].message.content.strip()
        # Ensure it's a JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if isinstance(x, (str, int, float))][:count]
    except Exception:
        return []
    return []


def expand_topic_ai(subject: str, depth: int, idx_prefix: str = "1", template_branches: Optional[List[str]] = None) -> Node:
    """Recursively build a Node tree using AI branch suggestions.
    First level can follow template branches if provided; deeper levels use AI.
    Safely falls back to heuristic if AI yields nothing.
    """
    root = Node(id=idx_prefix, label=subject, children=[])
    if depth <= 0:
        return root

    # Level 1 branches: prefer template branches if given, else ask AI
    if template_branches:
        level1 = template_branches
    else:
        level1 = _ai_suggest_branches(subject, count=5, context="Top-level categories for a brand/logo mind map")
        if not level1:
            level1 = [
                f"Definition of {subject}",
                f"Key elements of {subject}",
                f"Use cases of {subject}",
                f"Challenges in {subject}",
                f"Trends for {subject}",
            ]

    for i, b in enumerate(level1, start=1):
        node_label = b if isinstance(b, str) else str(b)
        child = Node(id=f"{idx_prefix}.{i}", label=node_label, children=[])
        root.children.append(child)
        # For deeper levels, use AI suggestions contextualized on the branch
        if depth - 1 > 0:
            sub = _ai_suggest_branches(
                topic=node_label,
                count=5,
                context=f"Subtopics expanding '{node_label}' for subject '{subject}' in a logo/brand design context.",
            )
            if not sub:
                sub = [
                    f"What is {node_label}",
                    f"How to approach {node_label}",
                    f"Examples of {node_label}",
                    f"Risks of {node_label}",
                    f"Next steps for {node_label}",
                ]
            # Recurse one more level (depth-1) using heuristic for the next layer
            for j, sublabel in enumerate(sub, start=1):
                grand = expand_topic(sublabel, depth - 2, f"{idx_prefix}.{i}.{j}") if depth - 2 > 0 else Node(id=f"{idx_prefix}.{i}.{j}", label=sublabel, children=[])
                child.children.append(grand)

    return root


# ----- Routes -----

@app.get("/health")
async def health():
    return {"status": "ok", "ai": bool(_client is not None)}

@app.get("/themes")
async def get_themes():
    return THEMES

@app.get("/templates")
async def get_templates():
    return list(TEMPLATES.values())

@app.post("/generate", response_model=GenerateResponse)
async def generate_map(req: GenerateRequest):
    subject = (req.subject or "").strip()
    if not subject:
        raise HTTPException(status_code=400, detail="Subject is required")

    depth = max(1, min(req.depth or 2, 3))

    template_cfg = TEMPLATES.get(req.template or "", None)
    branches = template_cfg["branches"] if template_cfg else None

    # Use AI if available; fallback otherwise
    if _client is not None:
        root = expand_topic_ai(subject, depth, branches)
    else:
        root = expand_topic(subject, depth, branches=branches)

    theme_id = req.style or "dark"

    return GenerateResponse(root=root, theme=theme_id, template=template_cfg["id"] if template_cfg else None)
