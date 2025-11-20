from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

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

# ----- Generator -----

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

# ----- Routes -----

@app.get("/health")
async def health():
    return {"status": "ok"}

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

    root = expand_topic(subject, depth, branches=branches)
    theme_id = req.style or "dark"

    return GenerateResponse(root=root, theme=theme_id, template=template_cfg["id"] if template_cfg else None)
