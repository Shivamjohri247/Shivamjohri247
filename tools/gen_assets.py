#!/usr/bin/env python3
"""Generate the terminal-style SVG panes for the GitHub profile README."""
import os, sys

W = 880
FS = 13
LH = 20
ART_LH = 16
X = 26
X2 = 420
FONT = "JetBrains Mono, SFMono-Regular, Menlo, Consolas, Liberation Mono, monospace"

DARK = dict(bg="#0d1117", bar="#161b22", txt="#c9d1d9", dim="#8b949e", rule="#21262d",
            grn="#7ee787", blu="#58a6ff", lblu="#a5d6ff", org="#d97757", yel="#e3b341",
            wht="#e6edf3", brd="#30363d")
LIGHT = dict(bg="#ffffff", bar="#f6f8fa", txt="#1f2328", dim="#656d76", rule="#d8dee4",
             grn="#1a7f37", blu="#0969da", lblu="#0550ae", org="#bc4c00", yel="#7d4e00",
             wht="#1f2328", brd="#d0d7de")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Pane:
    def __init__(self, pal, title="shivam@github — zsh"):
        self.p = pal
        self.title = title
        self.out = []
        self.y = 44

    def prompt(self, cmd):
        p = self.p
        self.y += 28
        self.out.append(
            f'<text x="{X}" y="{self.y}" font-size="{FS}" xml:space="preserve">'
            f'<tspan fill="{p["grn"]}">shivam@github</tspan><tspan fill="{p["dim"]}">:</tspan>'
            f'<tspan fill="{p["blu"]}">~</tspan><tspan fill="{p["dim"]}">$ </tspan>'
            f'<tspan fill="{p["wht"]}">{esc(cmd)}</tspan></text>')
        self.y += 6

    def lines(self, text, color="txt", x=X, lh=LH):
        for ln in text.split("\n"):
            self.y += lh
            self.out.append(
                f'<text x="{x}" y="{self.y}" font-size="{FS}" fill="{self.p[color]}" '
                f'xml:space="preserve">{esc(ln)}</text>')

    def art(self, text, color="txt", x=X):
        self.lines(text, color, x, ART_LH)

    def rows(self, rows, x=X, lh=LH):
        for ln, c in rows:
            self.y += lh
            self.out.append(
                f'<text x="{x}" y="{self.y}" font-size="{FS}" fill="{self.p[c]}" '
                f'xml:space="preserve">{esc(ln)}</text>')

    def gap(self, n=10):
        self.y += n

    def render(self, path):
        p = self.p
        h = self.y + 26
        head = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" '
            f'viewBox="0 0 {W} {h}" font-family="{FONT}" role="img">',
            f'<rect width="{W}" height="{h}" rx="10" fill="{p["bg"]}"/>',
            f'<path d="M0 10a10 10 0 0 1 10-10h{W-20}a10 10 0 0 1 10 10v24H0z" fill="{p["bar"]}"/>',
            '<circle cx="22" cy="17" r="6" fill="#ff5f56"/>',
            '<circle cx="42" cy="17" r="6" fill="#ffbd2e"/>',
            '<circle cx="62" cy="17" r="6" fill="#27c93f"/>',
            f'<text x="{W//2}" y="22" font-size="12" fill="{p["dim"]}" '
            f'text-anchor="middle">{esc(self.title)}</text>',
        ]
        with open(path, "w") as f:
            f.write("\n".join(head + self.out + ["</svg>"]) + "\n")
        return h


# ------------------------------------------------------------------ panes
def banner(pal):
    t = Pane(pal, "shivam@github — welcome")
    t.prompt("./welcome.sh")
    t.art(r""" __________________________________________
< you've reached shivam's agent orchestra.  >
< mind the loop.                            >
 ------------------------------------------""")
    t.art(r"""        \   ___
         \ [o_o]
           /|_|\
            d   b""", "org")
    return t


def neofetch(pal):
    t = Pane(pal, "shivam@github — neofetch")
    t.prompt("neofetch")
    top = t.y
    t.rows([
        ("          ┌─────────────┐",           "blu"),
        ("     ┌────┤   PLANNER   ├────┐",      "blu"),
        ("     │    └──────┬──────┘    │",      "blu"),
        ("     ▼           ▼           ▼",      "dim"),
        ("┌─────────┐ ┌────────┐ ┌──────────┐", "lblu"),
        ("│ RETRIEVE│ │  TOOL  │ │  VERIFY  │", "lblu"),
        ("└────┬────┘ └───┬────┘ └────┬─────┘", "lblu"),
        ("     └──────────┼───────────┘",       "dim"),
        ("                ▼",                   "dim"),
        ("         ┌─────────────┐",            "org"),
        ("         │   CRITIC    │",            "org"),
        ("         └──────┬──────┘",            "org"),
        ("                ▼",                   "dim"),
        ("            [ OUTPUT ]",              "grn"),
    ])
    left_end = t.y
    t.y = top
    t.rows([
        ("shivam@github",                                     "grn"),
        ("─────────────────────────────────────────",         "rule"),
        ("🏗  role    AI Architect, Applied R&D @ Suzega",     "txt"),
        ("🤖 focus   Multi-agent systems · agentic RAG",       "txt"),
        ("📄 papers  2 preprints on Zenodo (Aug 2026)",        "txt"),
        ("📦 ships   diff-guard — live on PyPI",               "txt"),
        ("🔬 into    Runtime verification & agent safety",     "txt"),
        ("🧰 stack   Python · Go · LangGraph · PyTorch · AWS", "txt"),
        ("🎓 certs   GCP ACE · AWS AI Practitioner",           "txt"),
        ("📍 where   Gurugram, India · remote",                "txt"),
        ("⚡ since   10+ yrs ML → now shipping agents",        "txt"),
    ], x=X2)
    t.gap(14)
    for i, c in enumerate([pal["bg"], pal["bar"], pal["blu"], pal["grn"], pal["org"], pal["txt"]]):
        t.out.append(f'<rect x="{X2 + i*36}" y="{t.y}" width="34" height="18" '
                     f'fill="{c}" stroke="{pal["brd"]}"/>')
    t.y += 18
    t.y = max(t.y, left_end)
    return t


def gitlog(pal):
    t = Pane(pal, "shivam@github — git log")
    t.prompt('git log --author="Shivam Johri" --oneline --decorate')
    t.rows([("a7f3c21 (HEAD -> main)  2025-07   AI Architect, Applied R&D · Suzega", "yel")])
    t.lines("""          multi-agent BOM automation — quote gen 3 days → <4 hrs
          underwriting assistant topology across 6 downstream systems
          enterprise AI guardrails for regulated BFSI + manufacturing""")
    t.gap(8)
    t.rows([("3d9b104                 2024-02   Senior ML Engineer · EPAM Systems", "yel")])
    t.lines("""          NER fine-tuning pipeline — accuracy +77% @ 100K+ queries/day
          query expansion — coverage +25%, latency −35%, tokens −40%
          semantic search + XGB rerank — 92% precision on 2M+ docs""")
    t.gap(8)
    t.rows([("8c15ae7                 2021-09   Senior Analyst, ML Engineering · Accenture", "yel")])
    t.lines("""          document intelligence for top-tier life sciences client
          regulatory + pharmacovigilance pipelines — manual effort −40%
          recommendation engine POC — 10K+ users, engagement +15%""")
    t.gap(8)
    t.rows([("1b0d8f2                 2016-03   ML Engineer / Test Analyst · TCS", "yel")])
    t.lines("""          OCR + CTPN ETL — 200K+ docs, 18–20 financial doc types
          manual validation cost −30% · CV models for BFSI R&D""")
    return t


def stack(pal):
    t = Pane(pal, "shivam@github — stack")
    t.prompt("tree -L 2 ~/.stack")
    t.rows([("~/.stack", "blu")])
    t.lines("├── agents/", "blu")
    t.lines("""│   ├── langgraph  crewai  openai-agents-sdk  semantic-kernel
│   ├── mcp  react  tool-calling  human-in-the-loop
│   └── agentic-rag  langsmith  guardrails""")
    t.lines("├── ml/", "blu")
    t.lines("""│   ├── pytorch  tensorflow  huggingface
│   └── peft  lora  qlora  prompt-engineering  ner""")
    t.lines("├── platform/", "blu")
    t.lines("""│   ├── aws{sagemaker,bedrock,lambda}  gcp-vertex  azure-ai
│   └── docker  kubernetes  mlflow  ci-cd  observability""")
    t.lines("└── data/", "blu")
    t.lines("""    ├── python  go  fastapi  spark  sql
    └── pinecone  chromadb  pgvector  elasticsearch""")
    return t


def footer(pal):
    t = Pane(pal, "shivam@github — crontab")
    t.prompt('sudo echo "* * * * * /usr/local/bin/diff-guard --watch" >> /tmp/crontab$$')
    t.lines("[sudo] Password for shivam:", "dim")
    t.gap(10)
    t.lines("BEARing down on bad diffs before they ship...")
    t.gap(8)
    t.art("""  .-\"\"\"\"-.
 /  ◕  ◕  \\        [agent]
|    ᴗ     |
|  \\___/   |
 \\  `-`   /
  `-.__.-'""", "org")
    return t


PANES = dict(banner=banner, neofetch=neofetch, gitlog=gitlog, stack=stack, footer=footer)

if __name__ == "__main__":
    dest = sys.argv[1]
    os.makedirs(dest, exist_ok=True)
    for name, fn in PANES.items():
        for suffix, pal in (("", DARK), ("-light", LIGHT)):
            path = os.path.join(dest, f"{name}{suffix}.svg")
            h = fn(pal).render(path)
            print(f"  {os.path.basename(path):24} {W}x{h}")
