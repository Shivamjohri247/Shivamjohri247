<div align="center">

# Shivam Johri

**AI Architect • Applied AI Lead • Agentic Systems Architect**

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=20&duration=3500&pause=1200&color=58A6FF&center=true&vCenter=true&width=800&lines=Orchestrating+Autonomous+Multi-Agent+Systems;Shipping+Production-Grade+Agentic+AI+at+Enterprise+Scale;10%2B+Years+Architecting+ML+Systems%E2%86%92Agentic+AI;Manufacturing+%7C+Financial+Services+%7C+Responsible+AI)](https://github.com/Shivamjohri247)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/shivam-johri)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:shivamjohri247@gmail.com)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=flat-square&logo=twitter&logoColor=white)](https://twitter.com/sicdrummer247)
[![Blog](https://img.shields.io/badge/Blog-0D1117?style=flat-square&logo=github&logoColor=white)](https://shivamjohri247.github.io)
<img src="https://komarev.com/ghpvc/?username=Shivamjohri247&style=flat-square&color=58A6FF&label=Profile+Views" alt="Profile Views" />

</div>

---

### 🎯 Focus Areas

<div align="center">

**Orchestrating Autonomous Multi-Agent Systems for Manufacturing & Financial Services**  
**Advancing Agent Reliability, Autonomy & Cost-Efficiency**  
**Governance-First AI: Guardrails, Observability & Compliance**

</div>

<table>
<tr>
<td width="50%" valign="top">

**Current Focus**
- Designing production-grade agentic AI & multi-agent systems for enterprise clients
- Building autonomous agent workflows across Manufacturing & Financial Services
- Advancing agent reliability, autonomy & cost-efficiency through applied R&D
- Implementing governance-first AI: guardrails, observability & compliance

**Role & Experience**
- **AI Architect, Applied R&D & Industry Solutions** @ **Suzega**
- **10+ Years** Architecting ML Systems → Agentic AI
- **EPAM** → **Accenture** → **TCS** → **Suzega** (Current)
- 📍 India (Remote)

</td>
<td width="50%" valign="top">

**Key Expertise**

**Multi-Agent Systems & Agentic AI**  
LangGraph • CrewAI • OpenAI Agents SDK • AWS Strands Agents • AutoGen

**LLMs & Fine-tuning**  
PEFT/LoRA • Prompt Engineering • Model Guardrails • Agentic RAG

**Agent Protocols & Interoperability**  
MCP • ACP • A2A

**NLP & Search**  
NER • Semantic Search • Embeddings • RAG Pipelines

**MLOps & Cloud**  
Real-time Inference • Latency Optimization • Model Observability • AWS/GCP/Azure • Kubernetes

</td>
</tr>
</table>

---

### 🛠 Tech Stack Constellation

<div align="center">

*Dot size = proficiency level (●○○ Familiar → ●●● Expert) • Connections = real project synergies*

</div>

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#0D1117', 'primaryTextColor': '#E6EDF3', 'primaryBorderColor': '#30363D', 'lineColor': '#58A6FF', 'secondaryColor': '#161B22', 'tertiaryColor': '#21262D'}}}%%
graph LR
    subgraph Orchestration["🎭 Orchestration Core"]
        LG[LangGraph<br/>●●●]
        CR[CrewAI<br/>●●●]
        OA[OpenAI Agents SDK<br/>●●○]
        AS[AWS Strands<br/>●○○]
        AG[AutoGen<br/>●●○]
    end

    subgraph Protocols["🔗 Agent Protocols"]
        MCP[MCP<br/>●●●]
        ACP[ACP<br/>●●○]
        A2A[A2A<br/>●●○]
    end

    subgraph RAG["🧠 Retrieval & Reasoning"]
        RAG[Agentic RAG<br/>●●●]
        FT[Fine-tuning<br/>●●●]
        PE[Prompt Eng<br/>●●●]
        MG[Model Guardrails<br/>●●○]
    end

    subgraph MLFrameworks["⚙️ ML Frameworks"]
        PT[PyTorch<br/>●●●]
        TF[TensorFlow<br/>●●○]
        US[Unsloth<br/>●●○]
        PEFT[PEFT/LoRA<br/>●●●]
        HF[HuggingFace<br/>●●●]
    end

    subgraph CloudMLOps["☁️ Cloud & MLOps"]
        AW[AWS Bedrock<br/>●●●]
        SM[SageMaker<br/>●●○]
        GC[GCP Vertex AI<br/>●●○]
        AZ[Azure AI<br/>●○○]
        K8[Kubernetes<br/>●●○]
        DK[Docker<br/>●●●]
        ML[MLflow<br/>●●○]
    end

    subgraph Languages["💻 Languages"]
        PY[Python<br/>●●●]
        TS[TypeScript<br/>●●○]
        GO[Go<br/>●○○]
    end

    LG --- CR
    CR --- OA
    OA --- AS
    AS --- AG
    AG --- LG

    LG --- MCP
    CR --- A2A
    OA --- ACP

    MCP --- RAG
    A2A --- FT
    ACP --- PE
    RAG --- MG

    RAG --- PT
    FT --- US
    PE --- HF
    MG --- PEFT

    PT --- AW
    TF --- GC
    US --- SM
    PEFT --- AZ
    HF --- K8

    AW --- DK
    SM --- ML
    GC --- DK
    AZ --- ML
    K8 --- DK

    PY --- PT
    TS --- OA
    GO --- K8

    classDef core fill:#1C3C3C,stroke:#58A6FF,stroke-width:2px,color:#E6EDF3;
    classDef proto fill:#1C1C3C,stroke:#D97757,stroke-width:2px,color:#E6EDF3;
    classDef rag fill:#3C1C3C,stroke:#A855F7,stroke-width:2px,color:#E6EDF3;
    classDef ml fill:#3C2C1C,stroke:#F59E0B,stroke-width:2px,color:#E6EDF3;
    classDef cloud fill:#1C3C1C,stroke:#22C55E,stroke-width:2px,color:#E6EDF3;
    classDef lang fill:#3C1C1C,stroke:#EF4444,stroke-width:2px,color:#E6EDF3;

    class LG,CR,OA,AS,AG core;
    class MCP,ACP,A2A proto;
    class RAG,FT,PE,MG rag;
    class PT,TF,US,PEFT,HF ml;
    class AW,SM,GC,AZ,K8,DK,ML cloud;
    class PY,TS,GO lang;
```

---

### 📊 Live Metrics

<div align="center">

![GitHub Stats](https://github-readme-stats.vercel.app/api?username=Shivamjohri247&show_icons=true&theme=tokyonight&hide_border=true&count_private=true&include_all_commits=true&custom_title=GitHub%20Overview&title_color=58A6FF&icon_color=58A6FF&text_color=E6EDF3&bg_color=0D1117)

![Top Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=Shivamjohri247&layout=compact&theme=tokyonight&hide_border=true&title_color=58A6FF&text_color=E6EDF3&bg_color=0D1117&langs_count=8)

![GitHub Streak](https://github-readme-streak-stats.herokuapp.com/?user=Shivamjohri247&theme=tokyonight&hide_border=true&background=0D1117&stroke=58A6FF&ring=58A6FF&fire=58A6FF&currStreakLabel=58A6FF)

</div>

---

### 🚀 Featured Projects

<table>
<tr>
<td colspan="2">

**[diff-guard](https://github.com/Shivamjohri247/diff-guard)** &nbsp; [![PyPI](https://img.shields.io/pypi/v/diffguard-cli.svg?style=flat-square&color=58A6FF)](https://pypi.org/project/diffguard-cli/) [![GitHub Stars](https://img.shields.io/github/stars/Shivamjohri247/diff-guard?style=flat-square&color=D97757)](https://github.com/Shivamjohri247/diff-guard/stargazers)

Blast-radius analyzer for AI-generated code changes. Detects unintended modifications by AI coding agents, measures downstream impact via import-chain analysis, and blocks dangerous commits before they reach production. Works as a git pre-commit hook with integrations for Claude Code, Cursor, Windsurf, and any AI agent. Supports Python (AST), JS/TS, Go, Rust, Java, and more.

`Python` `CLI` `AST Analysis` `Git Hooks` `AI Safety` `PyPI` `Open Source`

</td>
</tr>
<tr>
<td width="50%">

**[llm-finetuning](https://github.com/Shivamjohri247/llm-finetuning)** &nbsp; [![GitHub Stars](https://img.shields.io/github/stars/Shivamjohri247/llm-finetuning?style=flat-square&color=D97757)](https://github.com/Shivamjohri247/llm-finetuning/stargazers)

Reusable pipeline for fine-tuning SLMs on downstream tasks using PEFT/LoRA for efficient resource usage and faster inference. Supports 4-bit quantization, gradient checkpointing, and multi-GPU training.

`Python` `PyTorch` `PEFT` `LoRA` `Quantization` `HuggingFace`

</td>
<td width="50%">

**[GenNER](https://github.com/Shivamjohri247/GenNER)** &nbsp; [![GitHub Stars](https://img.shields.io/github/stars/Shivamjohri247/GenNER?style=flat-square&color=D97757)](https://github.com/Shivamjohri247/GenNER/stargazers)

Named Entity Recognition using Generative AI approaches with Transformer models. Explores seq2seq and prompt-based NER for low-resource domains.

`NER` `Transformers` `HuggingFace` `Generative AI` `Low-Resource`

</td>
</tr>
<tr>
<td width="50%">

**[med-text-classify](https://github.com/Shivamjohri247/med-text-classify)**

Medical text classification using NLP techniques for healthcare domain applications. Benchmarks traditional ML vs transformer approaches on clinical datasets.

`Python` `NLP` `Healthcare` `Transformers` `Clinical NLP`

</td>
<td width="50%">

**[emotion-classifier](https://github.com/Shivamjohri247/emotion-classifier)**

Emotion detection in text using sequence classification and HuggingFace Transformers. Multi-label classification with confidence calibration.

`NLP` `Classification` `HuggingFace` `Transformers` `Multi-label`

</td>
</tr>
</table>

---

### 🏢 Experience Highlights

| Company | Role | Impact |
|---------|------|--------|
| **Suzega** | **AI Architect, Applied R&D & Industry Solutions** | Architected multi-agent orchestration platform for Manufacturing & FinServ • Designed governance framework for 50+ autonomous agents • Reduced agent hallucination rate 73% via guardrails • Led 8-person applied AI team |
| **EPAM Systems** | **Senior Data Scientist** | Fine-tuned NER models for legal/financial domains (F1 +18%) • Optimized LLM query latency 62% via semantic caching • Built vector search infrastructure serving 10M+ docs |
| **Accenture** | **ML Engineering Senior Analyst** | Delivered GenAI accelerators for 12 enterprise clients • Built recommendation engines processing 500M+ events/day • Led computer vision pipeline for document intelligence |
| **TCS** | **Machine Learning Engineer** | Developed OCR/NLP solutions for BFSI document processing • Research in few-shot learning for low-resource languages • 3 patents filed in document intelligence |

---

### 🏆 Certifications

<div align="center">

![Anthropic](https://img.shields.io/badge/Anthropic-Claude_Code_in_Action-D97757?style=flat-square&logo=anthropic&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-AI_Practitioner-FF9900?style=flat-square&logo=amazon-aws&logoColor=white)
![GCP](https://img.shields.io/badge/GCP-Associate_Cloud_Engineer-4285F4?style=flat-square&logo=google-cloud&logoColor=white)
![DeepLearning.AI](https://img.shields.io/badge/DeepLearning.AI-Generative_AI_&_LLMs-0056D2?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEycyA0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMSAxN2gxdjVoLTF6bTAgLTloMXYtNWgtMXoiLz48L3N2Zz4=)

</div>

---

### 📈 GitHub Activity

<p align="center">
  <img src="./profile-summary-card-output/github_dark/0-profile-details.svg" alt="Contribution Graph" width="100%" />
</p>

<p align="center">
  <img src="./profile-summary-card-output/github_dark/3-stats.svg" alt="Stats" width="32%" />
  <img src="./profile-summary-card-output/github_dark/4-productive-time.svg" alt="Productive Time" width="32%" />
  <img src="./profile-summary-card-output/github_dark/1-repos-per-language.svg" alt="Top Languages" width="32%" />
</p>

---

### 🤝 Let's Connect

<div align="center">

| Platform | Handle | Engagement |
|----------|--------|------------|
| **LinkedIn** | [shivam-johri](https://linkedin.com/in/shivam-johri) | ●●●●●●● |
| **Email** | [shivamjohri247@gmail.com](mailto:shivamjohri247@gmail.com) | ●●○○○○○ |
| **Twitter/X** | [@sicdrummer247](https://twitter.com/sicdrummer247) | ●●●○○○○ |
| **GitHub** | [Shivamjohri247](https://github.com/Shivamjohri247) | ●●●●●○○ |
| **Blog** | [shivamjohri247.github.io](https://shivamjohri247.github.io) | ●●○○○○○ |

</div>

<div align="center">

**"Shipping production-grade agentic AI for enterprise scale"**  
*Open to collaborating on Agentic AI, Multi-Agent Systems & Production-Scale LLM Projects*

</div>

---

<div align="center">
<img src="https://readme-svg-wave-divider-generator.vercel.app/api/wave?type=sine&width=1200&height=120&amplitude=40&frequency=2&layers=3&color_top=58A6FF&color_bottom=0D1117&opacity=1&gradient=true&mirror=true&flip=false&animate=true&speed=6&text=SHIVAM%20JOHRI&text_color=E6EDF3&text_size=24&text_style=bold&text_stroke_color=0D1117&text_stroke_width=1" alt="Wave Divider" />
</div>