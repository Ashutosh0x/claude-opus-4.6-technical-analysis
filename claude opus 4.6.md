---
stylesheet: https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown.min.css
body_class: markdown-body
css: |-
  .markdown-body { max-width: 900px; margin: 0 auto; padding: 2rem; }
  table { width: 100%; font-size: 0.9em; }
  h1, h2, h3 { page-break-after: avoid; }
  .page-break { page-break-after: always; }
pdf_options:
  format: A4
  margin: 20mm
  displayHeaderFooter: true
  headerTemplate: '<span style="font-size:9px;width:100%;text-align:center;">Claude Opus 4.6 — Comprehensive Technical Analysis</span>'
  footerTemplate: '<span style="font-size:9px;width:100%;text-align:center;"><span class="pageNumber"></span> / <span class="totalPages"></span></span>'
---

# Claude Opus 4.6 — Comprehensive Technical Analysis

> **Disclaimer:** Claude Opus 4.6 is proprietary. Anthropic has not published exact architecture specs, parameter counts, or training details. Numbers in this document are **speculative estimates** based on publicly available information about comparable frontier models, Anthropic's system card (Feb 2026), official announcements, and independent analysis. Where facts are confirmed, they are cited; where speculative, they are marked as such.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture — Mixture-of-Experts Transformer](#2-architecture--mixture-of-experts-transformer)
3. [Scale & Parameter Estimates](#3-scale--parameter-estimates)
4. [Context Window & Long-Range Processing](#4-context-window--long-range-processing)
5. [Performance & Benchmarks](#5-performance--benchmarks)
6. [Training Data](#6-training-data)
7. [Training & Inference Infrastructure](#7-training--inference-infrastructure)
8. [Open-Weights Status](#8-open-weights-status)
9. [Hypothetical Open-Weight Release](#9-hypothetical-open-weight-release)
10. [Multimodality — Vision & Computer Use](#10-multimodality--vision--computer-use)
11. [Sampling & Decoding Strategies](#11-sampling--decoding-strategies)
12. [Prompt Caching](#12-prompt-caching)
13. [Normalization — RMSNorm vs LayerNorm](#13-normalization--rmsnorm-vs-layernorm)
14. [Activation Functions — SwiGLU](#14-activation-functions--swiglu)
15. [Learning Rate Schedule](#15-learning-rate-schedule)
16. [Loss Functions](#16-loss-functions)
17. [Numerical Stability & Mixed Precision](#17-numerical-stability--mixed-precision)
18. [Hardware Failure & Checkpointing](#18-hardware-failure--checkpointing)
19. [Continuous Batching & Inference Scheduling](#19-continuous-batching--inference-scheduling)
20. [Distillation & Model Compression](#20-distillation--model-compression)
21. [Expert Specialization Analysis](#21-expert-specialization-analysis)
22. [Structured Output / JSON Mode](#22-structured-output--json-mode)
23. [Watermarking](#23-watermarking)
24. [Logprobs & Uncertainty](#24-logprobs--uncertainty)
25. [Tool Use / Function Calling](#25-tool-use--function-calling)
26. [Multilingual Analysis](#26-multilingual-analysis)
27. [Curriculum Learning & Data Scheduling](#27-curriculum-learning--data-scheduling)
28. [Benchmark Contamination Testing](#28-benchmark-contamination-testing)
29. [Interpretability / Mechanistic Interpretability](#29-interpretability--mechanistic-interpretability)
30. [Streaming Architecture](#30-streaming-architecture)
31. [Thinking Guardrails & Adaptive Thinking](#31-thinking-guardrails--adaptive-thinking)
32. [Version History & Model Lineage](#32-version-history--model-lineage)
33. [Weight Initialization](#33-weight-initialization)
34. [Gradient Accumulation](#34-gradient-accumulation)
35. [Safety Classifiers](#35-safety-classifiers)
36. [A/B Testing & Deployment Pipeline](#36-ab-testing--deployment-pipeline)
37. [Fill-in-the-Middle (FIM) for Code](#37-fill-in-the-middle-fim-for-code)
38. [API Rate Limits & Operational Details](#38-api-rate-limits--operational-details)
39. [Economic Analysis](#39-economic-analysis)
40. [Regulatory & Legal Context](#40-regulatory--legal-context)
41. [Release & Competitive Context](#41-release--competitive-context)
42. [Who Built It](#42-who-built-it)
43. [Sources](#43-sources)

---

## 1. Overview

Claude Opus 4.6 is Anthropic's latest "frontier" model, released **February 5, 2026**, with massive scale and capabilities:

| Attribute | Value |
|---|---|
| **Release Date** | February 5, 2026 |
| **Context Window** | 1,000,000 tokens (beta) |
| **Architecture** | Mixture-of-Experts (MoE) Transformer (speculated) |
| **Total Parameters** | $\sim 2\text{–}5$ trillion (speculative) |
| **Active Parameters/Token** | $\sim 120\text{–}300$ billion |
| **API Pricing** | \$5 / \$25 per million tokens (input / output) |
| **Open Weights** | No — proprietary, API-only |
| **Knowledge Cutoff** | ~May 2025 (general), ~Aug 2025 (post-fine-tuning) |
| **Safety Level** | ASL-3 (Anthropic's classification) |

### Key Formula — Weight Size

$$\text{Size (bytes)} = N_{\text{params}} \times B_{\text{bytes/param}}$$

| Precision | Bytes/Param | Example (1B params) |
|---|---|---|
| FP32 | 4 | $\sim 4$ GB |
| FP16 / BF16 | 2 | $\sim 2$ GB |
| INT8 | 1 | $\sim 1$ GB |
| INT4 | 0.5 | $\sim 0.5$ GB |

---

## 2. Architecture — Mixture-of-Experts Transformer

Claude Opus 4.6 almost certainly uses **MoE layers**. In MoE transformers, each layer has many feed-forward "expert" modules, but only a small subset are routed for each token.

### How MoE Works

- **Sparse activation:** Only $2\text{–}4$ experts (out of $64\text{–}128$) process each token
- **Scaling:** MoE scales capacity without linearly scaling FLOPs
- **Expert Gating:** A learned gating network $G(x)$ routes tokens to relevant experts:

$$y = \sum_{i=1}^{k} G(x)_i \cdot E_i(x)$$

where $G(x)_i$ is the gating weight for expert $i$, $E_i(x)$ is the expert output, and $k$ is the number of active experts per token.

### Speculated Architecture Parameters

| Component | Estimated Value |
|---|---|
| Layers ($L$) | $\sim 160$ |
| Model Dimension ($d_{\text{model}}$) | $\sim 16{,}384\text{–}32{,}768$ |
| Attention Heads ($n_h$) | $\sim 128$ |
| KV Heads (GQA) ($n_{kv}$) | $\sim 16$ |
| Head Dimension ($d_h$) | $\sim 128$ |
| FFN Dimension ($d_{ff}$) | $\sim 4 \times d_{\text{model}}$ |
| Number of Experts ($E$) | $\sim 64\text{–}128$ |
| Active Experts/Token ($k$) | $\sim 2\text{–}4$ |
| Expert Size | $\sim 10\text{–}20$B params each |

### Comparison: MoE Models

| Model | Total Params | Active Params | Experts | Performance |
|---|---|---|---|---|
| Mixtral 8×7B | 47B | 13B | 8 | Matches Llama-2-70B |
| DBRX | 132B | 36B | 16 | Competitive with GPT-3.5 |
| Llama 4 Behemoth | $\sim 2$T | $\sim 288$B | Many | Frontier-class |
| **Claude Opus 4.6** | $\sim 2\text{–}5$**T** | $\sim 120\text{–}300$**B** | $\sim 64\text{–}128$ | **SOTA on many benchmarks** |

### MoE Layout (2T total model estimate)

```
┌─────────────────────────────────────────────┐
│        Claude Opus 4.6 — MoE Layout         │
│              (2T total params)              │
├─────────────────────────────────────────────┤
│                                             │
│  Shared Components (~200–400B params)       │
│  ├─ Embedding layers:     ~50–100B          │
│  ├─ Attention (all layers): ~100–200B       │
│  ├─ LayerNorms / biases:    ~1–5B           │
│  └─ Output head:            ~50–100B        │
│                                             │
│  Expert FFNs (~1.6–1.8T params total)       │
│  ├─ 128 experts × ~12–14B each             │
│  ├─ Router/gating networks: ~1–5B           │
│  └─ Only 2–4 experts active per token       │
│                                             │
│  Active params per token: ~120–300B         │
│                                             │
└─────────────────────────────────────────────┘
```

### Attention Computation

Multi-head attention with grouped-query attention (GQA):

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V$$

where $Q \in \mathbb{R}^{n \times d_h}$, $K, V \in \mathbb{R}^{m \times d_h}$.

**Total attention parameters per layer:**

$$P_{\text{attn}} = d_{\text{model}} \times (n_h \cdot d_h + 2 \cdot n_{kv} \cdot d_h + n_h \cdot d_h)$$

**Total FFN parameters per expert (SwiGLU):**

$$P_{\text{ffn}} = 3 \times d_{\text{model}} \times d_{ff}$$

---

## 3. Scale & Parameter Estimates

### Raw Weight Size Calculations

$$\text{Size} = N_{\text{params}} \times B_{\text{bytes/param}}$$

| Precision | Bytes/Param | 2T Params | 3.5T Params | 5T Params |
|---|---|---|---|---|
| FP32 | 4 | 8.0 TB | 14.0 TB | 20.0 TB |
| BF16 / FP16 | 2 | 4.0 TB | 7.0 TB | 10.0 TB |
| FP8 | 1 | 2.0 TB | 3.5 TB | 5.0 TB |
| INT8 | 1 | 2.0 TB | 3.5 TB | 5.0 TB |
| INT4 | 0.5 | 1.0 TB | 1.75 TB | 2.5 TB |
| INT3 | 0.375 | 0.75 TB | 1.31 TB | 1.875 TB |
| INT2 | 0.25 | 0.5 TB | 0.875 TB | 1.25 TB |

---

## 4. Context Window & Long-Range Processing

Opus 4.6's $1\text{M}$-token context is a major advance. Standard transformers scale **quadratically** with context length:

$$\text{Attention complexity} = \mathcal{O}(n^2 \cdot d_h)$$

### KV Cache Memory Formula

$$M_{kv} = 2 \times L \times n_{kv} \times d_h \times S \times b$$

where $L$ = layers, $n_{kv}$ = KV heads, $d_h$ = head dimension, $S$ = sequence length, $b$ = bytes per element.

**Example:** For the speculated architecture with $S = 1{,}000{,}000$:

$$M_{kv} = 2 \times 160 \times 16 \times 128 \times 1{,}000{,}000 \times 2 \approx 1.25 \text{ TB}$$

### Long-Context Techniques

| Technique | How It Works |
|---|---|
| **Sparse / Top-k Attention** | Retrieve only most relevant keys per query |
| **Infini-attention** | Bounded-size compressive memory: $M_{\text{compress}} = \mathcal{O}(d_h^2)$ |
| **RoPE Scaling** | Extend rotary embeddings: $\theta_i' = \theta_i / s$ where $s > 1$ |
| **Compaction** | Summarize prior context with smaller model |

---

## 5. Performance & Benchmarks

| Benchmark | Score | Notes |
|---|---|---|
| **BigLaw Bench** (legal) | 90.2% | Highest of any Claude model; 40% perfect scores |
| **Terminal-Bench 2.0** (agentic CLI) | 65.4% | #1 among all models |
| **SWE-bench Verified** (coding) | $\sim$80.8% | Tied for industry leading |
| **OSWorld-Verified** (multi-step SW) | 72.7% | — |
| **GPQA-Diamond** (physics) | 91.3% | Beats GPT-4(o) |
| **ARC-AGI-2** (abstract reasoning) | 68.8% | Far above GPT-4/Gemini |
| **MMLU** (multitask knowledge) | 91.1% | 10-choice format |
| **GDPval-AA** (economics) | 1606 Elo | $+144$ over GPT-5.2 ($\sim$1462) |
| **Humanity's Last Exam** | #1 | Advanced reasoning & synthesis |

---

## 6. Training Data

### Chinchilla Scaling Law (Hoffmann et al., 2022)

The compute-optimal training rule:

$$D_{\text{optimal}} \approx 20 \times N$$

where $D$ = training tokens and $N$ = model parameters. For a $2$T-param model:

$$D_{\text{optimal}} \approx 20 \times 2 \times 10^{12} = 4 \times 10^{13} \text{ tokens} = 40\text{T tokens}$$

### Industry Comparisons

| Model | Params ($N$) | Training Tokens ($D$) | $D/N$ Ratio |
|---|---|---|---|
| Chinchilla | 70B | 1.4T | 20× |
| GPT-4 | $\sim$1.8T | $\sim$13T | $\sim$7× |
| Llama 3.1 405B | 405B | >15T | $\sim$37× |
| Llama 4 Behemoth | $\sim$2T | >30T | $\sim$15× |
| **Claude Opus 4.6** | $\sim 2\text{–}5$**T** | **Unknown** | **Est. 20–40T+** |

### Disclosed Data Sources (System Card)

- Broadly-crawled web text (up to May 2025)
- Licensed corpora and books
- Contractor-curated/annotated data
- Opted-in Claude user content
- Synthetic/self-generated data

---

## 7. Training & Inference Infrastructure

### Training (Estimated)

| Aspect | Estimate |
|---|---|
| GPU Count | 20,000–60,000 NVIDIA H100 |
| Training Duration | 3–6 months |
| Training Data | 20–40+ trillion tokens |
| Interconnect | NVLink / NVSwitch |
| Estimated FLOPs | $\sim 3.6 \times 10^{25}$ |

### Training FLOPs Estimate (Kaplan approximation)

$$C \approx 6 \times N \times D$$

For $N = 2 \times 10^{12}$ and $D = 30 \times 10^{12}$:

$$C \approx 6 \times 2 \times 10^{12} \times 30 \times 10^{12} = 3.6 \times 10^{26} \text{ FLOPs}$$

### Inference — GPU RAM Requirements

| Component | Params | BF16 Size | In GPU RAM? |
|---|---|---|---|
| Shared attention + embeddings | $\sim$300B | $\sim$600 GB | Yes (always) |
| Each expert FFN | $\sim$14B | $\sim$28 GB | Only if routed |
| All 128 experts | $\sim$1.7T | $\sim$3.4 TB | No — offload |
| Active experts (2–4) | $\sim$28–56B | $\sim$56–112 GB | Yes |
| **Minimum GPU RAM** | **$\sim$330–360B** | **$\sim$660–720 GB** | — |

### Hardware Configurations

| Setup | Total GPU RAM | Feasibility |
|---|---|---|
| $8\times$ H100 80GB | 640 GB | Tight, needs expert offloading |
| $16\times$ H100 80GB | 1.28 TB | Comfortable |

---

## 8. Open-Weights Status

Claude Opus 4.6 is **not open-sourced**:

- Available only via API (\$5/\$25 per million tokens)
- Weights preserved internally "at minimum for the lifetime of Anthropic"
- RSP v3.0 (Feb 2026) strengthens safeguards against weight theft
- No credible leaks or unauthorized weights have surfaced
- Contrast: Meta (Llama), xAI (Grok), Alibaba (Qwen) release open weights

---

## 9. Hypothetical Open-Weight Release

> **Caveat:** This section is entirely speculative.

### Sharding Formula

$$N_{\text{shards}} = \left\lceil \frac{\text{Total Size}}{\text{Shard Size}} \right\rceil$$

| Shard Size | Shards (2T @ BF16) | Shards (5T @ BF16) |
|---|---|---|
| 5 GB | $\sim$800 files | $\sim$2,000 files |
| 10 GB | $\sim$400 files | $\sim$1,000 files |
| 20 GB | $\sim$200 files | $\sim$500 files |

### GGUF Community Quantizations

| Quant Method | Bits/Param | Size (2T) | Size (5T) | Quality |
|---|---|---|---|---|
| Q8\_0 | $\sim$8.5 | $\sim$2.1 TB | $\sim$5.3 TB | Near-lossless |
| Q6\_K | $\sim$6.6 | $\sim$1.6 TB | $\sim$4.1 TB | Excellent |
| Q5\_K\_M | $\sim$5.7 | $\sim$1.4 TB | $\sim$3.6 TB | Very good |
| Q4\_K\_M | $\sim$4.8 | $\sim$1.2 TB | $\sim$3.0 TB | Good (most popular) |
| Q3\_K\_M | $\sim$3.9 | $\sim$0.98 TB | $\sim$2.4 TB | Acceptable |
| Q2\_K | $\sim$3.2 | $\sim$0.8 TB | $\sim$2.0 TB | Degraded |
| IQ2\_XXS | $\sim$2.1 | $\sim$0.5 TB | $\sim$1.3 TB | Research-grade |

### GPU Quantization Formats

| Format | Typical Bits | Size (2T) | Best For |
|---|---|---|---|
| GPTQ | 4-bit | $\sim$1.0–2.0 TB | GPU inference (AutoGPTQ) |
| AWQ | 4-bit | $\sim$1.0–1.2 TB | GPU inference (vLLM, TGI) |
| EXL2 | 2–6 bpw | $\sim$0.5–1.5 TB | ExLlamaV2 |
| HQQ | 2–4 bit | $\sim$0.5–1.0 TB | Half-Quadratic Quantization |
| AQLM | 2-bit | $\sim$0.5 TB | Extreme compression |

### Download Summary

| Scenario | Format | Size (2T) | Size (5T) |
|---|---|---|---|
| Full official | SafeTensors BF16 | $\sim$4.0 TB | $\sim$10 TB |
| Best quality quant | GGUF Q8\_0 | $\sim$2.1 TB | $\sim$5.3 TB |
| Best tradeoff | GGUF Q4\_K\_M | $\sim$1.2 TB | $\sim$3.0 TB |
| Minimum viable | GGUF Q2\_K | $\sim$0.5–0.8 TB | $\sim$1.3–2.0 TB |
| GPU-optimized | AWQ 4-bit | $\sim$1.0 TB | $\sim$2.5 TB |

---

## 10. Multimodality — Vision & Computer Use

### Vision Encoder Architecture

$$\text{Image} \xrightarrow{\text{Patch}} \text{ViT} \xrightarrow{\text{Project}} \text{LLM embedding space}$$

**Patch tokenization formula:**

$$N_{\text{visual\_tokens}} = \frac{H}{P} \times \frac{W}{P}$$

where $H, W$ = image dimensions, $P$ = patch size (typically 14 or 16 pixels).

| Image Resolution | Patch Size | Visual Tokens |
|---|---|---|
| $224 \times 224$ | 14 | 256 |
| $336 \times 336$ | 14 | 576 |
| $672 \times 672$ | 14 | 2,304 |
| $1344 \times 1344$ | 14 | 9,216 |

### Vision Encoder Parameters

| Component | Params | Size (BF16) |
|---|---|---|
| ViT-Large (ViT-L/14) | $\sim$307M | $\sim$614 MB |
| ViT-Huge (ViT-H/14) | $\sim$632M | $\sim$1.26 GB |
| ViT-Giant (ViT-G/14) | $\sim$1.8B | $\sim$3.6 GB |
| Cross-attention projector | $\sim$100–500M | $\sim$0.2–1.0 GB |
| **Total** | **$\sim$1–3B** | **$\sim$2–6 GB ($< 0.1\%$ of total)** |

### Visual Token Context Cost

$$\text{Effective text context} = 1{,}000{,}000 - N_{\text{images}} \times N_{\text{visual\_tokens/image}}$$

### Computer Use Pipeline

$$\text{Screenshot} \xrightarrow{\text{ViT}} \text{Visual tokens} \xrightarrow{\text{LLM}} \text{Action}(x, y, \text{click/type/scroll})$$

---

## 11. Sampling & Decoding Strategies

### Temperature Scaling

$$P(t_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where $z_i$ = logit for token $i$, $T$ = temperature.

| Temperature | Effect | Use Case |
|---|---|---|
| $T = 0$ | Greedy (argmax) | Deterministic code |
| $T = 0.3$ | Low randomness | Professional writing |
| $T = 0.7$ | Balanced (default) | General conversation |
| $T = 1.0$ | Full softmax | Creative writing |
| $T > 1.0$ | High randomness | Brainstorming |

### Top-$p$ (Nucleus) Sampling

Select smallest set $V_p$ such that:

$$\sum_{t_i \in V_p} P(t_i) \geq p$$

Then renormalize and sample from $V_p$ only.

### Repetition Penalty

$$z'_i = \begin{cases} z_i / \alpha & \text{if } z_i > 0 \text{ and } t_i \in \text{context} \\ z_i \times \alpha & \text{if } z_i \leq 0 \text{ and } t_i \in \text{context} \end{cases}$$

where $\alpha > 1$ penalizes already-seen tokens.

---

## 12. Prompt Caching

### KV Cache Reuse Formula

$$M_{\text{cached}} = 2 \times L \times n_{kv} \times d_h \times S_{\text{prefix}} \times b$$

### Cost Savings

| Operation | Cost (Opus 4.6) |
|---|---|
| Input (uncached) | \$5.00 / M tokens |
| Cache write | \$6.25 / M tokens ($1.25\times$) |
| Cache read (hit) | \$0.50 / M tokens ($0.1\times$) |
| Output | \$25.00 / M tokens |

**Example:** 10,000-token system prompt used 1,000 times:

$$\text{Without cache: } 1{,}000 \times 10{,}000 \times \$5/\text{M} = \$50.00$$

$$\text{With cache: } \$0.0625 + 1{,}000 \times 10{,}000 \times \$0.50/\text{M} = \$5.06$$

**Savings: $\sim 90\%$**

### Cache Storage Size (10K tokens)

$$M_{\text{cache}} = 2 \times 160 \times 16 \times 128 \times 10{,}000 \times 2 = 12.5 \text{ GB}$$

---

## 13. Normalization — RMSNorm vs LayerNorm

### LayerNorm (Original Transformer)

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu = \frac{1}{d}\sum x_i$, $\sigma^2 = \frac{1}{d}\sum(x_i - \mu)^2$. Parameters: $2 \times d_{\text{model}}$.

### RMSNorm (Likely Used by Opus 4.6)

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}}$$

Parameters: $d_{\text{model}}$ (scale $\gamma$ only, no shift).

- $\sim 10\text{–}15\%$ faster than LayerNorm
- Empirically equivalent quality

**Total RMSNorm parameters:**

$$P_{\text{norm}} = 2 \times L \times d_{\text{model}} = 2 \times 160 \times 16{,}384 = 5.24\text{M} \quad (\text{negligible})$$

---

## 14. Activation Functions — SwiGLU

### SwiGLU (Standard in Modern Transformers)

$$\text{SwiGLU}(x) = \left(\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}\right) W_{\text{down}}$$

where:

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

and $\sigma$ is the sigmoid function. Requires **three** weight matrices per FFN:

$$P_{\text{ffn}} = 3 \times d_{\text{model}} \times d_{ff}$$

**Why SwiGLU over ReLU:**
- $\sim 1\text{–}2\%$ better perplexity at same compute
- Smoother gradients → more stable training
- Multiplicative gating provides richer expressivity

---

## 15. Learning Rate Schedule

### Warmup + Cosine Decay

$$\eta(t) = \begin{cases} \eta_{\max} \cdot \dfrac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\[10pt] \eta_{\min} + \dfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\dfrac{\pi(t - T_{\text{warmup}})}{T_{\text{total}} - T_{\text{warmup}}}\right)\right) & t > T_{\text{warmup}} \end{cases}$$

### Typical Hyperparameters

| Parameter | Typical Value |
|---|---|
| Peak LR ($\eta_{\max}$) | $1 \times 10^{-4}$ to $3 \times 10^{-4}$ |
| Final LR ($\eta_{\min}$) | $\eta_{\max}/10$ to $\eta_{\max}/100$ |
| Warmup steps ($T_{\text{warmup}}$) | 2,000–5,000 |
| $\beta_1, \beta_2$ (Adam) | 0.9, 0.95 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 (max grad norm) |
| Batch size | 2–16M tokens |

### Batch Size Scaling

$$B_{\text{total}} = B_{\text{micro}} \times N_{\text{accum}} \times N_{\text{data\_parallel}}$$

---

## 16. Loss Functions

### Primary Pre-training Loss (Cross-Entropy)

$$\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

### MoE Auxiliary Losses

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha \cdot \mathcal{L}_{\text{balance}} + \beta \cdot \mathcal{L}_z$$

| Loss | Purpose | Typical Weight |
|---|---|---|
| $\mathcal{L}_{\text{LM}}$ | Language modeling | 1.0 |
| $\mathcal{L}_{\text{balance}}$ | Prevent expert collapse | 0.01–0.1 |
| $\mathcal{L}_z$ | Router logit stability | 0.001 |

### Load Balancing Loss

$$\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

where $f_i$ = fraction of tokens routed to expert $i$, $p_i$ = average gating probability for expert $i$.

### Perplexity

$$\text{PPL} = \exp(\mathcal{L}_{\text{LM}})$$

Frontier models typically achieve $\text{PPL} \approx 5\text{–}8$ on standard benchmarks.

---

## 17. Numerical Stability & Mixed Precision

### Format Comparison

| Format | Exponent | Mantissa | Range | Precision |
|---|---|---|---|---|
| FP32 | 8 bits | 23 bits | $\pm 3.4 \times 10^{38}$ | $\sim$7 digits |
| FP16 | 5 bits | 10 bits | $\pm 65{,}504$ | $\sim$3.3 digits |
| BF16 | 8 bits | 7 bits | $\pm 3.4 \times 10^{38}$ | $\sim$2.4 digits |
| FP8 (E4M3) | 4 bits | 3 bits | $\pm 448$ | $\sim$1.5 digits |
| FP8 (E5M2) | 5 bits | 2 bits | $\pm 57{,}344$ | $\sim$1.2 digits |

### Loss Scaling (for FP16 training)

$$\hat{\mathcal{L}} = s \cdot \mathcal{L}, \qquad \hat{g} = \frac{g}{s}$$

BF16 typically doesn't need this — hence it is preferred.

---

## 18. Hardware Failure & Checkpointing

### Failure Rate at Scale

With $N = 32{,}000$ GPUs, $\sim 1\%$ annual failure rate:

$$p_{\text{fail/GPU/day}} \approx \frac{0.01}{365} \approx 2.7 \times 10^{-5}$$

$$\mathbb{E}[\text{failures/day}] = 32{,}000 \times 2.7 \times 10^{-5} \approx 0.87$$

Over 60 days: $\sim 52$ GPU failures expected.

### Checkpointing Cost

Full model state (weights $+$ optimizer $+$ RNG):

$$M_{\text{state}} = N_{\text{params}} \times 16 \text{ bytes} = 2\text{T} \times 16 = 32 \text{ TB}$$

$$T_{\text{checkpoint}} = \frac{32{,}000 \text{ GB}}{100 \text{ GB/s}} \approx 320 \text{ s} \approx 5.3 \text{ min}$$

---

## 19. Continuous Batching & Inference Scheduling

### GPU Utilization

$$\text{Utilization}_{\text{continuous}} = \frac{\sum_i T_i^{\text{compute}}}{B \times T_{\max}} \approx 2\text{–}3\times \text{ vs static}$$

### Iteration-Level Scheduling

```
Step 1: [User A token 1,  User B token 45, User C token 200, ...]
Step 2: [User A token 2,  User B token 46, User D token 1 (new!), ...]
```

---

## 20. Distillation & Model Compression

### Knowledge Distillation Loss

$$\mathcal{L}_{\text{distill}} = (1 - \alpha)\,\mathcal{L}_{\text{CE}}(y, \hat{y}) + \alpha\, T^2 \cdot \text{KL}\!\left(P_{\text{teacher}}^T \;\|\; P_{\text{student}}^T\right)$$

where $T$ = temperature for softening, $\alpha$ = interpolation weight.

| Model | Speculated Size | Distilled From |
|---|---|---|
| Opus 4.6 | $2\text{–}5$T total | Pre-trained from scratch |
| Sonnet 4.6 | $\sim$200–500B? | Likely distilled from Opus |
| Haiku 4.6 | $\sim$30–70B? | Likely distilled from Sonnet/Opus |

### Structured Pruning

$$W_{\text{pruned}} = W \odot M, \quad M_{ij} = \mathbf{1}[|W_{ij}| > \theta]$$

---

## 21. Expert Specialization Analysis

### Expert Utilization Entropy

$$H_{\text{expert}} = -\sum_{i=1}^{E} p_i \log p_i$$

- $H = \log E$ → perfectly balanced
- $H \ll \log E$ → some experts dominate (collapse risk)

### Expert Correlation Matrix

$$C_{ij} = \text{Corr}\!\left(\mathbf{1}[\text{expert } i \text{ active}],\; \mathbf{1}[\text{expert } j \text{ active}]\right)$$

---

## 22. Structured Output / JSON Mode

### Constrained Decoding

$$P'(t_i) = \begin{cases} P(t_i) / Z & \text{if } t_i \text{ is valid given grammar state} \\ 0 & \text{otherwise} \end{cases}$$

where $Z = \sum_{j \in \text{valid}} P(t_j)$ is the normalizing constant.

---

## 23. Watermarking

### Statistical Watermarking (Kirchenbauer et al., 2023)

At each step, partition vocabulary using $h_t = \text{Hash}(t_{i-1})$:

$$\text{Green list} = \{t : h_t(t) < |V|/2\}$$

$$z'_i = z_i + \delta \quad \text{if } t_i \in \text{green list}$$

**Detection:**

$$z\text{-score} = \frac{|G| - T/2}{\sqrt{T/4}}$$

If $z > 4$, almost certainly watermarked.

---

## 24. Logprobs & Uncertainty

### Log-Probabilities

$$\text{logprob}(t_i) = \log P(t_i \mid x_{<i})$$

### Entropy as Uncertainty

$$H(t) = -\sum_i P(t_i \mid x_{<t}) \log P(t_i \mid x_{<t})$$

### Expected Calibration Error

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left|\text{acc}(b) - \text{conf}(b)\right|$$

---

## 25. Tool Use / Function Calling

### Agent Loop — Token Cost

Input grows each turn (all previous messages re-sent):

$$C_{\text{input}}(i) = C_{\text{system}} + \sum_{j=1}^{i-1} \left(C_{\text{output}}(j) + C_{\text{tool\_result}}(j)\right)$$

$$C_{\text{agent}} = \sum_{i=1}^{N_{\text{turns}}} \left(C_{\text{input}}(i) + C_{\text{output}}(i)\right)$$

**Example:** 20-turn agent loop, 50K-token conversations:

$$\text{Without caching: } \approx 20 \times 50{,}000 \times \$5/\text{M} + 20 \times 2{,}000 \times \$25/\text{M} = \$6.00$$

$$\text{With caching: } \approx \$1.50 \quad (70\% \text{ savings})$$

---

## 26. Multilingual Analysis

### Tokenizer Fertility

$$F_{\text{lang}} = \frac{N_{\text{tokens}}}{N_{\text{characters or words}}}$$

Higher fertility $=$ less efficient $=$ more expensive per unit of meaning.

| Language | Tokens/Word | Effective Context (1M tokens) |
|---|---|---|
| English | $\sim$1.3 | $\sim$750K words |
| Spanish | $\sim$1.5 | $\sim$667K words |
| Chinese | $\sim$1.5–2.0/char | $\sim$500K–667K chars |
| Japanese | $\sim$2.0–3.0/char | $\sim$333K–500K chars |
| Arabic | $\sim$2.0 | $\sim$500K words |
| Hindi | $\sim$3.0–4.0 | $\sim$250K–333K words |

---

## 27. Curriculum Learning & Data Scheduling

### Data Mix Evolution

| Phase | % Compute | Data Emphasis |
|---|---|---|
| Phase 1 (0–60%) | 60% | Broad web text |
| Phase 2 (60–85%) | 25% | Higher-quality (Wikipedia, books) |
| Phase 3 (85–95%) | 10% | Code, math, reasoning |
| Phase 4 (95–100%) | 5% | Instruction-following |

### Annealing

$$\eta_{\text{anneal}} = \eta_{\min} + (\eta_{\text{current}} - \eta_{\min}) \cdot \cos\!\left(\frac{\pi t}{2 T_{\text{anneal}}}\right)$$

---

## 28. Benchmark Contamination Testing

### N-gram Overlap

$$\text{Contamination}(B) = \frac{|\{x \in B : \exists\, d \in D,\; \text{ngram\_overlap}(x, d) > \theta\}|}{|B|}$$

### Detection Methods

| Method | Description |
|---|---|
| N-gram overlap | Check for verbatim matches in training data |
| Canary strings | Plant unique strings; test if model completes them |
| Rephrased evaluation | Large drops = memorization, not understanding |

---

## 29. Interpretability / Mechanistic Interpretability

### Sparse Autoencoders (SAEs)

$$h = \text{ReLU}(W_{\text{enc}} \cdot x + b_{\text{enc}})$$

$$\hat{x} = W_{\text{dec}} \cdot h + b_{\text{dec}}$$

$$\mathcal{L}_{\text{SAE}} = \|x - \hat{x}\|^2 + \lambda \|h\|_1$$

Anthropic's published work:
- Mapped **millions of features** in Claude 3 Sonnet (May 2024)
- Found features for cities, code languages, emotions, safety concepts
- Can **steer behavior** by amplifying/suppressing features

---

## 30. Streaming Architecture

### Latency Breakdown

$$T_{\text{total}} = T_{\text{TTFT}} + N_{\text{output}} \times T_{\text{per\_token}} + T_{\text{network}}$$

| Phase | Typical Latency |
|---|---|
| TTFT (short prompt) | 0.5–2s |
| TTFT (100K context) | 5–15s |
| TTFT (1M context) | 30–120s |
| Per-token generation | 15–30ms |
| 500-token response | 8–17s |

---

## 31. Thinking Guardrails & Adaptive Thinking

### Effort Levels

| Level | Behavior |
|---|---|
| `low` | Minimal thinking, fast/cheap |
| `medium` | Balanced |
| `high` (default) | Deep, selective extended thinking |
| `max` | Maximum depth, revisits, caution |

### Speculative Internal Model (Not Official)

$$T = f(C \times E)$$

$$D \approx k \times C \times E$$

where $C$ = task complexity, $E \in \{1, 2, 3, 4\}$ = effort level, $T$ = trigger probability, $D$ = thinking depth.

### Safety Integration

- $0\%$ attack success on agentic coding attacks
- Constitutional AI checks during thinking
- Compaction at $\sim 5\%$ of cases

---

## 32. Version History & Model Lineage

```
Claude 1.0 (Mar 2023)
 └─ Claude 2.0 (Jul 2023)
     └─ Claude 3 (Mar 2024): Haiku / Sonnet / Opus
         └─ Claude 3.5 Sonnet (Jun 2024)
             └─ Claude 4.6 family (Feb 2026)
                 ├─ Opus 4.6    ← THIS
                 ├─ Sonnet 4.6
                 └─ Haiku 4.6 (?)
```

| Feature | Claude 3 Opus | Claude 4.6 Opus |
|---|---|---|
| Context | 200K | 1M (beta) |
| Architecture | Dense (likely) | MoE (speculated) |
| Params (est.) | $\sim$200–400B | $\sim$2–5T |
| Thinking | Extended (optional) | Adaptive (default) |
| Coding | Good | SOTA (80.8% SWE-bench) |

---

## 33. Weight Initialization

### Standard:

$$W \sim \mathcal{N}\!\left(0, \frac{\sigma}{\sqrt{d_{\text{model}}}}\right)$$

### Scaled for Deep Networks ($L > 160$):

$$W_{\text{out}} \sim \mathcal{N}\!\left(0, \frac{\sigma}{\sqrt{2L}}\right)$$

### MoE Router:

$$W_{\text{router}} \sim \mathcal{N}(0, 0.01)$$

---

## 34. Gradient Accumulation

$$g_{\text{accumulated}} = \frac{1}{K}\sum_{k=1}^{K} g_k$$

$$B_{\text{eff}} = B_{\text{micro}} \times K \times N_{\text{DP}}$$

---

## 35. Safety Classifiers

```
User Input → [Input Classifier] → Model → [Output Classifier] → Response
                    ↓                              ↓
              Block if harmful              Block if harmful
```

- $\sim 1\text{–}10$B params total (negligible vs main model)
- Add $\sim 10\text{–}50$ ms latency per request

---

## 36. A/B Testing & Deployment Pipeline

### Canary Deployment

$$\text{Response} = \begin{cases} M_{\text{new}} & \text{with probability } p_{\text{canary}} \\ M_{\text{old}} & \text{with probability } 1 - p_{\text{canary}} \end{cases}$$

---

## 37. Fill-in-the-Middle (FIM) for Code

### FIM Training Mix

$$\text{FIM mix} = (1 - r) \times \text{autoregressive} + r \times \text{FIM}$$

Typically $r = 0.5$ (50% of code data).

---

## 38. API Rate Limits & Operational Details

| Tier | Requests/min | Tokens/min | Tokens/day |
|---|---|---|---|
| Free | 5 | 20,000 | 300,000 |
| Build | 50 | 40,000 | 1,000,000 |
| Scale | 1,000 | 400,000 | 50,000,000 |
| Enterprise | Custom | Custom | Custom |

### Cost Formula

$$\text{Cost} = \frac{N_{\text{input}} \times P_{\text{input}} + N_{\text{output}} \times P_{\text{output}} + N_{\text{thinking}} \times P_{\text{output}}}{10^6}$$

| Scenario | Input | Thinking | Output | Cost |
|---|---|---|---|---|
| Simple question | 100 | 0 | 200 | \$0.0055 |
| Complex reasoning | 1,000 | 10,000 | 500 | \$0.268 |
| Agentic (10 turns) | 50,000 | 50,000 | 5,000 | \$1.63 |
| Full 1M context | 1,000,000 | 5,000 | 2,000 | \$5.18 |

---

## 39. Economic Analysis

### Revenue Estimate (10M API requests/day)

$$\text{Daily revenue} = 10^7 \times (2{,}000 \times \$5/\text{M} + 500 \times \$25/\text{M}) = \$225{,}000/\text{day}$$

$$\text{Annual revenue} \approx \$82\text{M}$$

### Infrastructure Cost

$$\text{Serving cost} \approx 1{,}000 \times \$2.50/\text{hr} \times 8{,}760\text{ hr/yr} = \$21.9\text{M/yr}$$

**Gross margin: $\sim 70\text{–}75\%$**

---

## 40. Regulatory & Legal Context

### EU AI Act

- Opus 4.6 = **General-Purpose AI (GPAI)** model
- Estimated training compute: $\sim 3.6 \times 10^{25}$ FLOPs $>$ $10^{25}$ threshold → **systemic risk**
- Requirements: Adversarial testing, incident reporting, energy disclosure

---

## 41. Release & Competitive Context

| Date | Event |
|---|---|
| Feb 5, 2026 | Opus 4.6 released |
| Feb 5, 2026 | OpenAI releases GPT-5.3-Codex ($\sim$15 min after) |
| Feb 17, 2026 | Claude Sonnet 4.6 released |
| $\sim$Mar 5, 2026 | OpenAI releases GPT-5.4 |

### Notable Achievements

- Solved the **graph decomposition conjecture** (31 explorations, $\sim$1 hour)
- #1 on Arena.ai leaderboard
- Agent teams for enterprise workflows
- Computer use (screenshots → actions)

### Security Issues

- Red-teamed in 30 minutes
- Service outage March 2, 2026
- \$200M Pentagon contract unraveled

---

## 42. Who Built It

| Person | Role |
|---|---|
| **Dario Amodei** | CEO & Co-founder |
| **Daniela Amodei** | President & Co-founder |
| **Boris Cherny** | Head of Claude Code |

Built by hundreds of Anthropic employees. 213-page system card credits it as a team effort.

---

## 43. Sources

- Anthropic — Claude Opus 4.6 system card (Feb 2026)
- Anthropic — RSP v3.0 and deprecation commitments
- Hoffmann et al. — "Training Compute-Optimal LLMs" (Chinchilla), 2022
- Jiang et al. — "Mixtral of Experts", 2024
- Munkhdalai et al. — "Infini-attention", 2024
- Meta — Llama 3.1 and Llama 4 model cards
- Kirchenbauer et al. — "A Watermark for Large Language Models", 2023
- Anthropic — "Scaling Monosemanticity" (Claude 3 Sonnet features), May 2024
- Leaked GPT-4 architecture analysis (SemiAnalysis)
- Arena.ai leaderboard (March 2026)

---

*Last updated: March 8, 2026*
