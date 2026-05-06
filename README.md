
# AI Engineer and Researcher Roadmap

This roadmap consists of three parallel tracks: **Core AI**, **Applied AI**, and **Research**.
Link: https://ai-engineer-researcher-roadmap.netlify.app/

## How to use this roadmap
Core AI and Applied AI run sequentially but can be interleaved from phase 3 onward. The Research track starts once you're inside transformers. You don't need to finish Core AI before starting Applied AI — start building basic apps at phase 3, and go deep on the theory in parallel.

---

## Track 1: Core AI

### Phase 1: Math foundations
**Duration:** 2–4 weeks  
**Why:** Every paper you'll ever read is written in linear algebra and probability. Without this, you're reading code without knowing the language. Don't skip it.

#### Key Topics
- **Linear algebra**: Vectors, matrices, dot products, eigenvalues, SVD
- **Calculus**: Derivatives, chain rule, partial derivatives, gradients
- **Probability & stats**: Distributions, Bayes' theorem, MLE, information theory
- **Optimization basics**: Loss surfaces, convexity, saddle points

#### Resources & Tags
3Blue1Brown, Gilbert Strang, Khan Academy

> **Milestone Checkpoint:** Can derive backpropagation by hand and explain cross-entropy loss without looking it up.

---

### Phase 2: Python + scientific stack
**Duration:** 2–3 weeks  
**Why:** Not just syntax — fluency. You should be able to implement a paper in Python faster than you can explain it in English.

#### Key Topics
- **Python fluency**: OOP, decorators, generators, type hints, dataclasses
- **NumPy**: Vectorization, broadcasting, einsum notation
- **Pandas**: Data manipulation, cleaning, merging, groupby
- **Matplotlib / Seaborn**: Visualization for analysis and debugging

#### Resources & Tags
fast.ai, Karpathy micrograd, numpy docs

> **Milestone Checkpoint:** Implement a neural net in pure NumPy — no frameworks. Backprop, gradient descent, the works.

---

### Phase 3: Classical ML
**Duration:** 2–3 weeks  
**Why:** The conceptual vocabulary of ML — loss functions, overfitting, regularization, hyperparameters — lives here. Don't call this 'fine-tuning'; it's model selection and optimization.

#### Key Topics
- **Supervised learning**: Linear/logistic regression, SVMs, decision trees
- **Unsupervised learning**: Clustering, PCA, dimensionality reduction
- **Model evaluation**: Train/val/test splits, cross-validation, metrics
- **Sklearn + pipelines**: End-to-end ML pipelines, preprocessing, grid search

#### Resources & Tags
Sklearn docs, ISLR textbook, StatQuest

> **Milestone Checkpoint:** Build an end-to-end ML pipeline from raw data to deployed model with proper evaluation. No shortcuts.

---

### Phase 4: Deep learning
**Duration:** 4–6 weeks  
**Why:** The substrate everything else is built on. PyTorch fluency is non-negotiable — you'll use it to implement every architecture from here forward.

#### Key Topics
- **ANN from scratch**: Forward pass, backprop, optimizers (SGD, Adam, AdamW)
- **CNNs**: Convolutions, pooling, ResNets, BatchNorm
- **RNNs / LSTMs**: Sequence modeling, vanishing gradient, attention precursors
- **PyTorch fluency**: Autograd, DataLoaders, training loops, GPU, debugging

#### Resources & Tags
Karpathy Neural Nets Zero to Hero, fast.ai Part 1, d2l.ai

> **Milestone Checkpoint:** Implement a ResNet and an LSTM from scratch in PyTorch. Train on a real dataset. Debug the training curves.

---

### Phase 5: Transformer architecture
**Duration:** 3–4 weeks  
**Why:** The single most important architecture in AI right now. Go deep — not just 'attention is all you need' but why every design decision was made.

#### Key Topics
- **Self-attention**: Q/K/V, scaled dot-product, masking, complexity
- **Multi-head attention**: Parallel heads, projection, concatenation
- **Positional encoding**: Sinusoidal, learned, RoPE, ALiBi
- **Full transformer stack**: LayerNorm, FFN, residuals, pre-norm vs post-norm

#### Resources & Tags
Annotated Transformer, Jay Alammar, Karpathy makemore, **key topic**

> **Milestone Checkpoint:** Implement a GPT-2 class transformer from scratch. Train a character-level LM. Understand every line.

---

### Phase 6: Transformer variants + modern architectures
**Duration:** 2–3 weeks  
**Why:** The field moves fast. You need to recognize and reason about architectural differences — not just run HuggingFace code.

#### Key Topics
- **Encoder-only**: BERT, RoBERTa — bidirectional, masked LM
- **Decoder-only**: GPT family — causal, autoregressive generation
- **Encoder-decoder**: T5, BART — seq2seq tasks
- **MoE + SSMs**: Mixture of Experts, Mamba, hybrid architectures

#### Resources & Tags
HuggingFace model cards, Yannic Kilcher, Papers with Code

> **Milestone Checkpoint:** Explain the architectural tradeoffs of BERT vs GPT vs T5 vs Mamba to a non-specialist. Then read the original papers.

---

### Phase 7: LLMs — pretraining + scaling
**Duration:** 3–4 weeks  
**Why:** This is where you understand what frontier labs actually do. Pretraining is the moat. Understanding it separates serious engineers from API consumers.

#### Key Topics
- **Pretraining objectives**: Next-token prediction, masked LM, span corruption
- **Data pipelines**: Tokenization (BPE, SentencePiece), data mixture, deduplication
- **Scaling laws**: Chinchilla compute-optimal, emergent capabilities
- **Infrastructure basics**: Tensor/pipeline/data parallelism, mixed precision, FSDP

#### Resources & Tags
Chinchilla paper, GPT-4 tech report, LLaMA papers, **key topic**

> **Milestone Checkpoint:** Read Chinchilla paper and the LLaMA-2 paper. Summarize the data and scaling decisions made. Critically evaluate them.

---

### Phase 8: Alignment — RL, RLHF, DPO
**Duration:** 3–4 weeks  
**Why:** This is how every frontier model you use is actually trained. RL is also the substrate of agents. Missing this means not understanding half of what matters.

#### Key Topics
- **RL fundamentals**: MDPs, policy gradient, PPO, reward modeling
- **RLHF pipeline**: SFT → reward model → PPO, InstructGPT
- **DPO / ORPO**: Direct preference optimization, offline RL for LLMs
- **Constitutional AI**: Anthropic's RLAIF approach, self-critique

#### Resources & Tags
InstructGPT paper, DPO paper, Spinning Up RL, **key topic**

> **Milestone Checkpoint:** Implement a toy RLHF loop: SFT a small model, train a reward model, run PPO. Understand where it can fail.

---

### Phase 9: Fine-tuning (all methods)
**Duration:** 2–3 weeks  
**Why:** One consolidated block — not scattered across three places. Covers everything from full fine-tune to parameter-efficient methods to preference tuning.

#### Key Topics
- **Full fine-tuning**: When to use it, catastrophic forgetting, compute cost
- **LoRA / QLoRA**: Low-rank adaptation, quantization, rank selection
- **SFT pipeline**: Dataset formatting, instruction templates, chat format
- **DPO / ORPO in practice**: Building preference datasets, training stability

#### Resources & Tags
LoRA paper, QLoRA paper, Axolotl, Unsloth

> **Milestone Checkpoint:** Fine-tune a 7B model on a custom task with QLoRA. Evaluate it properly against a baseline. Document what changed.

---

### Phase 10: Evaluation — first-class discipline
**Duration:** 2–3 weeks  
**Why:** Evals are how you know if anything you build actually works. This is criminally underrated. Senior engineers obsess over evals. Junior engineers eyeball outputs.

#### Key Topics
- **Benchmark design**: What makes a good eval, contamination, saturation
- **LLM-as-judge**: Pairwise evals, G-Eval, self-consistency, bias in judges
- **Human eval design**: Annotation frameworks, inter-rater agreement, sampling
- **Eval frameworks**: EleutherAI lm-eval, HELM, Braintrust, custom harnesses

#### Resources & Tags
HELM paper, G-Eval paper, Braintrust docs, **key topic**

> **Milestone Checkpoint:** Build a custom eval harness for a task you care about. Run it on 3 different models. Present findings with confidence intervals.

---

### Phase 11: Deployment + optimization
**Duration:** 2–3 weeks  
**Why:** A model that can't serve production traffic at acceptable cost is a research artifact. Inference optimization is an engineering discipline in its own right.

#### Key Topics
- **Quantization**: INT8, INT4, GPTQ, AWQ, bitsandbytes
- **Inference serving**: vLLM, TGI, TensorRT-LLM, KV cache
- **Speculative decoding**: Draft models, acceptance rates, latency tradeoffs
- **Deployment patterns**: Batch vs streaming, load balancing, autoscaling

#### Resources & Tags
vLLM docs, FlashAttention, GPTQ paper

> **Milestone Checkpoint:** Deploy a fine-tuned model with vLLM. Profile its throughput and latency. Optimize it with quantization and measure the delta.

---

## Track 2: Applied AI

### Phase 1: LLM APIs + SDK fluency
**Duration:** 1–2 weeks  
**Why:** Start here the moment you're through Core AI phase 3. Building is the best way to learn. Go raw first — understand what the abstractions are hiding.

#### Key Topics
- **OpenAI + Anthropic SDKs**: Messages API, streaming, token counting, error handling
- **Raw HTTP calls**: Understand the wire format — don't rely on SDK magic
- **Open source models**: HuggingFace transformers, Ollama for local inference
- **Model selection**: Capability vs cost tradeoffs, context windows, rate limits

#### Resources & Tags
Anthropic docs, OpenAI docs, HuggingFace

> **Milestone Checkpoint:** Build a multi-model wrapper that routes requests to different providers based on task type and cost. No LangChain.

---

### Phase 2: Prompt engineering
**Duration:** 1–2 weeks  
**Why:** Not just a beginner skill — serious prompt engineering is system design. How you structure context, roles, and instructions determines reliability at scale.

#### Key Topics
- **System prompt design**: Role framing, persona, output constraints, format control
- **Chain-of-thought**: CoT, self-consistency, step-back prompting
- **Few-shot design**: Example selection, format consistency, contrastive examples
- **Prompt testing**: A/B testing prompts, regression suites, edge case coverage

#### Resources & Tags
Anthropic prompt engineering guide, DSPY, PromptLayer

> **Milestone Checkpoint:** Build a prompt evaluation pipeline. A/B test 5 prompt variants on a real task. Report results with statistical rigor.

---

### Phase 3: Structured outputs + tool/function calling
**Duration:** 1–2 weeks  
**Why:** This is the backbone of every production agent and data pipeline. JSON mode, tool use, and structured extraction are not optional — they are the interface between LLMs and systems.

#### Key Topics
- **Function calling**: Tool schemas, parallel tool use, error handling
- **Structured extraction**: Pydantic + Instructor, JSON mode, schema validation
- **Multi-step tool use**: Chaining tool calls, state accumulation across turns
- **Reliability patterns**: Retry logic, fallbacks, output validation loops

#### Resources & Tags
Instructor library, Anthropic tool use docs, Pydantic, **key topic**

> **Milestone Checkpoint:** Build a structured data extraction pipeline that takes unstructured documents and outputs validated Pydantic models. 95%+ accuracy.

---

### Phase 4: RAG — retrieval-augmented generation
**Duration:** 2–3 weeks  
**Why:** RAG is the most deployed LLM pattern in production. Go deeper than tutorials — understand why retrieval fails and how to fix it.

#### Key Topics
- **Embeddings + vector DBs**: Dense vs sparse, cosine similarity, Chroma, Pinecone, Qdrant
- **Chunking strategies**: Semantic, sliding window, document hierarchy
- **Advanced retrieval**: Hybrid search, reranking (Cohere, cross-encoders), HyDE
- **RAG evaluation**: Faithfulness, answer relevance, context recall — RAGAS

#### Resources & Tags
RAGAS, LlamaIndex, Qdrant docs, **key topic**

> **Milestone Checkpoint:** Build a RAG pipeline, evaluate it with RAGAS, identify the weakest component, improve it, and show the delta. Document everything.

---

### Phase 5: Orchestration frameworks
**Duration:** 1–2 weeks  
**Why:** Learn the abstractions but don't become dependent on them. They change fast. Understanding them lets you know when to use them and when to go raw.

#### Key Topics
- **LangChain**: Chains, retrievers, memory — know what it does under the hood
- **LlamaIndex**: Document ingestion, query engines, agent tools
- **Custom orchestration**: When and why to skip frameworks entirely
- **DSPY**: Programmatic prompt optimization, signatures, teleprompters

#### Resources & Tags
LangChain docs, LlamaIndex docs, DSPY paper

> **Milestone Checkpoint:** Re-implement your RAG pipeline in raw Python (no LangChain). Compare performance, latency, and maintainability. Know the tradeoffs.

---

### Phase 6: AI agents
**Duration:** 3–4 weeks  
**Why:** Agents are the direction everything is moving. This is where applied AI and Core AI (especially RL) reconnect — agent design is systems design.

#### Key Topics
- **Agent architectures**: ReAct, Plan-and-Execute, Reflexion, tree-of-thought
- **Memory systems**: Working memory, episodic memory, knowledge stores
- **Multi-agent systems**: Agent roles, orchestrator/subagent patterns, consensus
- **Agent evals**: Trajectory evaluation, tool-use accuracy, goal completion

#### Resources & Tags
AutoGen, CrewAI, LangGraph, Anthropic agents guide, **key topic**

> **Milestone Checkpoint:** Build a multi-agent system that completes a real-world task end-to-end (research, write, verify). Evaluate its reliability over 50 runs.

---

### Phase 7: Multimodal AI
**Duration:** 2–3 weeks  
**Why:** Vision-language models are shipping in every product category. Audio and video are next. This is no longer optional for a serious AI engineer.

#### Key Topics
- **Vision-language models**: GPT-4V, Claude vision, LLaVA, PaliGemma
- **Multimodal pipelines**: Image + text inputs, document parsing, chart understanding
- **Audio models**: Whisper, TTS APIs, speech-to-speech patterns
- **Multimodal RAG**: Embedding images, cross-modal retrieval, ColPali

#### Resources & Tags
OpenAI vision docs, Anthropic vision docs, ColPali paper

> **Milestone Checkpoint:** Build a document intelligence pipeline that handles PDFs with mixed text, tables, and images. Extract structured data from all three.

---

### Phase 8: Fine-tuning in practice
**Duration:** 2–3 weeks  
**Why:** Applied fine-tuning is different from the theory in Core AI — it's about dataset construction, iteration speed, and knowing when fine-tuning is even the right answer.

#### Key Topics
- **Dataset construction**: Synthetic data generation, data flywheel, quality filters
- **Training pipelines**: Axolotl, Unsloth, cloud GPU management, cost estimation
- **When NOT to fine-tune**: Prompt engineering vs RAG vs fine-tune decision framework
- **Post-training evals**: Regression testing, capability probing, benchmark comparison

#### Resources & Tags
Axolotl, Unsloth, Modal, Together AI

> **Milestone Checkpoint:** Fine-tune a model for a specific vertical task (e.g. legal, medical). Show it beats a prompted frontier model on your eval suite.

---

### Phase 9: LLMOps + production systems
**Duration:** 3–4 weeks  
**Why:** Production AI is 10x harder than demo AI. This is where engineering discipline separates the people who ship from the people who prototype.

#### Key Topics
- **Observability**: Tracing (LangSmith, Langfuse, Braintrust), logging, alerting
- **Evals in production**: Online evals, shadow mode testing, A/B model experiments
- **Guardrails + safety**: Input/output filtering, PII detection, jailbreak defense
- **Cost optimization**: Caching (semantic cache), model routing, prompt compression

#### Resources & Tags
Langfuse, Braintrust, LangSmith, Guardrails AI, **key topic**

> **Milestone Checkpoint:** Deploy a production LLM system with full observability, automated evals on every deployment, and a rollback mechanism. Document the architecture.

---

### Phase 10: Safety, red-teaming + responsible deployment
**Duration:** 1–2 weeks  
**Why:** Not optional for the researcher track. Understanding failure modes and adversarial inputs is part of building robust systems — and it matters for Enigmax verticals operating in regulated domains.

#### Key Topics
- **Red-teaming**: Jailbreaks, prompt injection, adversarial inputs, systematic testing
- **Bias + fairness evals**: Demographic parity, representation audits, fairness metrics
- **Hallucination mitigation**: Grounding, citations, factuality evals, abstention
- **Regulatory awareness**: EU AI Act basics, GDPR for AI, sector-specific compliance

#### Resources & Tags
Anthropic usage policies, NIST AI RMF, AI Safety fundamentals

> **Milestone Checkpoint:** Conduct a red-team exercise on your own deployed system. Document 10 failure modes. Implement mitigations for the top 3.

---

## Track 3: Research

### Phase 1: Paper reading methodology
**Duration:** Ongoing from Core phase 5  
**Why:** Most people read papers wrong. There's a system — and once you have it, you can extract the core idea of most papers in 20 minutes.

#### Key Topics
- **The 3-pass method**: Skim → understand → critique. Never start with the math.
- **What to look for**: Problem setup, key insight, experimental design, ablations
- **Paper annotation**: Margin notes, idea connections, open questions
- **Paper implementation**: The real test of understanding — can you reproduce key results?

#### Resources & Tags
How to read a paper (Keshav), Connected Papers, Semantic Scholar

> **Milestone Checkpoint:** Read and implement one foundational paper per week for 8 weeks. Start with: Attention Is All You Need, GPT-2, LoRA, InstructGPT, RLHF, DPO.

---

### Phase 2: Mechanistic interpretability
**Duration:** After Core phase 5–6  
**Why:** The field asking 'what is the model actually doing internally?' It's the most scientifically rigorous part of AI research right now — and the most underexplored commercially.

#### Key Topics
- **Circuits framework**: Features, neurons, attention heads as algorithms
- **Superposition**: Polysemanticity, feature geometry, sparse autoencoders
- **Activation patching**: Causal interventions, path patching, locating facts
- **SAE training**: Sparse autoencoders for feature extraction, monosemanticity

#### Resources & Tags
Anthropic interpretability team, TransformerLens, Neel Nanda, ARENA, **key topic**

> **Milestone Checkpoint:** Complete ARENA mech interp modules. Reproduce one result from an Anthropic or EleutherAI interpretability paper.

---

### Phase 3: Scaling laws + emergent behavior
**Duration:** After Core phase 7  
**Why:** Scaling laws are the closest thing AI has to a theoretical foundation right now. Understanding them is understanding why frontier models are built the way they are.

#### Key Topics
- **Chinchilla scaling**: Compute-optimal training, token-to-parameter ratios
- **Emergent capabilities**: Phase transitions, grokking, in-context learning theory
- **Data scaling**: Quality vs quantity tradeoffs, data mixing laws
- **Inference scaling**: Test-time compute, o1/o3 paradigm, chain-of-thought scaling

#### Resources & Tags
Chinchilla paper, Grokking paper, Pythia suite, Olmo

> **Milestone Checkpoint:** Train a series of small models at different scales. Fit a scaling law curve. Predict performance of a 10x larger model. Check the prediction.

---

### Phase 4: AI safety + alignment research
**Duration:** After Core phase 8  
**Why:** For the researcher track, this isn't optional ethics reading — it's where some of the most technically interesting open problems live. It also matters for Enigmax operating in regulated verticals.

#### Key Topics
- **Scalable oversight**: Debate, amplification, weak-to-strong generalization
- **Reward hacking**: Specification gaming, Goodhart's law, reward model collapse
- **Interpretability for safety**: Deceptive alignment, anomaly detection, circuit breaking
- **Constitutional AI**: Anthropic's approach, RLAIF, self-critique pipelines

#### Resources & Tags
Anthropic alignment team, ARC Evals, AI Safety fundamentals course

> **Milestone Checkpoint:** Read 10 alignment papers. Write a 2-page critical summary of the open problem you find most tractable. Publish it on The AI Stack.

---

### Phase 5: Novel architecture exploration
**Duration:** From Core phase 6 onward  
**Why:** The researcher track ultimately needs you to be able to identify what's missing in current architectures and have a view on what might work better.

#### Key Topics
- **SSMs (Mamba, S4)**: Linear recurrence, selective state spaces, vs attention
- **Mixture of Experts**: Sparse routing, load balancing, capacity factors
- **Efficient attention**: FlashAttention, linear attention, sliding window
- **Architecture ablations**: How to run controlled architecture experiments at small scale

#### Resources & Tags
Mamba paper, Switch Transformer, FlashAttention 2, MegaByte

> **Milestone Checkpoint:** Implement Mamba from scratch. Compare its performance vs a transformer on a sequence task. Write up the tradeoffs in a structured post.

---

### Phase 6: Original contribution pathway
**Duration:** 12–18 months in  
**Why:** The end goal of the research track: contributing something new. Doesn't need to be a published paper — a rigorous blog post, a well-documented experiment, or a new eval benchmark all count.

#### Key Topics
- **Finding open problems**: Reading limitations sections, attending NeurIPS/ICLR
- **Experiment design**: Controlled ablations, baselines, statistical significance
- **Scientific writing**: Abstract, intro, method, experiments, discussion structure
- **Public research**: Blog posts, GitHub releases, Twitter/X dissemination

#### Resources & Tags
NeurIPS, ICLR, arXiv, The AI Stack

> **Milestone Checkpoint:** Publish one original contribution — an experiment, a benchmark, or a rigorous replication study — with full code and writeup.

---
