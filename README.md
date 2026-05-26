<div align="center">

<!-- Banner -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:00d4ff,100:ffb340&height=200&section=header&text=AI%20Engineering%20Mastery&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=From%20Zero%20to%20Production-Grade%20AI%20Engineer&descAlignY=60&descSize=18" />

<br/>

[![Track 01 — Core AI](https://img.shields.io/badge/Track%2001-Core%20AI%20%E2%80%94%207%20Topics-00d4ff?style=for-the-badge&labelColor=0f1114)](https://github.com)
[![Track 02 — Applied AI](https://img.shields.io/badge/Track%2002-Applied%20AI%20%E2%80%94%206%20Topics-ffb340?style=for-the-badge&labelColor=0f1114)](https://github.com)
[![Subtopics](https://img.shields.io/badge/Subtopics-80%2B-00e5a0?style=for-the-badge&labelColor=0f1114)](https://github.com)
[![Tools & Frameworks](https://img.shields.io/badge/Tools%20%26%20Frameworks-40%2B-ffffff?style=for-the-badge&labelColor=0f1114)](https://github.com)

<br/>

> **A complete, structured path covering Core AI foundations and Applied AI engineering —**  
> **every topic expanded into the exact subtopics, tools, and frameworks you need to master.**

<br/>

**[AI Engineering Mastery Website Link 👆](https://ai-engineering-mastery-101.netlify.app/)** 👆

</div>

---

## 📌 Table of Contents

### 🔵 Track 01 — Core AI
| # | Topic |
|---|-------|
| [01](#01--python-fundamentals) | Python Fundamentals |
| [02](#02--classical-machine-learning) | Classical Machine Learning |
| [03](#03--deep-learning) | Deep Learning |
| [04](#04--specializations-pick-12) | Specializations (Pick 1–2) |
| [05](#05--transformers) | Transformers |
| [06](#06--large-language-models) | Large Language Models |
| [07](#07--fine-tuning) | Fine-tuning |

### 🟡 Track 02 — Applied AI
| # | Topic |
|---|-------|
| [01](#01--llm-apis--prompt-engineering) | LLM APIs & Prompt Engineering |
| [02](#02--embeddings--vector-databases) | Embeddings & Vector Databases |
| [03](#03--rag--retrieval-augmented-generation) | RAG — Retrieval-Augmented Generation |
| [04](#04--ai-agents) | AI Agents |
| [05](#05--mcp--model-context-protocol) | MCP — Model Context Protocol |
| [06](#06--mlops--production) | MLOps & Production |

---

<br/>

# 🔵 Track 01 — Core AI

> `python → ml → dl → specializations → transformers → llms → fine-tuning`

---

## 01 · Python Fundamentals

> `Python` `NumPy` `Pandas` `Matplotlib` `Sklearn` `FastAPI` `PyTorch`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Core Python
- Variables, data types, control flow
- Functions, decorators, generators
- OOP — classes, inheritance, dunder methods
- File I/O, context managers
- Error handling & exceptions
- Comprehensions, lambda, map/filter
- Modules, packages, virtual environments
- Type hints & mypy basics

### Scientific Stack
- NumPy — arrays, broadcasting, vectorization
- Pandas — DataFrames, groupby, merge, time series
- Matplotlib & Seaborn — plots, subplots, styling

### ML & Web Libraries
- Scikit-learn — pipelines, estimators, transforms
- FastAPI — routes, pydantic, async, OpenAPI
- PyTorch — tensors, autograd, Dataset/DataLoader

</details>

**🛠 Tools:** `Python 3.11+` `NumPy` `Pandas` `Matplotlib` `Sklearn` `FastAPI` `PyTorch`

---

## 02 · Classical Machine Learning

> `Supervised` `Unsupervised` `XGBoost` `Feature Eng.`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Supervised Learning
- Linear & logistic regression
- Decision trees & random forests
- Gradient boosting — XGBoost, LightGBM
- Support vector machines
- k-Nearest neighbors
- Naive Bayes

### Unsupervised Learning
- K-means & hierarchical clustering
- PCA & dimensionality reduction
- DBSCAN, Gaussian Mixture Models
- Autoencoders (bridge to DL)

### Core Concepts
- Bias-variance tradeoff
- Cross-validation & regularization (L1/L2)
- Feature engineering & selection
- Hyperparameter tuning — grid, random, Bayesian
- Evaluation metrics — AUC, F1, RMSE, confusion matrix
- Imbalanced datasets — SMOTE, class weights

</details>

**🛠 Tools:** `Scikit-learn` `XGBoost` `LightGBM` `Optuna`

---

## 03 · Deep Learning

> `Backprop` `CNNs` `RNNs` `Optimization`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Foundations
- Perceptrons, MLPs, universal approximation
- Forward & backpropagation
- Activation functions — ReLU, GELU, Swish
- Loss functions — CE, MSE, focal loss
- Optimizers — SGD, Adam, AdamW, LR schedules

### Architectures
- Convolutional Neural Networks (CNNs)
- Recurrent Networks — RNN, LSTM, GRU
- Batch norm, dropout, weight initialization
- Residual connections & skip connections

### Training Techniques
- Data augmentation strategies
- Transfer learning & fine-tuning
- Early stopping, regularization
- Mixed precision training (fp16 / bf16)
- Gradient clipping & accumulation

</details>

**🛠 Tools:** `PyTorch` `Lightning` `Wandb` `TensorBoard`

---

## 04 · Specializations (Pick 1–2)

> `NLP` `Computer Vision` `Reinforcement Learning`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Natural Language Processing
- Tokenization — BPE, WordPiece, SentencePiece
- Word embeddings — Word2Vec, GloVe, FastText
- Sequence models — seq2seq, attention
- Text classification, NER, POS tagging
- Machine translation, summarization
- BERT, RoBERTa, T5 fine-tuning

### Computer Vision
- Image preprocessing & augmentation
- CNN architectures — VGG, ResNet, EfficientNet
- Object detection — YOLO, Faster R-CNN
- Semantic & instance segmentation
- Vision Transformers (ViT)
- Multimodal — CLIP, image-text alignment

### Reinforcement Learning
- MDPs, Bellman equation, value functions
- Q-learning, DQN, policy gradients
- Actor-critic — A2C, PPO, SAC
- RLHF — reward modeling, PPO from human feedback
- Multi-agent RL fundamentals

</details>

**🛠 Tools:** `HuggingFace` `OpenCV` `Gymnasium` `TRL`

---

## 05 · Transformers

> `Self-Attention` `Positional Enc.` `MoE` `FlashAttention`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Architecture Deep Dive
- Self-attention — Q, K, V matrices
- Multi-head attention
- Positional encodings — sinusoidal, RoPE, ALiBi
- Encoder, decoder, encoder-decoder variants
- Feed-forward sublayers & layer norm
- Pre-norm vs post-norm
- KV cache & inference optimization

### Variants & Extensions
- BERT-style — masked LM, encoder-only
- GPT-style — causal LM, decoder-only
- T5-style — seq2seq, encoder-decoder
- Efficient attention — FlashAttention, Sparse
- Mixture of Experts (MoE)
- Mamba & state space models (SSMs)

### Implementation
- Build transformer from scratch in PyTorch
- Implement attention with causal masking
- Read key papers — *Attention Is All You Need*, *BERT*, *GPT*

</details>

**🛠 Tools:** `PyTorch` `HuggingFace` `FlashAttention-2` `xFormers`

---

## 06 · Large Language Models

> `Pretraining` `RLHF` `Scaling Laws` `Quantization`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Pretraining
- Next-token prediction (causal LM objective)
- Data curation, deduplication, quality filtering
- Tokenizer training — BPE, SentencePiece
- Scaling laws — Chinchilla compute-optimal
- Distributed training — DDP, FSDP, tensor parallelism
- Mixed precision & gradient checkpointing

### Alignment & Post-training
- Instruction tuning (SFT)
- RLHF — reward model + PPO
- DPO — Direct Preference Optimization
- Constitutional AI & RLAIF
- Safety fine-tuning & refusals

### Model Families
- GPT family — GPT-2, 3, 4, o-series
- Llama 2 / 3, Mistral, Mixtral
- Gemma, Phi, Qwen, DeepSeek
- Claude model series

### Inference & Efficiency
- Quantization — GPTQ, GGUF, AWQ, INT4/INT8
- Speculative decoding
- Continuous batching, vLLM PagedAttention
- Prompt caching & KV cache management

</details>

**🛠 Tools:** `HuggingFace Hub` `vLLM` `llama.cpp` `Axolotl`

---

## 07 · Fine-tuning

> `LoRA` `QLoRA` `DPO` `PEFT`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Full Fine-tuning
- When to use full fine-tuning vs PEFT
- Dataset curation — instruction pairs, JSONL format
- SFT training loop with HuggingFace Trainer
- Learning rate warmup & scheduler tuning

### Parameter-Efficient Methods (PEFT)
- LoRA — rank, alpha, target modules
- QLoRA — 4-bit base + LoRA adapters
- Prefix tuning, prompt tuning, IA³
- Adapter layers

### Alignment Fine-tuning
- DPO — preference dataset formatting
- ORPO, SimPO
- Reward model training
- PPO with TRL library

### Evaluation & Deployment
- Perplexity, ROUGE, BLEU, BERTScore
- LLM-as-judge — MT-Bench, Alpaca Eval
- GGUF export & quantization for serving
- Merging adapters back to base model

</details>

**🛠 Tools:** `PEFT` `TRL` `Axolotl` `Unsloth` `LM Eval Harness`

---

<br/>

# 🟡 Track 02 — Applied AI

> `apis → embeddings → rag → agents → mcp → mlops`

---

## 01 · LLM APIs & Prompt Engineering

> `OpenAI SDK` `Anthropic SDK` `CoT` `Function Calling`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Working with APIs
- OpenAI SDK — chat completions, streaming
- Anthropic SDK — messages API, tool use
- Cohere, Mistral, Groq API patterns
- Rate limits, retries, error handling
- Async calls & concurrent requests

### Prompt Engineering
- Zero-shot, few-shot, chain-of-thought (CoT)
- Role prompting & system prompts
- ReAct, Tree of Thought, Self-consistency
- Structured output — JSON mode, function calling
- Prompt chaining & decomposition
- Context window management & chunking

### Evaluation
- PromptFoo for prompt testing
- LLM-as-judge patterns
- Regression testing across model versions
- RAGAS for RAG-specific evaluation

</details>

**🛠 Tools:** `OpenAI SDK` `Anthropic SDK` `LangChain` `PromptFoo` `LiteLLM`

---

## 02 · Embeddings & Vector Databases

> `HNSW` `Pinecone` `Qdrant` `Dense Retrieval`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Embeddings
- What embeddings encode, dimensionality
- Models — text-embedding-3, BGE, E5, nomic-embed
- Cosine similarity, dot product, euclidean distance
- Sparse vs dense embeddings
- Late interaction models — ColBERT
- Multimodal embeddings — CLIP

### Vector Databases
- FAISS — indexing types (Flat, IVF, HNSW)
- Pinecone — namespaces, metadata filtering
- Weaviate — hybrid search, modules
- Qdrant — payload filtering, collections
- Chroma — local dev, persistent storage
- pgvector — embeddings in PostgreSQL

### ANN Algorithms
- HNSW — hierarchical navigable small world
- IVF — inverted file index
- PQ — product quantization
- Approximate vs exact search tradeoffs

</details>

**🛠 Tools:** `FAISS` `Pinecone` `Qdrant` `Chroma` `pgvector` `sentence-transformers`

---

## 03 · RAG — Retrieval-Augmented Generation

> `Chunking` `Reranking` `HyDE` `RAGAS`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Naive RAG Pipeline
- Document loading — PDF, HTML, Markdown parsers
- Chunking strategies — fixed, recursive, semantic
- Embedding & indexing pipeline
- Query → retrieve → augment → generate
- Retrieval — top-k, similarity threshold

### Advanced RAG
- Hybrid search — BM25 + dense + RRF reranking
- Reranking — cross-encoder, Cohere Rerank
- HyDE — hypothetical document embeddings
- Multi-query retrieval & query rewriting
- Parent document retriever
- RAPTOR — recursive tree retrieval
- Knowledge graph RAG

### Agentic RAG
- Self-RAG — adaptive retrieval decisions
- CRAG — corrective RAG with grading
- RAG fusion
- Routing — query classification before retrieval

### Evaluation
- RAGAS — faithfulness, answer relevance, context precision
- DeepEval, TruLens
- Human eval vs automated eval

</details>

**🛠 Tools:** `LangChain` `LlamaIndex` `RAGAS` `DeepEval` `Cohere Rerank`

---

## 04 · AI Agents

> `ReAct` `LangGraph` `Multi-agent` `Tool Design`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Agent Fundamentals
- ReAct pattern — reason + act loop
- Tool calling / function calling
- Planning — task decomposition, subgoal generation
- Memory — in-context, external (vector store), episodic

### Agent Architectures
- Single agent with tools
- Multi-agent — orchestrator + worker pattern
- Supervisor agent pattern
- Hierarchical agents
- Parallelization patterns
- Human-in-the-loop (HITL)

### Tool Design
- Designing effective tools for agents
- Tool schemas — names, descriptions, parameters
- Error handling & retries in tool execution
- Stateful vs stateless tools

### Frameworks & Safety
- LangGraph — graph-based agent state machines
- CrewAI — role-based multi-agent
- AutoGen — conversational agents
- Smolagents (HuggingFace)
- Prompt injection defense
- Output validation & guardrails
- Cost & latency monitoring

</details>

**🛠 Tools:** `LangGraph` `CrewAI` `AutoGen` `Guardrails AI`

---

## 05 · MCP — Model Context Protocol

> `Protocol` `MCP Servers` `Tools / Resources` `JSON-RPC`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Protocol Fundamentals
- MCP architecture — host, client, server
- Transport layers — stdio, SSE, WebSocket
- Resources, tools, prompts — three primitives
- JSON-RPC 2.0 message format
- Capability negotiation & initialization

### Building MCP Servers
- MCP Python SDK — server setup
- Defining tools with `@mcp.tool` decorator
- Exposing resources with `@mcp.resource`
- Reusable prompt templates with `@mcp.prompt`
- Authentication & security in MCP servers

### MCP Clients & Integration
- Connecting to MCP servers from agents
- Claude Desktop MCP configuration
- Multi-server routing & aggregation
- MCP with LangChain / LangGraph
- Testing with MCP Inspector

### Production Patterns
- Stateless vs stateful MCP servers
- Rate limiting & quota management
- Error propagation in MCP
- Logging & observability for MCP tools

</details>

**🛠 Tools:** `MCP Python SDK` `MCP TS SDK` `Claude Desktop` `MCP Inspector`

---

## 06 · MLOps & Production

> `Docker` `vLLM` `Langfuse` `CI/CD` `Monitoring`

<details>
<summary><strong>📖 View Subtopics</strong></summary>

<br/>

### Experimentation & Tracking
- Experiment tracking — MLflow, Weights & Biases
- Dataset versioning — DVC, LakeFS
- Model registry & artifact management
- Hyperparameter sweep management

### Model Serving
- REST APIs — FastAPI + Uvicorn + Gunicorn
- Async inference with async endpoints
- Batching — static, dynamic, continuous
- vLLM for LLM serving
- Triton Inference Server
- BentoML & Ray Serve

### Containerization & Infra
- Docker — multi-stage builds for ML
- Kubernetes — deployments, services, HPAs
- GPU scheduling — NVIDIA device plugin
- Helm charts for ML workloads

### CI/CD & Monitoring
- GitHub Actions ML pipelines
- Automated retraining triggers
- Blue-green & canary model deployments
- Data drift — Evidently, Alibi Detect
- LLM tracing — Langfuse, LangSmith
- Cost monitoring for LLM APIs

### LLM-Specific Ops
- Prompt versioning & A/B testing
- Continuous eval pipelines in production
- Structured output validation
- Semantic caching strategies
- Token budget & context window management

</details>

**🛠 Tools:** `MLflow` `Wandb` `FastAPI` `Docker` `vLLM` `Langfuse` `Evidently`

---

<br/>

## 🗺 Suggested Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│                     CORE AI TRACK                           │
│                                                             │
│  Python  →  Classical ML  →  Deep Learning                  │
│       ↓                                                     │
│  Specializations (NLP / CV / RL)                            │
│       ↓                                                     │
│  Transformers  →  LLMs  →  Fine-tuning                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                   APPLIED AI TRACK                          │
│                                                             │
│  LLM APIs  →  Embeddings  →  RAG                            │
│       ↓                                                     │
│  Agents  →  MCP  →  MLOps & Production                      │
└─────────────────────────────────────────────────────────────┘
```

---

<br/>

<div align="center">

**Built for Enigmax Labs — AI Engineering Mastery Roadmap v1.0**  
`13 topics · 80+ subtopics · 2 tracks · 40+ tools & frameworks`

<br/>

*Star ⭐ this repo if you find it useful!*

</div>
