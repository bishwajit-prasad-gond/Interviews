# Data Scientist Interview Prep — Part 2: Deep Learning, GenAI/LLMs, MLOps, Governance, Cases
Targets: Amex, HSBC, PwC, Deloitte (banking-focused)

---

## 1. Deep Learning Basics

**Q1. Explain backpropagation.** Forward pass computes loss; chain rule propagates gradients layer by layer; optimiser (SGD/Adam) updates weights. Vanishing/exploding gradients mitigated by ReLU/GELU, residual connections, normalisation, gradient clipping.

**Q2. When use DL for banking tabular data?** Rarely first choice. Use for: sequences of transactions (LSTM/Transformer), embeddings of high-cardinality categoricals (merchant, MCC), multimodal (text + tabular), very large data. Otherwise GBM + good features.

**Q3. Sequence models for fraud.** Treat each customer's last N transactions as a sequence; Transformer/LSTM learns behavioural patterns; combine embedding with tabular features → GBM or MLP head. Gains on account-takeover and card-testing patterns.

**Q4. Autoencoders for anomaly detection.** Train on normal behaviour; high reconstruction error = anomaly. Pitfalls: threshold choice, drift, hard to explain → pair with SHAP-like reconstruction-error attribution per feature.

**Q5. Overfitting remedies in NN.** Dropout, weight decay, early stopping, data augmentation, smaller model, batch norm, more data.

**Q6. CNN/RNN/Transformer — one-liners.** CNN: local patterns (cheque images, documents). RNN/LSTM: sequential, long-range limits. Transformer: self-attention, parallel, scales; foundation of LLMs.

---

## 2. NLP & Transformers

**Q7. Explain self-attention.** Attention(Q,K,V)=softmax(QKᵀ/√d_k)V. Each token attends to all others, weighting by relevance. Multi-head learns different relations. Positional encodings give order. Cost O(n²) in sequence length.

**Q8. BERT vs GPT.** BERT: encoder, bidirectional, masked LM → classification/NER/embeddings. GPT: decoder-only, autoregressive → generation. T5/BART: encoder-decoder.

**Q9. Tokenisation (BPE/WordPiece).** Subword units; balance vocabulary size and OOV handling. Token count drives cost/latency/context usage.

**Q10. Embeddings & similarity.** Dense vectors capturing semantics; cosine similarity/dot product; used in search, RAG, clustering, dedupe. Domain-adapted embeddings (finance) improve retrieval.

**Q11. Classic NLP in banking.** Complaint classification, sentiment of customer calls, NER on KYC/loan documents, adverse-media screening, news-based risk signals, regulatory-text mapping (obligations → controls).

---

## 3. Generative AI & LLMs (core for this role)

**Q12. What is an LLM, how trained?** Transformer pre-trained on huge corpora via next-token prediction → supervised fine-tuning (instruction tuning) → alignment (RLHF/DPO/RLAIF). Capabilities: summarise, extract, classify, reason, code, converse.

**Q13. Key inference parameters.** Temperature (randomness; 0–0.2 for deterministic bank tasks), top-p/top-k, max tokens, stop sequences, seed. Context window limits prompt+response.

**Q14. Prompt engineering techniques.** Clear role/task/format, few-shot examples, chain-of-thought (or "think then answer"), structured output (JSON schema), delimiters, negative constraints, self-consistency, prompt chaining, system prompts for policy. Always version and test prompts with an eval set.

**Q15. What is hallucination, why in banking is it dangerous, mitigation?**
Fluent but unsupported/incorrect output. Risk: wrong interest rate, fabricated policy, incorrect regulatory citation → customer harm, mis-selling, fines. Mitigation: RAG with citations, constrain to retrieved context ("answer only from sources, else say 'not found'"), low temperature, structured outputs, verification/guardrail layer, tool calls for facts (balances, rates from APIs, not from memory), human review, groundedness evals.

**Q16. Explain RAG architecture end-to-end.**
1. **Ingest**: documents (policies, product T&Cs, regulations) → clean → chunk (e.g., 300–800 tokens, 10–15% overlap, structure-aware by headings) → embed → vector DB (+ metadata: doc, version, date, access level).
2. **Retrieve**: query rewrite → hybrid search (BM25 + dense) → top-k → metadata filters (entitlements!) → **re-ranker** (cross-encoder).
3. **Generate**: prompt with context + instructions → LLM → answer with citations.
4. **Post**: groundedness check, PII redaction, policy filters, logging.
5. **Evaluate**: retrieval (recall@k, MRR, nDCG) + generation (faithfulness, answer relevance, correctness).
Banking example: internal policy assistant for relationship managers, answering from latest credit policy with page-level citations and access controls.

**Q17. RAG vs fine-tuning — when each?**
RAG: knowledge changes, need citations, access control, cheaper updates. Fine-tuning: style/format, domain language, task specialisation, latency/cost with small models. Often combine. Rule: *RAG for knowledge, fine-tune for behaviour.*

**Q18. Chunking strategy trade-offs.** Small chunks → precise but lose context; big chunks → more context but noise/cost. Use semantic/structure-aware chunking, parent-child retrieval, tables handled separately, add titles/metadata to chunks.

**Q19. How do you improve poor RAG answers?**
Diagnose: is it retrieval or generation? Check recall@k. Fixes: better chunking, hybrid search, query expansion/HyDE, re-ranking, metadata filtering, larger k + re-rank, domain embeddings, handle tables/PDF parsing (layout-aware), prompt tightening, "no answer" behaviour, golden test set.

**Q20. Vector databases & indexes.** FAISS, Pinecone, Weaviate, Milvus, pgvector, Azure AI Search, Vertex Vector Search, OpenSearch. ANN indexes: HNSW (fast, memory heavy), IVF-PQ (compressed). Trade-off: recall vs latency vs memory.

**Q21. Fine-tuning approaches: full vs PEFT (LoRA/QLoRA).** LoRA trains low-rank adapters (W+BA) — fraction of parameters, cheap, swappable per use-case. QLoRA adds 4-bit quantisation. Use when you have quality labelled examples (hundreds–thousands). Risks: catastrophic forgetting, data privacy, evaluation overhead.

**Q22. What are AI agents? Tool use?**
LLM + planning + tools (APIs, SQL, calculators, search) + memory, in a loop (ReAct: reason→act→observe). Banking examples: KYC analyst agent (pull docs, screen sanctions, draft summary), dispute-resolution agent, reconciliation agent. Guardrails: least-privilege tools, read-only by default, approval for actions, step/cost limits, full audit logs, deterministic workflows where possible (LangGraph state machines rather than free-form).

**Q23. Function calling / structured output.** Model returns JSON for a defined schema or tool call; your code executes it and returns results. Reduces hallucination for facts; validate with Pydantic; retry on schema failure.

**Q24. Frameworks.** LangChain, LangGraph, LlamaIndex, Semantic Kernel, Haystack, DSPy; Vertex AI / Azure OpenAI / AWS Bedrock as managed platforms; MCP (Model Context Protocol) for standardised tool/data connections.

**Q25. How do you evaluate LLM applications?**
- **Offline**: curated golden set (questions, reference answers, source docs) from SMEs; metrics: faithfulness/groundedness, answer relevance, context precision/recall (RAGAS), exact-match/F1 for extraction, rubric-based **LLM-as-judge** (calibrate vs human labels, watch bias/position effects), toxicity, PII leakage, jailbreak/red-team suite.
- **Online**: thumbs up/down, escalation rate, task completion, latency, cost/query, drift in query topics.
- **Regression tests** on every prompt/model/index change (CI for prompts).
- Human review sample for high-risk flows.

**Q26. Prompt injection & security.** Direct (user says "ignore instructions") and indirect (malicious text in retrieved documents/emails). Mitigate: separate instructions from data, treat retrieved content as untrusted, output filtering, least-privilege tools, allow-lists, no secrets in prompts, input/output classifiers, human approval for sensitive actions, red-teaming (OWASP Top 10 for LLM).

**Q27. Data privacy with LLMs in a bank.** PII detection/redaction/tokenisation before calling models, private endpoints (Azure OpenAI in-tenant, Bedrock VPC, Vertex), no training on customer data (contractual), data residency (GDPR, UK GDPR, regional rules), encryption, retention limits, role-based retrieval, audit trails. Option: on-prem/open-weights (Llama, Mistral) for highest sensitivity.

**Q28. Cost & latency optimisation.** Right-size model (small model for classification, big for reasoning), prompt caching, shorter prompts, semantic cache, batching, streaming, quantisation, distillation, routing/cascade (cheap model first, escalate on low confidence), limit retrieved context.

**Q29. Open vs closed models?** Closed APIs (Claude, GPT, Gemini): top quality, quick start, vendor risk/data-governance negotiation. Open-weights (Llama, Mistral, Qwen): control, on-prem, fine-tuning freedom, but you run infra and safety. Decide by data sensitivity, cost at scale, quality needs, regulatory stance.

**Q30. Multimodal / document AI.** Extract from scanned statements, cheques, IDs: OCR + layout-aware models or vision-capable LLMs → schema validation → confidence thresholds → human review queue. Cross-check numbers (totals reconcile) to catch silent errors.

**Q31. Synthetic data with GenAI.** Generate realistic but non-identifying data for testing/training (tabular GANs, CTGAN, diffusion, LLM-generated text). Validate utility (train-on-synthetic, test-on-real) and privacy (membership inference, nearest-neighbour distance). Useful for rare fraud patterns and sharing data across teams/regions.

**Q32. Limits of LLMs for numeric/analytical tasks.** Weak at exact arithmetic and long tables; use code-execution/SQL tools ("text-to-SQL"), then verify; constrain with schema; show query to analyst.

---

## 4. GenAI Banking Use Cases (design-style questions)

For each: **Problem → Architecture → Data → Risks → Metrics → Rollout**.

### UC1. Customer-service assistant (HSBC / Amex style)
- **Problem**: Reduce contact-centre handling time; consistent answers.
- **Design**: Intent routing → RAG over product/FAQ/policy → tool calls for account-specific data (authenticated, read-only) → response with citations; agent-assist mode first (suggest replies to human), then limited customer-facing automation for low-risk intents.
- **Risks**: hallucinated terms, PII leakage, mis-selling, vulnerable-customer handling.
- **Guardrails**: refusal on advice, compliance-approved templates, escalation triggers (complaint, fraud, distress), call/chat logging.
- **Metrics**: AHT, containment rate, CSAT, escalation, groundedness, complaint rate.

### UC2. Contact-centre summarisation & QA
Transcribe → LLM summarise, extract intent/outcome/next steps → auto-fill CRM; scan for compliance phrases and mis-selling → QA scoring for 100% of calls vs 2% manual sampling. Metric: after-call work time, QA coverage, false flag rate.

### UC3. Relationship-manager / analyst copilot
Summarise client portfolios, news, earnings; draft meeting notes; query internal research via RAG. Entitlement-aware retrieval is crucial (information barriers / Chinese walls).

### UC4. KYC / onboarding & adverse media
Extract entities from ID/company docs; summarise UBO structures; screen news for adverse media with LLM triage (classify relevance, severity, link to entity) → analyst reviews. Metric: analyst time per case, recall of true adverse hits (must remain ≥ legacy), false positives.

### UC5. AML alert investigation assistant
For each alert: gather account profile, 90-day transactions, counterparties, previous SARs → LLM drafts a narrative and highlights red flags (with links to evidence) → investigator decides and edits → SAR narrative generated. Human remains decision maker. Metrics: time per alert, quality score, regulator feedback.

### UC6. Credit memo generation (commercial lending)
Pull financial statements, bureau, covenants, sector news → structured extraction → LLM drafts memo sections with source citations → credit officer reviews. Risks: numeric errors → reconcile with extracted tables, show sources.

### UC7. Regulatory change management (PwC / Deloitte favourite)
Ingest new regulations (e.g., Basel IV, DORA, EU AI Act, Consumer Duty) → LLM summarises, maps clauses to internal policies/controls, flags gaps → compliance team validates. Chunk by article; legal citations required.

### UC8. Code & data copilots for engineers/analysts
Text-to-SQL on curated semantic layer, code generation/migration (COBOL → Java/Python), unit-test generation, documentation. Guardrails: no secrets, secure SDLC reviews, licence scanning.

### UC9. Fraud/scam communication
Detect APP/romance/investment scam patterns in payment context; LLM generates tailored warning messages (sent via rules, not free-form); NLP over customer chat to detect coaching by fraudsters.

### UC10. Financial document Q&A (annual reports, 10-K, prospectus)
Parse PDFs with layout + table extraction; hybrid retrieval; cite page numbers; compute ratios by tool (not LLM arithmetic).

### UC11. Marketing content generation
Personalised email/offer copy generated within pre-approved templates and brand/compliance constraints; automated compliance classifier; A/B test uplift.

### UC12. Dispute/chargeback automation
Classify dispute reason, retrieve transaction + merchant evidence, draft response letter, route exceptions. Metric: cycle time, win rate.

---

## 5. Responsible AI, Governance & Regulation

**Q33. Model Risk Management (SR 11-7 / PRA SS1/23).** Three lines of defence; model inventory and tiering; independent validation (conceptual soundness, data, performance, sensitivity, limitations); ongoing monitoring; change management; documentation. For GenAI: validate prompts/RAG/guardrails, third-party LLM risk, define "model change" (new model version, prompt change, index refresh), human oversight levels.

**Q34. EU AI Act risk tiers.** Unacceptable (social scoring), **High-risk** (credit scoring, creditworthiness assessment, insurance pricing — requires risk management, data governance, logging, transparency, human oversight, accuracy/robustness), limited risk (chatbots — disclose AI), minimal. GPAI obligations for foundation models.

**Q35. Fairness in credit models.** Protected attributes (gender, race, age, religion); avoid proxies (postcode). Metrics: demographic parity, equal opportunity (TPR parity), equalised odds, adverse impact ratio (80% rule), calibration within groups (can't satisfy all simultaneously). Mitigation: remove/limit proxies, reweighting, constrained optimisation, threshold adjustment (legal caveats), less-discriminatory-alternative search; monitor continuously. Document trade-offs.

**Q36. Explainability layers.** Intrinsic (scorecards, GAMs, monotone GBM) vs post-hoc (SHAP, counterfactuals "if income were £X higher you'd be approved", PDP/ICE). For LLMs: citations, retrieved sources, reasoning summaries, but not true causal explanation → rely on traceability and controls.

**Q37. GDPR Art. 22 / adverse-action.** Right not to be subject to solely automated decisions with significant effects; right to meaningful information about logic; human review. In US, ECOA/Reg B requires specific reasons for denial.

**Q38. Responsible GenAI checklist.** Use-case risk tiering → data/privacy review → model selection & vendor due diligence → guardrails (input/output filters) → evaluation incl. red-team → human-in-the-loop design → monitoring & incident response → user disclosure → audit logs → periodic revalidation.

**Q39. Differential privacy / federated learning.** DP adds calibrated noise to protect individuals; federated learning trains across silos without sharing raw data (cross-bank fraud, cross-jurisdiction). Trade-offs: utility loss, complexity.

---

## 6. MLOps & Engineering

**Q40. End-to-end ML lifecycle.** Problem framing → data (lineage, quality checks) → features (feature store) → experiment tracking (MLflow/W&B) → training pipeline → validation & governance sign-off → packaging (Docker) → deployment (batch/real-time/shadow/canary) → monitoring → retraining.

**Q41. Batch vs real-time vs streaming inference.** Batch: nightly propensity/collections lists. Real-time API: card authorisation (<100 ms), digital credit decision. Streaming: AML pattern detection, feature computation with Kafka/Flink/Spark Structured Streaming.

**Q42. Feature store.** Single definition of features for training and serving to avoid training-serving skew; point-in-time correctness (no leakage); online (Redis/DynamoDB) + offline (warehouse/lake) stores. Tools: Feast, Tecton, Vertex/SageMaker/Databricks feature stores.

**Q43. Deployment strategies.** Shadow (log only), canary (small %), blue-green, A/B, champion–challenger. Banking prefers shadow + champion–challenger with rollback.

**Q44. Monitoring checklist.** Data quality (nulls, ranges), data drift (PSI), prediction drift, performance with delayed labels, calibration, fairness metrics, latency/errors, business KPIs. **LLMOps adds**: token usage/cost, latency, groundedness/hallucination rate, refusal rate, guardrail triggers, retrieval hit-rate, prompt/model version in logs.

**Q45. CI/CD for ML.** Unit tests for data/feature code, model tests (performance thresholds, bias checks), reproducibility (seeds, pinned deps, data versioning with DVC/Delta), IaC (Terraform), approval gates tied to model-risk sign-off.

**Q46. Big-data stack.** Spark/PySpark (joins, partitioning, skew handling, broadcast joins), Delta/Iceberg, Airflow orchestration, Kafka, Snowflake/BigQuery/Databricks. Typical question: "How do you handle a skewed join?" → salting, broadcast small table, AQE.

**Q47. Cloud platforms.** AWS (SageMaker, Bedrock), Azure (Azure ML, Azure OpenAI, AI Search — common at PwC/Deloitte/HSBC), GCP (Vertex AI — HSBC & others). Know one deeply.

---

## 7. Business / Case-Study Questions (structured answers)

Use framework: **Clarify → Define objective & metric → Data → Approach → Evaluation → Deployment/risk → Impact**.

**Case 1. "Amex card fraud losses rose 15% in one quarter. What do you do?"**
Clarify (type: CNP/ATO/counterfeit? region? merchant? new vs existing cards?). Decompose losses = volume × rate × avg loss. Slice by channel, MCC, geography, device, time to find concentration. Hypotheses: new attack pattern, model drift, rule gap, new product launch. Short-term: tighten rules on the hot segment, step-up auth. Medium-term: retrain with recent data, new features (device, velocity), graph for rings. Measure with fraud bps and false-positive ratio; monitor customer friction.

**Case 2. "Design a credit-line increase programme."**
Eligibility (risk + policy) → PD/affordability → uplift model for incremental spend → optimisation maximising profit under loss constraint → randomised test cells → monitor delinquency and utilisation → fairness review.

**Case 3. "HSBC wants to cut AML false positives by 40% without missing suspicious activity."**
Baseline alert quality; label history (SAR outcomes); build alert-prioritisation model; backtest recall at reduced volume; below-the-line testing; regulator engagement; explainability; phased rollout (shadow → parallel → live for lowest-risk tier); governance and ongoing tuning.

**Case 4. "Build a GenAI assistant for relationship managers."** (Deloitte/PwC style)
Use-case value (time saved/RM), data sources and entitlements, RAG design, evaluation set with SMEs, guardrails, security, pilot with 50 users, KPIs (time saved, accuracy, adoption), change management, TCO.

**Case 5. "Predict customer attrition for premium cardholders."**
Define churn; features; model; uplift for retention offers; offer economics (cost vs CLV); A/B test; avoid cannibalising loyal customers.

**Case 6. "Model performance dropped after 6 months. What now?"**
Check data pipeline breaks → data drift (PSI) → label/definition changes → population shift (new acquisition channel, macro shock) → concept drift → recalibrate vs retrain → validate OOT → champion–challenger → governance approval.

**Case 7. "Estimate ROI of a GenAI initiative."**
Benefit = hours saved × volume × loaded cost × adoption − (build + run + governance + risk-mitigation costs); include quality uplift, risk reduction, ramp-up; sensitivity analysis; pilot to validate assumptions.

---

## 8. Behavioural (STAR) — prepare 6 stories
1. Model that went to production and moved a KPI (quantify).
2. Disagreement with stakeholder / pushing back on a request.
3. Failure/bug (data leakage found late) and what changed.
4. Explaining a complex model to non-technical audience.
5. Working with ambiguity / tight deadline.
6. Leading/mentoring or influencing without authority.

Tips: Situation (1 line) → Task → Action (most detail, "I") → Result (numbers) → Learning.

## 9. Questions to ask the interviewer
- How are models governed and how long from idea to production?
- What's the GenAI strategy — build vs buy, which platform?
- How does the team measure success of DS projects?
- What data quality/infra challenges exist?

## 10. 2-Week Study Plan
| Days | Focus |
|---|---|
| 1–2 | Stats + A/B testing + metrics |
| 3–4 | Credit risk (PD/LGD/EAD, scorecard, IFRS 9) |
| 5–6 | Fraud + AML systems, imbalance |
| 7 | SQL + Python drills (daily 5 problems) |
| 8–9 | LLM, RAG, agents, evaluation |
| 10 | Governance: SR 11-7, EU AI Act, fairness, explainability |
| 11 | MLOps/cloud |
| 12–13 | Mock cases, STAR stories, resume deep-dive |
| 14 | Company research (Amex: charge card/Membership Rewards/merchant network; HSBC: global trade/wealth/FinCrime; PwC/Deloitte: client industries, Responsible AI offerings) |

## 11. Resume Deep-Dive Rule
For every project on your CV be ready to answer: objective, data size/sources, target definition, features, model + why, metrics (with numbers), validation design, deployment, business impact, what you'd do differently.
