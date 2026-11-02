# Part 3: Reported Interview Questions + Coding/Technical Bank

> **Honesty note.** Live web search was unavailable in this environment, and Interview Query / AmbitionBox blocked automated access. Section A contains only what I could verify from GeeksforGeeks interview-experience pages (mostly campus hires, so lighter than experienced-hire loops). Section B–E are **commonly reported patterns from my own knowledge, NOT web-verified** — treat them as practice material, not a leak of actual questions. For real recent questions, check Glassdoor, AmbitionBox, Interview Query, LeetCode Discuss and Blind for your exact role and level.

## A. Verified reported questions (GeeksforGeeks, ~2023–2025)

### American Express — Data Science (IIT on-campus, updated Jan 2025)
- Online test (Unstop, MCQ): 24 numerical, 24 logical/data interpretation, 12 coding; easy–medium.
- Round 1 (tech + HR): self-intro; walk through every resume project and your unique contribution; **SQL on two tables and draw inferences**; guesstimate *annual sales of 1 kg packets of a snack brand*; "what do you know about Amex?"
- Round 2: two things that matter to you from your resume/life; guesstimate *weight of the International Space Station*; puzzle *10 boxes of balls, one box has 20 g balls, others 10 g — find it with fewest weighings*.
  - Answer: take 1 ball from box 1, 2 from box 2 … 10 from box 10, weigh together once. Excess over 550 g ÷ 10 = box number.

### Deloitte — Analyst (campus, 2023–2025)
- Behavioural: team difficulties, meaning of success, failure story, "why shouldn't we hire you?", masters plans, reporting to someone younger.
- Project: what changes/difficulties/improvements.
- Coding: minimum steps to climb a staircase (DP), array manipulation, string processing, first repeating character, character frequency > 2, series 5, 10, 35, 75, 195, SQL multi-table join, API definition, group discussion "AI replaces humans".

### HSBC / PwC
- No data-science-specific write-ups found on the pages I could read (HSBC listings were software-engineer trainee; PwC had one Data Engineer 2-YOE post I did not open). Use Section B–D patterns.

**Takeaway:** Amex tests SQL + guesstimates + puzzles + resume depth; Deloitte campus loops test fundamentals, DSA basics, behavioural and communication.

---

## B. Typical Amex-style technical themes (pattern, not verified)
- SQL: window functions, customer spend cohorts, top-N per group, retention, self joins.
- Stats: A/B test design, hypothesis tests, bias/variance, imbalanced classification.
- ML case: card attrition, credit line increase, fraud, propensity/response modelling, Basel/IFRS knowledge for risk roles.
- Guesstimates (market size, daily card swipes in a city) and puzzles.
- Business judgement: "which metric would you track for rewards programme?"

## C. Coding bank (DSA + data coding, with solutions)

**C1. First repeating / non-repeating character**
```python
from collections import Counter
def first_unique(s):
    c = Counter(s)
    return next((ch for ch in s if c[ch] == 1), None)

def first_repeating(s):
    seen = set()
    for ch in s:
        if ch in seen: return ch
        seen.add(ch)
```

**C2. Climbing stairs (min steps / ways)** — Fibonacci DP
```python
def ways(n):
    a, b = 1, 1
    for _ in range(n - 1): a, b = b, a + b
    return b
```
Min steps with allowed jumps {1,2,3}: `dp[i]=min(dp[i-j])+1`.

**C3. Two Sum** — hash map O(n)
```python
def two_sum(a, t):
    seen = {}
    for i, x in enumerate(a):
        if t - x in seen: return seen[t - x], i
        seen[x] = i
```

**C4. Valid parentheses** — stack.
**C5. Merge intervals** — sort by start, merge overlaps (used for session/overlapping-transaction windows).
**C6. Longest substring without repeating characters** — sliding window.
**C7. Detect cycle in linked list** — Floyd tortoise-hare.
**C8. Kadane's max subarray** — `cur=max(x,cur+x); best=max(best,cur)`.
**C9. Binary search & variants.**
**C10. Top-K frequent merchants** — `Counter(...).most_common(k)` or heap, O(n log k).
**C11. Moving average / rolling sum of transactions** — deque or prefix sums.
**C12. Implement k-means / logistic regression from scratch (numpy)**
```python
import numpy as np
def logreg(X, y, lr=0.1, it=2000, l2=0.0):
    w = np.zeros(X.shape[1]); b = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * (p - y).mean()
    return w, b
```
**C13. Dedupe records (entity resolution mini)** — normalise strings, block on key, compare with `difflib`/Levenshtein.
**C14. Pandas drills:** pivot/melt, merge with indicator, groupby-transform, rolling by time, `qcut` binning, `value_counts(normalize=True)`, handling duplicates, datetime parsing.

**C15. SQL advanced drills (be able to write from memory)**
- Running total: `SUM(amount) OVER (PARTITION BY acct ORDER BY ts)`.
- Gaps & islands (consecutive days of activity): `date - ROW_NUMBER()` grouping.
- Median per group: `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)`.
- Customers with transactions in 3 consecutive months.
- Users whose latest transaction is in a different country than previous (`LAG`).
- Duplicate transactions (same acct, amount, merchant within 2 minutes).
- Delete duplicates keeping latest: `ROW_NUMBER() ... WHERE rn>1`.
- Pivot spend by category per month with `CASE WHEN`.

## D. Guesstimate & puzzle bank (Amex/consulting style)
Framework: clarify → segment → assumptions with round numbers → compute → sanity-check.
1. Number of credit card transactions per day in India/UK/US.
2. Annual revenue of a premium card portfolio of 1M customers (annual fee + interchange ~2% × spend − rewards − losses).
3. How many ATMs are needed in a city of 5M?
4. Estimate fraud loss for a card base.
5. Puzzles: 8 balls/1 heavier (2 weighings), 3 switches/3 bulbs, 25 horses/5 tracks, two eggs/100 floors.

## E. Likely technical themes by firm (pattern, not verified)
- **HSBC:** FinCrime/AML ML, credit risk (IFRS 9, Basel), model explainability, GCP/Vertex or Azure, PySpark/SQL at scale, stakeholder communication, GenAI for compliance and ops with governance.
- **PwC:** Risk/regulatory analytics, model validation, forensic analytics (Benford's law, anomaly detection on journal entries), GenAI governance, client case, Azure stack.
- **Deloitte:** End-to-end project delivery, RAG POCs, MLOps, responsible AI, consulting case, behavioural ("tell me about leading a client deliverable").

**Extra topics likely in 2024–2026 GenAI rounds:** RAG evaluation (RAGAS), chunking choices, hybrid search + reranking, agents/tool use and MCP, prompt injection, guardrails, LLM-as-judge pitfalls, cost/latency optimisation, EU AI Act high-risk obligations, text-to-SQL, fine-tuning vs RAG, small language models/distillation, LLM observability.

**Forensic analytics extras (PwC/Deloitte):** Benford's law first-digit test, duplicate payment detection, round-number analysis, journal-entry anomaly scoring (weekend postings, unusual users), vendor-master vs employee-master matching.

## F. How to get *real* recent questions (do this yourself)
1. Glassdoor / AmbitionBox → filter by role and last 12 months.
2. Interview Query company guides (Amex, HSBC, Deloitte, PwC).
3. LeetCode Discuss → search "American Express", "HSBC".
4. LinkedIn: ask recruiters/alumni for the round structure.
5. Add each real question you find back into this file with your own answer.
