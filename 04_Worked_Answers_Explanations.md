# Part 4: Worked Answers + Explanations (companion to Part 3)

Every item: **Answer → Why it works → Complexity / pitfalls**. Python 3, pandas, standard SQL (Postgres-style).

---

## 1. Coding

### 1.1 First non-repeating / first repeating character
```python
from collections import Counter
def first_unique(s):
    c = Counter(s)                      # one pass to count
    return next((ch for ch in s if c[ch] == 1), None)   # second pass keeps original order

def first_repeating(s):
    seen = set()
    for ch in s:
        if ch in seen: return ch        # first char whose second occurrence appears earliest
        seen.add(ch)
    return None
```
**Explanation:** Counting first then scanning in original order preserves "first". A set gives O(1) membership. **O(n) time, O(k) space** (k = alphabet). Pitfall: "first repeating" (earliest second occurrence) differs from "first character that repeats anywhere" — clarify with the interviewer.

### 1.2 Character frequency, only those appearing more than twice
```python
def freq_gt2(s):
    return {ch: n for ch, n in Counter(s).items() if n > 2}
```
Dict comprehension filters counts. O(n).

### 1.3 Climbing stairs
**Ways to reach step n using 1 or 2 steps:** `ways(n) = ways(n-1) + ways(n-2)` — the last move was either a 1-step or a 2-step.
```python
def ways(n):
    a, b = 1, 1                 # ways(0), ways(1)
    for _ in range(n - 1):
        a, b = b, a + b
    return b
```
O(n) time, O(1) space (Fibonacci).

**Minimum steps to reach top with allowed jumps {1,2,3}:** greedy takes the largest jump: `ceil(n/3)`. With arbitrary jump sizes or blocked steps use DP:
```python
def min_steps(n, jumps=(1,2,3), blocked=()):
    INF = float("inf"); dp = [INF]*(n+1); dp[0] = 0
    for i in range(1, n+1):
        if i in blocked: continue
        for j in jumps:
            if i-j >= 0: dp[i] = min(dp[i], dp[i-j] + 1)
    return dp[n] if dp[n] != INF else -1
```
**Why DP:** the best way to step i depends only on best ways to earlier steps (optimal substructure). O(n·|jumps|).

### 1.4 Two Sum
```python
def two_sum(a, target):
    seen = {}                           # value -> index
    for i, x in enumerate(a):
        if target - x in seen:
            return seen[target - x], i
        seen[x] = i
```
**Why:** for each x we need its complement; storing earlier values in a hash map avoids O(n²) brute force. O(n). Check complement *before* inserting so one element isn't used twice.

### 1.5 Valid parentheses
```python
def valid(s):
    pair = {')':'(', ']':'[', '}':'{'}
    st = []
    for ch in s:
        if ch in pair:
            if not st or st.pop() != pair[ch]: return False
        else:
            st.append(ch)
    return not st
```
Last-opened must be first-closed → stack (LIFO). O(n).

### 1.6 Merge intervals
```python
def merge(iv):
    iv.sort(key=lambda x: x[0])
    out = [iv[0]]
    for s, e in iv[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out
```
Sorting by start means any overlap can only be with the last merged interval. O(n log n). Banking use: merge overlapping delinquency periods or session windows.

### 1.7 Longest substring without repeating characters (sliding window)
```python
def longest(s):
    last, left, best = {}, 0, 0
    for r, ch in enumerate(s):
        if ch in last and last[ch] >= left:
            left = last[ch] + 1         # jump left pointer past previous occurrence
        last[ch] = r
        best = max(best, r - left + 1)
    return best
```
Each index enters/leaves the window once → O(n).

### 1.8 Cycle in linked list (Floyd)
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast: return True
    return False
```
If a loop exists, the fast pointer gains one node per step on the slow pointer and must meet it. O(n), O(1) space.

### 1.9 Kadane's maximum subarray
```python
def max_sub(a):
    cur = best = a[0]
    for x in a[1:]:
        cur = max(x, cur + x)           # extend the run or restart at x
        best = max(best, cur)
    return best
```
A negative running sum can only hurt, so restart. O(n). Use: best/worst consecutive-day net cash flow.

### 1.10 Binary search
```python
def bsearch(a, t):
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if a[mid] == t: return mid
        lo, hi = (mid + 1, hi) if a[mid] < t else (lo, mid - 1)
    return -1
```
Halves the search space → O(log n). Requires sorted input.

### 1.11 Top-K frequent merchants
```python
import heapq
from collections import Counter
def topk(merchants, k):
    c = Counter(merchants)
    return heapq.nlargest(k, c, key=c.get)        # O(n + m log k)
```
Heap of size k beats sorting all m distinct keys when k ≪ m.

### 1.12 Rolling 7-day transaction sum (sliding window / deque)
```python
from collections import deque
def rolling(txns, days=7):                 # txns: [(day, amount)] sorted by day
    q, total, out = deque(), 0, []
    for d, a in txns:
        q.append((d, a)); total += a
        while q[0][0] <= d - days:
            total -= q.popleft()[1]
        out.append(total)
    return out
```
Each element enters and exits once → O(n). This is exactly the "velocity feature" in fraud models.

### 1.13 Logistic regression from scratch — explanation
```python
import numpy as np
def logreg(X, y, lr=0.1, it=2000, l2=0.0):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ w + b)))              # sigmoid
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)      # gradient of log-loss + L2
        b -= lr * (p - y).mean()
    return w, b
```
**Why it works:** the log-loss gradient w.r.t. weights is `Xᵀ(p−y)/n`. Standardise features first or gradient descent converges slowly. Be ready to say: Newton/IRLS is what statsmodels uses; L2 term is ridge.

### 1.14 K-means from scratch
```python
def kmeans(X, k, it=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(it):
        lab = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)     # assign to nearest centroid
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(C, newC): break
        C = newC
    return lab, C
```
Alternates assignment and mean update, minimising within-cluster sum of squares (monotone decrease, local optimum). Use k-means++ init and multiple restarts in practice; standardise features.

### 1.15 Dedupe / fuzzy match names (entity-resolution mini)
```python
import difflib, re
def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
def similar(a, b, th=0.88):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= th
```
**Explanation:** normalise → block (e.g., same first letter/DOB) so you don't compare all n² pairs → score with edit-distance/Jaro-Winkler → threshold tuned on labelled pairs (precision vs recall). Sanctions screening favours recall.

### 1.16 Number series 5, 10, 35, 75, 195
This puzzle is odd; one plausible reading: differences are 5, 25, 40, 120 — no clean rule. Interviewer-dependent; the right move is to state your hypothesis aloud (differences, ratios, alternating ops), test it, and say what you'd ask the interviewer. Don't claim a certain answer you can't justify.

### 1.17 Pandas drills (with answers)
```python
# top-3 categories per customer by spend
(df.groupby(["cust","cat"], as_index=False)["amt"].sum()
   .sort_values(["cust","amt"], ascending=[True,False]).groupby("cust").head(3))

# share of each customer's spend per category (transform keeps row alignment)
df["share"] = df["amt"] / df.groupby("cust")["amt"].transform("sum")

# decile bins of a score, with bad rate per bin
df["dec"] = pd.qcut(df["score"], 10, labels=False, duplicates="drop")
df.groupby("dec")["bad"].agg(["count","mean"])

# time-based rolling without leakage (excludes current row)
df = df.sort_values(["cust","ts"])
df["amt_7d"] = (df.set_index("ts").groupby("cust")["amt"]
                  .rolling("7D", closed="left").sum().reset_index(level=0, drop=True).values)

# duplicates
df.drop_duplicates(subset=["acct","amt","merchant","ts"], keep="first")
```

---

## 2. SQL (answers + explanation)

Tables: `transactions(txn_id, acct_id, cust_id, txn_ts, amount, merchant_id, country)`.

**2.1 Running total**
```sql
SELECT acct_id, txn_ts, amount,
       SUM(amount) OVER (PARTITION BY acct_id ORDER BY txn_ts
                         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM transactions;
```
Window function keeps every row (unlike GROUP BY); the frame sets which rows are summed.

**2.2 Top-N per group (top 3 transactions per customer)**
```sql
SELECT * FROM (
  SELECT t.*, ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY amount DESC) rn
  FROM transactions t) x
WHERE rn <= 3;
```
`ROW_NUMBER` gives exactly 3; use `DENSE_RANK` if ties should all be included.

**2.3 Gaps and islands: consecutive-day activity streaks**
```sql
WITH d AS (SELECT DISTINCT cust_id, txn_ts::date AS d FROM transactions),
g AS (SELECT cust_id, d,
             d - (ROW_NUMBER() OVER (PARTITION BY cust_id ORDER BY d))::int AS grp
      FROM d)
SELECT cust_id, MIN(d) start_d, MAX(d) end_d, COUNT(*) streak_days
FROM g GROUP BY cust_id, grp;
```
**Why:** within a run of consecutive dates, `date − row_number` is constant, so it labels each streak.

**2.4 Median per group**
```sql
SELECT cust_id, PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS median_amt
FROM transactions GROUP BY cust_id;
```

**2.5 Customers active in 3 consecutive months**
```sql
WITH m AS (SELECT DISTINCT cust_id, DATE_TRUNC('month', txn_ts) mth FROM transactions),
x AS (SELECT cust_id, mth,
             LAG(mth, 2) OVER (PARTITION BY cust_id ORDER BY mth) AS m2
      FROM m)
SELECT DISTINCT cust_id FROM x
WHERE m2 = mth - INTERVAL '2 month';
```
If the month two rows back is exactly two months earlier, there is no gap.

**2.6 Country change vs previous transaction (travel/ATO signal)**
```sql
SELECT * FROM (
  SELECT t.*, LAG(country) OVER (PARTITION BY cust_id ORDER BY txn_ts) prev_c,
         LAG(txn_ts)  OVER (PARTITION BY cust_id ORDER BY txn_ts) prev_ts
  FROM transactions t) x
WHERE country <> prev_c AND txn_ts - prev_ts < INTERVAL '2 hour';
```
Two countries within 2 hours is physically implausible → "impossible travel" rule.

**2.7 Duplicate transactions within 2 minutes**
```sql
SELECT a.txn_id, b.txn_id
FROM transactions a JOIN transactions b
  ON a.acct_id = b.acct_id AND a.amount = b.amount AND a.merchant_id = b.merchant_id
 AND a.txn_id < b.txn_id
 AND b.txn_ts - a.txn_ts BETWEEN INTERVAL '0' AND INTERVAL '2 minute';
```
`a.txn_id < b.txn_id` avoids pairing a row with itself and counting pairs twice.

**2.8 Delete duplicates, keep latest**
```sql
DELETE FROM transactions WHERE txn_id IN (
  SELECT txn_id FROM (
    SELECT txn_id, ROW_NUMBER() OVER (PARTITION BY acct_id, amount, merchant_id, txn_ts
                                      ORDER BY txn_id DESC) rn
    FROM transactions) x WHERE rn > 1);
```

**2.9 Pivot spend by category per month**
```sql
SELECT DATE_TRUNC('month', txn_ts) mth,
       SUM(CASE WHEN cat='TRAVEL' THEN amount ELSE 0 END) travel,
       SUM(CASE WHEN cat='DINING' THEN amount ELSE 0 END) dining
FROM transactions GROUP BY 1;
```

**2.10 Amex-style "analyse two tables and infer"** (customers, spend)
Approach to say aloud: check keys and grain (one row per what?), join type (LEFT to keep customers without spend), aggregate before joining to avoid fan-out duplication, then report: spend per segment, % customers inactive, top/bottom segments, outliers, and caveats (sample bias, missing data). The *reasoning and insight* are scored, not just the query.

**2.11 Common SQL traps:** `NULL` comparisons (`= NULL` is never true, use `IS NULL`); `COUNT(*)` vs `COUNT(col)`; joining before aggregating (double counting); `WHERE` filters rows before grouping, `HAVING` after; integer division; time-zone handling.

---

## 3. Guesstimates — worked answers

**Method:** clarify scope → split into drivers → state round-number assumptions → compute → sanity check → name what would change it.

### 3.1 Annual sales of 1 kg packets of a snack brand (India)
- Population ≈ 1.4B → ~300M households (≈4.7 per household).
- Assume 30% buy this packaged namkeen category regularly → 90M households.
- Assume each buys ~1 packet/month of 1 kg (mix with smaller packs, so say 0.5 per month 1 kg-equivalent) → 6 packets/year.
- 90M × 6 ≈ **~540M packets/year** for the whole category; if the brand has 10% share → **~50M**.
- Sanity: 50M kg = 50,000 tonnes; plausible for a large regional brand. State that 1 kg is a minority pack size vs 100–200 g packs, so reduce further depending on the brand.
*What matters:* clean structure and sensible, adjustable assumptions, not the exact number.

### 3.2 Weight of the International Space Station
- Reasoning: modules are roughly cylinders; ISS is about 109 m × 73 m truss, pressurised volume ~900 m³.
- Mass can be built bottom-up from modules: ~15 major modules × ~15–20 tonnes plus solar arrays/truss ≈ 300–450 tonnes.
- Known figure: ~420 tonnes (~420,000 kg). Mention you'd cross-check with launch capacity: ~40 launches of 10 t each ≈ 400 t. ✔

### 3.3 Credit card transactions per day (e.g., UK)
- 67M people → ~50M adults → ~60% have a credit card = 30M cards.
- Average ~2 transactions per active card per day, 70% active → 30M × 0.7 × 2 ≈ **~40M/day**.
- Cross-check against annual stats (~10–12B credit card transactions/year ÷ 365 ≈ 30M). Same order of magnitude ✔

### 3.4 Annual revenue of a premium card portfolio, 1M customers
- Annual fee: say £400 × 1M = £400M.
- Spend: £25k/customer/year → £25B × interchange/discount revenue 2.0% = £500M.
- Interest/fees: 10% revolve × £3k balance × 20% APR ≈ £60M (low for charge-card models).
- Costs: rewards ~1.2% of spend = £300M, credit losses ~0.5% of spend = £125M, marketing/servicing/fraud.
- Gross revenue ≈ £960M; net contribution after rewards and losses ≈ £500M. Say what drives sensitivity: spend per card and reward rate.

### 3.5 ATMs needed for a 5M-person city
- ~1 ATM per 1,500–2,000 people in developed markets → 5M/1,750 ≈ ~2,900.
- Alternative demand-side: 5M × 60% use ATM × 2 withdrawals/month = 6M/month ÷ 30 = 200k/day; each ATM handles ~150/day at 30% utilisation → ~1,300 for steady demand; add peaks, coverage and redundancy → between 1,500 and 3,000.

---

## 4. Puzzles — answers

**4.1 10 boxes, one has 20 g balls, others 10 g, one weighing.**
Take 1 ball from box 1, 2 from box 2 … 10 from box 10 (55 balls). If all were 10 g the total would be 550 g. Each ball from the heavy box adds 10 g extra, so (total − 550) ÷ 10 = box number. Example: 570 g → 20 extra → 2 heavy balls → box 2. Fewest weighings = 1.

**4.2 8 balls, one heavier, balance scale.** 2 weighings: split 3-3-2. Weigh 3 vs 3: if unequal, heavier group has it; if equal, it's in the 2 left. Next weigh 1 vs 1 (from heavier 3, leaving one aside; or the 2). Works because each weighing splits possibilities into 3 outcomes (3² = 9 ≥ 8).

**4.3 3 switches, 3 bulbs, one visit.** Turn switch 1 on for ~5 min, turn it off, turn switch 2 on, go in: lit bulb = switch 2, warm-but-off = switch 1, cold-and-off = switch 3.

**4.4 25 horses, 5 per race, find top 3.** 7 races. 5 heats; race the 5 winners; the winner of that is #1; the top 3 can only come from: 2nd/3rd of the fastest group, 1st/2nd of the second-fastest group, and the 1st of the third-fastest group. Race those 5 (excluding #1) → top 2 of it are #2 and #3.

**4.5 Two eggs, 100 floors.** Drop at 14, then +13, +12 … (14+13+…+1 = 105 ≥ 100). Worst case 14 drops. Egg 1 steps in decreasing increments so total worst case stays constant.

---

## 5. Behavioural / short-answer questions (model answers)

**"Why should we not hire you?"** Be honest but safe: "I'm less experienced with [X, e.g., large-scale streaming systems]. To address this I've [built a project / taken a course] and I ramp up fast — here's an example." Shows self-awareness + action.

**"What does success mean to you?"** Impact + learning: "Shipping something that measurably helps the business or customer, learning from the process, and helping the team succeed."

**"Describe a failure."** STAR: real failure → what you did to fix → what changed afterward (process/checklist). Example: a leakage bug caught late; now I add time-based validation and a leakage checklist.

**"What do you know about Amex?"** Closed-loop network (issuer + network), premium cards (Platinum, Gold), Membership Rewards, merchant services, strong risk/fraud analytics, charge-card heritage, customer service reputation. Mention recent news if you can find it.

**"Group discussion: AI replaces humans."** Structure: define → jobs displaced vs created → augmentation in banking (fraud analysts with ML triage; relationship managers with copilots) → regulation and human oversight → your balanced position with an example.

**"Where do you see yourself in 5 years?"** Technical depth in applied ML/GenAI for financial services, then leading a small team; tie it to the firm's growth areas.

---

## 6. Final tips
- Think aloud; state assumptions; ask clarifying questions before solving.
- For coding: brute force → optimise → test with edge cases (empty, one element, duplicates, negatives).
- For SQL: state the grain and the join keys, then write.
- For guesstimates: structure beats the number.
- If you don't know: reason from first principles and say what you'd verify.
