# Data Scientist Interview Prep — Part 1: Stats, ML, Python/SQL, Banking Use Cases
Targets: American Express, HSBC, PwC, Deloitte (and similar banks / Big-4 / consulting analytics roles)

> How to use: read the answer, then close the file and say it aloud in 60–90 seconds. Always finish with a **banking example** and a **trade-off**. Interviewers reward structure: *Definition → Why it matters → Example → Pitfall*.

## 0. What each firm typically tests

| Firm | Flavour | Focus |
|---|---|---|
| **Amex** | Product + risk analytics | Fraud, credit risk, marketing/propensity, SQL + Python, A/B tests, case on card spend/attrition, model governance |
| **HSBC** | Global bank, heavy regulation | Credit risk (PD/LGD/EAD), AML/KYC/transaction monitoring, explainability, model risk, GenAI for ops/compliance, cloud (GCP) |
| **PwC** | Consulting + risk/regulatory | Business framing, stakeholder comms, model validation, AML/forensics, GenAI governance, case study |
| **Deloitte** | Consulting + implementation | End-to-end delivery, MLOps, client case, GenAI/RAG POCs, responsible AI, storytelling |

Typical loop: (1) resume deep-dive, (2) stats/ML theory, (3) SQL/Python live coding, (4) business case, (5) behavioural, (6) GenAI discussion.

---

## 1. Statistics & Probability

**Q1. Explain p-value, and a common misinterpretation.**
The p-value is P(data at least this extreme | H0 true). It is *not* the probability H0 is true. Banking: testing whether a new fraud rule lowers fraud rate; p=0.03 means that if the rule had no effect, we'd see a difference this large 3% of the time.

**Q2. Type I vs Type II error — which matters more in fraud detection / AML?**
Type I = false positive (legit customer blocked / false alert); Type II = false negative (fraud missed). In fraud, FN costs the fraud loss; FP costs customer friction and investigator time. In AML, regulators penalise misses heavily, so recall is prioritised, but alert volume must stay workable. Answer with cost matrix: expected cost = FN×loss + FP×review cost.

**Q3. Central Limit Theorem and why it matters?**
Sample means tend to normal as n grows regardless of population shape (finite variance). Enables confidence intervals and t/z-tests on average transaction value even though spend is heavily right-skewed.

**Q4. Confidence interval — interpretation?**
If we repeated the sampling many times, 95% of such intervals would contain the true parameter. Not "95% probability the parameter lies in this one."

**Q5. How do you design an A/B test for a new credit-card offer?**
1. Define metric (activation rate at 90 days; guardrails: delinquency, cost). 2. Unit of randomisation (customer, not transaction). 3. Power analysis: n = 2(Zα/2+Zβ)²σ²/δ². 4. Run full cycles (avoid novelty/day-of-week effects). 5. Check SRM (sample-ratio mismatch). 6. Test, correct for multiple comparisons (Bonferroni/BH). 7. Decide with CI and business impact. Pitfalls: peeking, interference, long-term risk metrics (defaults) need long horizons → use proxy + holdout.

**Q6. What if the A/B test isn't possible (e.g., credit-limit policy)?**
Causal inference: diff-in-diff, propensity score matching/IPW, regression discontinuity (e.g., score cut-offs), synthetic control, uplift modelling.

**Q7. Bayes' theorem — fraud base-rate example.**
Fraud rate 0.1%, model sensitivity 95%, false-positive rate 2%. P(fraud | flagged) = (0.95×0.001)/(0.95×0.001+0.02×0.999) ≈ **4.5%**. Lesson: with rare events, precision is low even for a "good" model → why thresholds and tiered review matter.

**Q8. Correlation vs causation; Simpson's paradox?**
Aggregate trend reverses within segments. Example: Overall approval rate for Group A looks lower, but within each income band A is higher, because A applies in riskier bands. Always segment and look for confounders.

**Q9. Handling skewed distributions (income, balances)?**
Log / Box-Cox / Yeo-Johnson, winsorising, robust stats (median, MAD), quantile binning, tree models are insensitive.

**Q10. Multicollinearity — detection and effect?**
VIF > 5–10, correlation matrix, condition number. Inflates coefficient variance (unstable signs), harms interpretability in logistic scorecards, not predictions. Fix: drop/combine, PCA, ridge, WOE-clustering (variable clustering).

**Q11. Bias–variance trade-off.**
Error = bias² + variance + noise. Complex model (deep tree) → low bias, high variance. Regularisation, bagging reduce variance; boosting reduces bias. Regulated credit models often accept slightly higher bias for stability and explainability.

**Q12. Parametric vs non-parametric tests; when t-test vs Mann-Whitney?**
Use Mann-Whitney when non-normal, small n or ordinal. For large n, t-test is robust by CLT.

**Q13. Chi-square test use?**
Association between categorical variables, e.g., is default status associated with employment type? Also goodness-of-fit. Used in feature selection and in PSI-type comparisons.

**Q14. Explain survival analysis & its banking use.**
Models time-to-event with censoring. Cox PH: h(t|x)=h0(t)exp(βx). Uses: time to default (lifetime PD for IFRS 9), time to churn, time to prepayment. Handles customers who haven't yet defaulted (censored) – something plain logistic regression ignores.

**Q15. Sampling for rare events (fraud)?**
Stratified/undersampling the majority with weights, SMOTE (careful: only on training fold), class weights, focal loss, anomaly detection. Evaluate on the *untouched* natural distribution with PR-AUC.

---

## 2. Machine Learning Fundamentals

**Q16. Logistic regression — why still the king in banking?**
Interpretable coefficients (log-odds), monotonic with WOE, stable, easy to document for regulators, easy scorecard conversion (points-to-double-odds), well-calibrated. Trade-off: misses non-linear interactions.

**Q17. Random forest vs gradient boosting?**
RF: bagging, parallel, reduces variance, robust, less tuning. GBM (XGBoost/LightGBM/CatBoost): sequential, fits residuals, reduces bias, usually best tabular accuracy, needs tuning (learning rate, depth, n_estimators, regularisation). CatBoost handles categoricals natively; LightGBM is fastest on large data.

**Q18. Why do tree-based models dominate tabular data vs deep learning?**
Handle mixed types, missing values, non-linearities, interactions with little preprocessing, robust to scale, less data hungry. DL wins on unstructured (text, images, sequences).

**Q19. Regularisation: L1 vs L2.**
L1 (Lasso) → sparsity/feature selection; L2 (Ridge) → shrinks, handles collinearity; Elastic Net mixes. In scorecards, L1 helps pick from hundreds of bureau attributes.

**Q20. Evaluation metrics for imbalanced classification.**
Avoid accuracy. Use Precision, Recall, F1/Fβ, **PR-AUC**, ROC-AUC, KS, Gini (=2·AUC−1), lift/gains at top decile, expected cost. Choose threshold by business cost, not 0.5.

**Q21. ROC-AUC vs PR-AUC?**
ROC-AUC insensitive to class imbalance (can look great at 0.99 even when precision is poor). PR-AUC reflects performance on the positive class → preferred for fraud.

**Q22. What are KS, Gini and when do banks use them?**
KS = max separation between cumulative good and bad distributions (typical credit models 30–50). Gini = 2AUC−1 (~40–70% for application scorecards). Used for ranking power in validation and monitoring.

**Q23. Calibration — why does it matter?**
Predicted probabilities must match observed frequencies (a "5% PD" bucket should default ~5%). Needed for pricing, provisioning (IFRS 9/CECL), capital. Check reliability curve, Brier score, Hosmer–Lemeshow; fix via Platt scaling / isotonic regression.

**Q24. Cross-validation approaches for banking data.**
Stratified K-fold for iid; **time-based (out-of-time) validation** for credit/fraud to avoid leakage and reflect drift; group K-fold by customer so the same customer isn't in train and test.

**Q25. What is data leakage? Give a banking example.**
Using information not available at prediction time. e.g., using "days past due next month" or "collections contact flag" as a feature for default; scaling/imputing on the full dataset before split; target encoding without folds; including post-approval variables in an application score.

**Q26. How to handle missing values?**
Understand mechanism (MCAR/MAR/MNAR). Options: median/mode, model-based (KNN, iterative), missing-indicator flags, separate WOE bin for missing (scorecards). In credit, "missing" is often informative (thin-file customers).

**Q27. Feature engineering ideas for transaction data (cards).**
Recency–Frequency–Monetary (RFM), rolling windows (1h/24h/7d/30d counts and sums), velocity, ratio to customer average, merchant category entropy, geo-distance from last txn, time-since-last-txn, device/IP novelty, share of night-time spend, graph features (shared device/merchant), embeddings of merchants.

**Q28. Dimensionality reduction methods?**
PCA (linear, variance), t-SNE/UMAP (visualisation), autoencoders; feature selection via IV, mutual information, permutation importance, Boruta, L1.

**Q29. Hyperparameter tuning?**
Grid → Random → Bayesian (Optuna, Hyperopt). Use nested CV or a separate OOT set to avoid optimistic bias. Early stopping.

**Q30. Handle concept drift / data drift?**
Monitor PSI/CSI (PSI <0.1 stable, 0.1–0.25 watch, >0.25 retrain/investigate), KS test, performance on lagged labels, rolling windows, scheduled retraining, champion–challenger.

**Q31. Clustering for customer segmentation.**
K-means (elbow/silhouette), GMM, hierarchical, DBSCAN. Standardise first; for mixed data, K-prototypes or Gower. Banking: segmenting card holders by spend behaviour for tailored offers; always validate segments for business actionability and stability.

**Q32. Anomaly detection techniques?**
Isolation Forest, One-Class SVM, LOF, autoencoder reconstruction error, statistical (z-score, MAD), graph anomalies. Use for unlabeled AML/fraud patterns.

**Q33. Explain SHAP vs LIME.**
SHAP: game-theoretic additive attribution, consistent, global + local, TreeSHAP fast and exact for trees. LIME: local linear surrogate, unstable. Banking use: adverse-action reason codes (top-N negative contributions), validation, bias checks. Caveats: correlated features, not causal.

**Q34. Recommender approaches for card offers/merchant rewards.**
Collaborative filtering/matrix factorisation, content-based, two-tower networks, contextual bandits for exploration. Cold start → use popularity/segment priors.

**Q35. Time-series forecasting for call-centre volume / ATM cash demand.**
Baselines (seasonal naive), ARIMA/SARIMA, ETS, Prophet, gradient boosting with lag features, LSTM/Temporal Fusion Transformer. Validate with rolling-origin CV; metrics MAPE/WAPE/sMAPE; handle holidays and payday effects.

---

## 3. Banking Use Cases (deep-dive Q&A)

### 3.1 Credit Risk

**Q36. Explain PD, LGD, EAD, EL.**
EL = PD × LGD × EAD. PD: probability of default within 12 months (or lifetime). LGD: share of exposure lost after recoveries. EAD: exposure at default (incl. undrawn limit × CCF). Used for IFRS 9 provisioning, Basel capital (IRB), pricing.

**Q37. Walk through building an application scorecard.**
1. Define default (90 DPD within 12 months), observation and performance windows. 2. Reject inference (parcelling/fuzzy augmentation) because only approved loans have outcomes. 3. Data prep & split (out-of-time). 4. Binning with WOE/IV (IV: <0.02 useless, 0.1–0.3 medium, >0.5 suspicious). 5. Variable selection, check correlation. 6. Logistic regression. 7. Scale to points (e.g., 600 points = 50:1 odds, PDO 20). 8. Validate: Gini/KS, calibration, stability (PSI). 9. Cut-off strategy and policy overlay. 10. Documentation and monitoring.

**Q38. WOE and IV formulas.**
WOE_i = ln(%Goods_i / %Bads_i); IV = Σ (%Goods_i − %Bads_i) × WOE_i. WOE gives monotonic, linearised relationship with log-odds, handles outliers/missing.

**Q39. IFRS 9 stages and ML?**
Stage 1: 12-month ECL; Stage 2: lifetime ECL after significant increase in credit risk (SICR); Stage 3: credit-impaired. ML can improve SICR detection and lifetime PD curves (survival models), but must incorporate forward-looking macro scenarios (GDP, unemployment) and remain explainable and auditable. CECL (US) uses lifetime from day one.

**Q40. Why might a more accurate XGBoost not be deployed for credit decisions?**
Explainability/adverse-action requirements (ECOA/FCRA in US, GDPR Art. 22 and EBA guidelines in EU/UK), model-risk validation burden (SR 11-7), stability, monotonicity constraints, fairness. Mitigations: monotonic constraints, SHAP reason codes, GAMs/EBMs, or XGBoost as challenger/feature discovery for the scorecard.

**Q41. Reject inference — what and why?**
Applicants declined have no performance labels → sample selection bias. Methods: augmentation, parcelling, extrapolation, bivariate approach. Not perfect; best data is from controlled approvals in a test cell.

**Q42. Basel II/III IRB vs Standardised approach?**
Standardised uses regulator risk weights/external ratings. IRB lets banks use internal PD (and LGD/EAD in advanced) → needs validated models, downturn LGD, conservatism margins.

**Q43. Model for credit limit increase at Amex?**
Target: incremental spend and revenue vs incremental risk. Use uplift modelling + risk model; optimisation under constraints (loss budget); test with randomised holdout; monitor utilisation, delinquency, attrition.

**Q44. Collections strategy modelling.**
Roll-rate/transition matrices, propensity-to-pay/cure models, contact-channel optimisation with bandits, next-best-action; measure vs control; fairness and conduct (vulnerable customers).

**Q45. Stress testing.**
Link macro variables (unemployment, HPI, rates) to PD/LGD via regression/VAR or satellite models, run baseline/adverse/severely adverse scenarios, estimate losses and capital impact (CCAR/ICAAP/BoE ACS).

### 3.2 Fraud Detection

**Q46. Design a real-time card fraud detection system.**
- **Data**: auth stream (amount, merchant, MCC, device, geo, channel), customer history, feature store (online/offline consistency).
- **Features**: velocity, behaviour profile, merchant risk, graph links.
- **Models**: GBM for tabular + sequence model (LSTM/Transformer) on recent txns + rules for known patterns.
- **Serving**: <50–100 ms latency, streaming (Kafka/Flink), online feature store (Redis), fallbacks.
- **Decision**: approve / step-up (OTP) / decline / refer; thresholds from cost matrix.
- **Feedback**: chargebacks arrive weeks later → label delay; use analyst dispositions as faster labels.
- **Monitoring**: drift, alert rates, precision by segment, adversarial adaptation, champion–challenger.
- **Trade-offs**: customer friction vs loss; explainability to investigators.

**Q47. Label delay and selection bias in fraud?**
Chargebacks take 30–120 days; declined transactions have no outcomes. Use mature windows for training, treat recent as unlabeled/weak, use semi-supervised learning, add random approval exploration where policy allows.

**Q48. How to evaluate fraud models business-wise?**
Fraud basis-points saved, value detection rate (VDR), false-positive ratio (good:fraud flagged), investigator hit rate, customer-declines per 1000, ROI. Compare at fixed alert capacity.

**Q49. Graph analytics in fraud/AML?**
Build entity graph (accounts, devices, addresses, merchants). Use community detection, PageRank, connected components, GNN (GraphSAGE) to detect rings, mule networks, synthetic identities. Neo4j/TigerGraph/GraphFrames.

**Q50. Types of card fraud?**
Card-not-present, account takeover, application/synthetic ID, first-party/friendly fraud, merchant collusion, card testing (small-amount bursts), skimming/counterfeit, authorised push payment (APP) scams.

### 3.3 AML / KYC / Financial Crime (very relevant for HSBC, PwC, Deloitte)

**Q51. Transaction monitoring today vs ML approach.**
Rule-based scenarios (structuring, rapid movement, high-risk geographies) generate 90–99% false positives. ML triage: score alerts to prioritise, cut false positives while keeping SAR recall; supplement with anomaly detection and network analysis. Regulators expect explainability, tuning documentation, below-the-line testing.

**Q52. How do you evaluate an AML model with very few labelled SARs?**
Use SAR filings + investigator outcomes as noisy labels; evaluate recall at fixed alert volume, "below-the-line" sampling, backtesting vs historical scenarios, expert review, typologies coverage. Consider semi-supervised/PU learning.

**Q53. Entity resolution for KYC?**
Match customers across systems/watchlists: normalisation, blocking, fuzzy string match (Jaro-Winkler, Levenshtein), phonetic encoding, embeddings, supervised pairwise classifier; threshold tuning vs sanctions risk. LLM can help with name-transliteration and adverse-media summarisation (with human review).

**Q54. Customer risk rating model?**
Inputs: geography, product, occupation, PEP/sanction hits, transaction behaviour. Use scorecard/GBM with expert override; must be explainable and auditable; periodic recalibration.

**Q55. Sanctions screening false positives?**
Fuzzy matching creates huge noise. ML re-ranks alerts using context (DOB, country, entity type), reducing manual review while keeping a conservative recall target; always keep rule-based safety net.

### 3.4 Customer Analytics & Marketing

**Q56. Churn/attrition model for credit cards.**
Define churn (no spend 90 days or account closed), observation window features (spend trend, declines, call-centre contacts, reward redemption), GBM or survival model; act with uplift model rather than churn propensity alone (target persuadables). Retention offers cost money → optimise on incremental value.

**Q57. Uplift modelling — why not just propensity?**
Propensity targets those likely to buy; uplift targets those whose behaviour *changes because* of the treatment (persuadables) – excludes sure-things and sleeping dogs. Methods: two-model, T/X-learner, causal forests; need RCT data; evaluate with Qini/uplift curves.

**Q58. Customer lifetime value (CLV).**
BG/NBD + Gamma-Gamma for non-contractual, survival/regression for contractual; CLV = Σ (margin × survival prob) / (1+d)^t. Used for acquisition spend and offer design.

**Q59. Next-best-offer for a retail bank.**
Eligibility rules → propensity per product → expected value (margin × propensity − cost) → contact policy/fatigue → channel & timing → bandit exploration → compliance checks (suitability, TCF/Consumer Duty).

**Q60. Transaction categorisation (PFM apps).**
Text classification of merchant strings → rules + ML + LLM fallback; handle noisy descriptors; active learning from user corrections.

### 3.5 Operations & Others

**Q61. Forecasting ATM cash / call volume** — see Q35.
**Q62. Document processing (KYC docs, loan applications).** OCR (Textract/Document AI) + layout models + LLM extraction with validation; human-in-the-loop for low confidence.
**Q63. Voice/NLP in contact centres.** ASR → diarisation → sentiment/intent → summarisation (LLM) → QA scoring and compliance (mis-selling) checks.
**Q64. Model for loan pricing.** Risk-based pricing: PD/LGD + funding cost + opex + target ROE; price elasticity (acceptance model); optimise margin subject to volume constraints and fairness.

---

## 4. Python Coding Questions

**Q65. Compute PSI between train and production scores.**
```python
import numpy as np
def psi(expected, actual, bins=10):
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(actual, cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return np.sum((a - e) * np.log(a / e))
```

**Q66. Compute KS statistic and Gini.**
```python
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, _ = roc_curve(y, p)
ks = (tpr - fpr).max()
gini = 2 * roc_auc_score(y, p) - 1
```

**Q67. WOE / IV for a binned variable.**
```python
import pandas as pd, numpy as np
def woe_iv(df, col, target, bins=10):
    d = df[[col, target]].copy()
    d["bin"] = pd.qcut(d[col], bins, duplicates="drop")
    g = d.groupby("bin", observed=True)[target].agg(["count", "sum"])
    g["bad"], g["good"] = g["sum"], g["count"] - g["sum"]
    g["pb"], g["pg"] = g["bad"] / g["bad"].sum(), g["good"] / g["good"].sum()
    g["woe"] = np.log((g["pg"] + 1e-6) / (g["pb"] + 1e-6))
    g["iv"] = (g["pg"] - g["pb"]) * g["woe"]
    return g, g["iv"].sum()
```

**Q68. Fraud-model pipeline with time-based split and imbalance handling.**
```python
from sklearn.metrics import average_precision_score
import lightgbm as lgb
train = df[df.txn_date < "2025-07-01"]; test = df[df.txn_date >= "2025-07-01"]
pos_w = (train.y == 0).sum() / (train.y == 1).sum()
m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                       scale_pos_weight=pos_w, subsample=0.8, colsample_bytree=0.8)
m.fit(train[feats], train.y, eval_set=[(test[feats], test.y)],
      callbacks=[lgb.early_stopping(50)])
print(average_precision_score(test.y, m.predict_proba(test[feats])[:, 1]))
```

**Q69. Rolling features without leakage (pandas).**
```python
df = df.sort_values(["cust_id", "ts"])
df["amt_7d"] = (df.set_index("ts").groupby("cust_id")["amount"]
                  .rolling("7D", closed="left").sum().reset_index(level=0, drop=True).values)
```
`closed="left"` excludes the current transaction → no leakage.

**Q70. Pandas: top 3 spending categories per customer.**
```python
(df.groupby(["cust_id", "category"])["amount"].sum().reset_index()
   .sort_values(["cust_id", "amount"], ascending=[True, False])
   .groupby("cust_id").head(3))
```

**Q71. Why is `apply` slow and how to speed up?** Row-wise Python loops; use vectorised ops, `numpy`, `merge`, `groupby.transform`, or Polars/Spark for big data.

**Q72. Python concepts often asked:** list vs tuple, generators, decorators, `*args/**kwargs`, shallow vs deep copy, GIL, mutable default args, context managers, `__init__` vs `__new__`, vectorisation, type hints, unit testing with pytest.

---

## 5. SQL Questions (banking schema)

Schema: `customers(cust_id, open_dt, segment)`, `accounts(acct_id, cust_id, product, status)`, `transactions(txn_id, acct_id, txn_ts, amount, mcc, merchant_id, channel, is_fraud)`.

**Q73. Customers with spend > 2× their own 6-month average this month.**
```sql
WITH m AS (
  SELECT a.cust_id, DATE_TRUNC('month', t.txn_ts) mth, SUM(t.amount) spend
  FROM transactions t JOIN accounts a USING (acct_id)
  GROUP BY 1, 2),
r AS (
  SELECT *, AVG(spend) OVER (PARTITION BY cust_id ORDER BY mth
          ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING) avg6
  FROM m)
SELECT * FROM r WHERE spend > 2 * avg6;
```

**Q74. Second-highest transaction per customer.** `DENSE_RANK() OVER (PARTITION BY cust_id ORDER BY amount DESC) = 2`.

**Q75. Detect structuring: ≥3 cash deposits between 9,000 and 9,999 within 7 days.**
```sql
SELECT acct_id, COUNT(*) n, SUM(amount) total
FROM transactions
WHERE channel='CASH_DEP' AND amount BETWEEN 9000 AND 9999
  AND txn_ts >= CURRENT_DATE - INTERVAL '7 day'
GROUP BY acct_id HAVING COUNT(*) >= 3;
```

**Q76. Month-over-month retention / churn.**
```sql
WITH a AS (SELECT DISTINCT cust_id, DATE_TRUNC('month', txn_ts) mth FROM transactions t JOIN accounts USING(acct_id))
SELECT cur.mth, COUNT(DISTINCT cur.cust_id) active,
       COUNT(DISTINCT prev.cust_id) retained
FROM a cur LEFT JOIN a prev
  ON cur.cust_id = prev.cust_id AND prev.mth = cur.mth - INTERVAL '1 month'
GROUP BY 1;
```

**Q77. Fraud rate by merchant category with at least 1,000 txns.**
```sql
SELECT mcc, COUNT(*) n, AVG(is_fraud::int) fraud_rate
FROM transactions GROUP BY mcc HAVING COUNT(*) >= 1000 ORDER BY fraud_rate DESC;
```

**Q78. Concepts:** WHERE vs HAVING; INNER/LEFT/ANTI joins; `ROW_NUMBER` vs `RANK` vs `DENSE_RANK`; `LAG/LEAD`; CTE vs subquery; deleting duplicates; indexes; `NULL` handling; `COALESCE`; query optimisation (EXPLAIN, partition pruning, avoid `SELECT *`); window frames.

---

## 6. Quick Metric Cheat Sheet

| Metric | Formula / Meaning | Use |
|---|---|---|
| Precision | TP/(TP+FP) | Alert quality |
| Recall | TP/(TP+FN) | Fraud/SAR caught |
| F1 | 2PR/(P+R) | Balance |
| AUC / Gini | rank quality / 2AUC−1 | Credit models |
| KS | max(TPR−FPR) | Separation |
| PSI | Σ(A−E)ln(A/E) | Stability |
| Brier | mean((p−y)²) | Calibration |
| Lift@10% | rate in top decile / base rate | Marketing |
| Qini | uplift curve area | Uplift |
| WAPE | Σ|e|/Σ|y| | Forecasts |
