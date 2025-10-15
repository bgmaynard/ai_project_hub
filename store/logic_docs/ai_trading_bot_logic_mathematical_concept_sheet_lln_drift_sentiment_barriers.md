# AI Trading Bot – Logic & Mathematical Concept Sheet (LLN, Drift, Sentiment, Round/Half Barriers)

**Purpose:** A durable, implementation‑agnostic spec you can embed in project docs and code comments. It defines signals, math, update rules, and how to fuse them into a single tradeable probability used for sizing and order type.

---

## 1) Scope & Objectives
- Convert microstructure + sentiment into a short‑horizon probability of **up move** \(P_{\uparrow}(\tau)\).
- Model **fill probability** and **slippage** to choose market vs. limit.
- Use **LLN** (running averages/EWMAs) so estimators stabilize as samples grow.
- Auto‑update and calibrate online; adapt to regime changes.

**Horizon (τ):** fixed short interval (e.g., 1–5 seconds). Labels/targets are defined over τ.

---

## 2) Data & Notation
- Prices: best bid \(b_t\), best ask \(a_t\), mid \(m_t = (b_t+a_t)/2\).
- Depths (top‑of‑book or summed top-N): \(D^B_t\) (buy), \(D^A_t\) (sell).
- Spread: \( \text{spr}_t = \frac{a_t-b_t}{m_t} \).
- VWAP: session VWAP \(\text{VWAP}_t\) (or rolling proxy).
- Sentiment score \(s_t \in [-1,1]\).
- Return: \( r_t = \frac{m_t - m_{t-1}}{m_{t-1}} \).
- Label (realized next move up within τ): \( y_t \in \{0,1\} \).

**IBKR mapping (examples):**
- L1: bid, ask, bidSize, askSize, last, VWAP.
- L2 (optional): per‑level sizes to compute \(D^B_t, D^A_t\) (sum top 5, etc.).

---

## 3) Feature Engineering
1. **Order Book Imbalance** \( \text{imb}_t = \frac{D^B_t - D^A_t}{D^B_t + D^A_t + \varepsilon} \in [-1,1] \).
2. **Short Momentum (vol‑normalized)**
   - Volatility proxy: EWMA of \(|r_t|\): \(\hat{\sigma}_t = \text{EWMA}_\alpha(|r_t|)\).
   - Momentum: \( \text{mom}_t = r_t / (\hat{\sigma}_t + \epsilon) \).
3. **Drift (slow)** \( \text{drift}_t = \text{EWMA}_\beta(r_t) \).
4. **VWAP Distance (scaled)** \( \text{vwapd}_t = \frac{m_t - \text{VWAP}_t}{0.001\,m_t} \)  (≈ z‑score vs 0.1% band).
5. **Spread (normalized)** \( \text{spr}_t = \frac{a_t-b_t}{m_t} \).
6. **Round/Half Barriers**  (\(x.00\), \(x.50\))
   - Fractional part: \(f_t = m_t - \lfloor m_t \rfloor\).
   - Proximities: \(p_{00} = 1 - |f_t - 0.00|/0.05\), \(p_{50} = 1 - |f_t - 0.50|/0.05\), clipped to \([0,1]\).
   - **Barrier strength**: \( \text{barrier}_t = \max(p_{00}, p_{50}) \).
7. **Sentiment**: external feed scaled \([-1,1]\).

---

## 4) LLN‑Style Estimators
- **EWMA**: \(X_t^{(\text{EWMA})} = \alpha X_t + (1-\alpha) X_{t-1}^{(\text{EWMA})}\), for drift and volatility.
- **Running mean/variance (Welford)** for realized slippage and fill rates; variance shrinks \(\sim\) LLN with more samples.

---

## 5) Direction Model (Logistic)
**Form:**
\[
P_{\uparrow}(t) = \sigma\big(\beta_0 + \beta_1\,\text{imb}_t + \beta_2\,\text{mom}_t + \beta_3\,\text{drift}_t + \beta_4\,s_t + \beta_5\,\text{barrier}_t + \beta_6\,\text{vwapd}_t + \beta_7\,\text{spr}_t\big),
\]
with \(\sigma(z) = 1/(1+e^{-z})\).

**Online update (per tick, SGD on logistic loss):**
- Error: \( e_t = y_t - P_{\uparrow}(t) \).
- Update: \( \beta \leftarrow \beta - \eta \,( -e_t \cdot x_t + \lambda \beta) \)  (\(\eta\): learning rate, \(\lambda\): L2).

---

## 6) Calibration & Reliability (Rolling Backtest)
- Partition \([0,1]\) into B bins. For each predicted \(p\), store \((p,y)\) into its bin.
- **Bin hit rate:** \(\hat{h}_b = \frac{\sum y}{\text{count}}\). **Mean p:** \(\bar{p}_b\).
- **Recalibrate:** \( p_{cal} = p \times (\hat{h}_b / (\bar{p}_b + \epsilon)) \) for bins with enough samples.
- **Reliability score (R ∈ [0,1])** from rolling Brier error: \( \text{Brier} = \overline{(p - y)^2} \), then map to \(R\) (e.g., \(R = \max(0, \min(1, 1 - 2\,\text{Brier}))\)).

---

## 7) Similarity‑Based Probability Coefficient
- Build a compact **signature**: \(\xi_t = (\text{imb},\text{mom},\text{drift},s,\text{barrier},\text{vwapd},\text{spr})\).
- Compute cosine‑like similarity to a rolling history; take top‑k neighbors.
- **Similarity win rate:** \( w_t = \frac{\#\,\text{ups in top‑k}}{k} \).
- Convert to a **boost** around 0.5: \( \text{boost}_t = c\,(w_t - 0.5) \) (e.g., \(c=0.4\)).
- **Multiplier:** \( M_t = e^{\text{boost}_t} \in [\approx 0.82, 1.22] \).

---

## 8) Fused Final Probability
\[
\boxed{\; p_{final}(t) = \text{clip}_{[0,1]}\Big(\underbrace{p_{cal}(t)}_{\text{calibration}} \times \underbrace{R(t)}_{\text{reliability}} \times \underbrace{M_t}_{\text{similarity boost}}\Big) \;}
\]
Use \(p_{final}\) for thresholding, sizing, and gating order type.

---

## 9) Fill Probability & Slippage
**Limit at touch:** with queue ahead \(Q\), market‑order hit rate \(\lambda_m\) and cancel rate ahead \(\lambda_c\), over horizon \(\tau\):
\[ P(\text{fill}\,|\,\tau) \approx 1 - e^{- (\lambda_m + \lambda_c)\,\tau / Q } .\]

**Realized slippage (market):**
- Buy: \( \text{slip} = \max(0, \text{fill} - \text{ask}) \).
- Sell: \( \text{slip} = \max(0, \text{bid} - \text{fill}) \).
Track running mean/std (Welford). Use mean slippage as **cost add‑on** in expectancy.

---

## 10) Decision Policy (example)
- **Trade gate:** only act if \(p_{final} \ge \theta\) and \(\text{spr}\) below cap.
- **Order type:**
  - If **limit**: require \(P(\text{fill}|\tau) \ge \phi\). Else fall back to market.
  - If **market**: expected entry = \(a_t + \mathbb{E}[\text{slip}]\) (buy) / \(b_t - \mathbb{E}[\text{slip}]\) (sell).
- **Sizing:** Kelly‑fraction proxy on calibrated edge: size \(\propto (p_{final}-0.5)\), capped by risk.
- **Exits:** mirror logic for down‑move; incorporate spread/slippage into stop/target.

---

## 11) Online Learning Protocol
1. On tick t: compute features, get \(p_t\).
2. After horizon τ: realize \(y_t\), push \((p_t,y_t)\) to calibration; update logistic \(\beta\).
3. Update similarity store with signature & outcome.
4. Update EWMAs and running slippage/fill stats.

**Hyperparameters:** learning rate \(\eta\), L2 \(\lambda\), EWMA alphas, bins B, window sizes, k‑neighbors, thresholds \(\theta,\phi\).

---

## 12) Backtesting & Metrics
- **Accuracy / AUC** over rolling windows.
- **Calibration**: reliability diagram, Brier score.
- **Trade metrics**: hit rate, avg P&L per trade, Sharpe, drawdown, slippage vs. model.
- **Ablations**: turn off features (sentiment, barrier) to measure marginal value.

---

## 13) Guardrails & Regime Handling
- Pause trading on extreme spreads / halted markets.
- Reduce size when reliability \(R\) drops below threshold.
- Include volatility regime (e.g., high/low via \(\hat{\sigma}_t\) quantiles) as a feature or interaction (barrier×regime).

---

## 14) Pseudocode (live loop)
```text
init AlphaFusion state (β, EWMAs, calibration, similarity store)
for each tick t:
  read (b_t, a_t, D^B_t, D^A_t, VWAP_t, s_t)
  compute features -> p_model
  p_cal  = calibrate(p_model)
  R      = reliability()
  w_t    = similarity_winrate(signature)
  M_t    = exp(c*(w_t-0.5))
  p_final= clip(p_cal * R * M_t)

  if p_final >= θ and spr <= spr_cap:
      decide market vs limit using P(fill|τ) & slippage stats
      place order & log

  when label y_t realized after τ:
      update β via SGD; push to calibration & similarity store
      update LLN stats (slippage/fill)
```

---

## 15) Implementation Notes
- Keep τ constant during training; re‑train if τ changes.
- Normalize/clip outliers in features.
- Persist state (β, calibration bins, similarity cache, EWMAs) snapshot to disk.
- Version hyperparameters in config; log feature values alongside decisions.

---

## 16) References (conceptual)
- Logistic regression for short‑horizon direction in microstructure.
- Reliability diagrams/Brier score for calibration.
- Queueing/Poisson approximations for best‑price limit fills.
- EWMA estimators for intraday drift/volatility.

---

**Deliverables already provided:**
- Python module (LLN + microstructure + sentiment + barriers + calibration + similarity): `alpha_fusion/`.
- Demo runner & JSON summary for quick validation.

> This sheet is designed to drop into your project wiki/`/docs/` and code comments so the math stays aligned with implementation as we iterate.

