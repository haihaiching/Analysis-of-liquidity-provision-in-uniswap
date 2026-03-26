## Feature Definitions

Let $\alpha \in \mathcal{A}$ be an agent and let $\mathcal{T} \subset \mathbb{Z}_+$ denote the set of blocks in the observation period (one calendar month). $L(t, p, B)$ denotes the liquidity profile, $p(t)$ the pool price at the end of block $t$, and $\Delta p_y$ the price change for trade $y$. Let $V(t)$ denote the total trading volume in block $t$, and let $\mathrm{ABV} = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} V(t)$ be the average block volume over the observation period.

Block $t$ is said to be **active** for agent $\alpha$ if the agent holds strictly positive net liquidity at the current pool price:

$$\mathcal{T}^*(\alpha) = \bigl\{ t \in \mathcal{T} : L(t,\, p(t),\, \{\alpha\}) > 0 \bigr\}$$

Unless otherwise stated, all per-agent statistics below are computed over the active blocks $\mathcal{T}^*(\alpha)$.

---

### Position Activity

**\# Positions**

Total number of liquidity positions submitted by agent $\alpha$ during the observation period:

$$N(\alpha) = \bigl|\{ x : \alpha_x = \alpha \}\bigr|$$

**\# Active blocks**

Total number of blocks during which at least one of the agent's positions is in range, i.e. $p(t) \in [k_x^1, k_x^2]$:

$$\mathrm{ActiveBlocks}(\alpha) = |\mathcal{T}^*(\alpha)|$$

---

### Liquidity Scale

**Mean liquidity share**

Mean liquidity share across all active blocks:

$$\bar{s}(\alpha) = \frac{1}{|\mathcal{T}^*(\alpha)|} \sum_{t \in \mathcal{T}^*(\alpha)} \frac{L(t,\, p(t),\, \{\alpha\})}{L(t,\, p(t),\, \mathcal{A})}$$

**Mean active liquidity**

To compare liquidity provision across pools of different activity levels, liquidity is normalised by ABV. Define $\ell(\alpha, t) = L(t,\, p(t),\, \{\alpha\}) / \mathrm{ABV}$. The mean over active blocks is:

$$\bar{\ell}(\alpha) = \frac{1}{|\mathcal{T}^*(\alpha)|} \sum_{t \in \mathcal{T}^*(\alpha)} \ell(\alpha, t)$$

**SD active liquidity**

Standard deviation of the normalised active liquidity amount across active blocks:

$$\sigma_\ell(\alpha) = \sqrt{\frac{1}{|\mathcal{T}^*(\alpha)|} \sum_{t \in \mathcal{T}^*(\alpha)} \bigl(\ell(\alpha, t) - \bar{\ell}(\alpha)\bigr)^2}$$

**Mean tick-range width**

Mean of $(k_x^2 - k_x^1)$ across all positions of agent $\alpha$, measured in ticks. Separates narrow-range (active/JIT) LPs from wide-range (passive) LPs:

$$\overline{\Delta k}(\alpha) = \frac{1}{N(\alpha)} \sum_{\{x:\, \alpha_x = \alpha\}} (k_x^2 - k_x^1)$$

**SD tick-range width**

Standard deviation of $(k_x^2 - k_x^1)$ across all positions of agent $\alpha$. Distinguishes agents who consistently use the same range width from those who vary it, e.g. adapting to volatility regimes:

$$\sigma_{\Delta k}(\alpha) = \sqrt{\frac{1}{N(\alpha)} \sum_{\{x:\, \alpha_x = \alpha\}} \bigl((k_x^2 - k_x^1) - \overline{\Delta k}(\alpha)\bigr)^2}$$

---

### Position Dynamics

**Rebalancing frequency**

Number of (burn, mint) pairs executed within a short time window (e.g. $\leq 5$ blocks), normalised by total positions. High values indicate active rebalancers or protocol vaults that systematically adjust ranges:

$$\mathrm{Rebal}(\alpha) = \frac{|\{(x, x') : \alpha_x = \alpha_{x'} = \alpha,\; t^{\mathrm{burn}}_x \leq t^{\mathrm{mint}}_{x'} \leq t^{\mathrm{burn}}_x + 5\}|}{N(\alpha)}$$

**Fraction of single-block positions**

Fraction of positions whose mint and burn occur in the same block. High values are a strong signal of just-in-time (JIT) liquidity provision:

$$\mathrm{JIT}(\alpha) = \frac{|\{x : \alpha_x = \alpha,\; t^{\mathrm{mint}}_x = t^{\mathrm{burn}}_x\}|}{N(\alpha)}$$

---

### Market Conditions

**Mean \# trades**

Let $n(t) = |\{y : t_y = t\}|$ denote the number of trading events in block $t$. Mean number of trades per active block:

$$\bar{n}(\alpha) = \frac{1}{|\mathcal{T}^*(\alpha)|} \sum_{t \in \mathcal{T}^*(\alpha)} n(t)$$

**SD \# trades**

Standard deviation of the number of trades per active block:

$$\sigma_n(\alpha) = \sqrt{\frac{1}{|\mathcal{T}^*(\alpha)|} \sum_{t \in \mathcal{T}^*(\alpha)} \bigl(n(t) - \bar{n}(\alpha)\bigr)^2}$$

**Mean price movement**

Let $\Delta p(t) = \sum_{\{y:\, t_y = t\}} \Delta p_y$ denote the aggregated price change in block $t$ (equivalent to OFI per unit of liquidity). Mean price change per active block:

$$\overline{\Delta p}(\alpha) = \frac{1}{|\mathcal{T}^*(\alpha)|} \sum_{t \in \mathcal{T}^*(\alpha)} \Delta p(t)$$

**SD price movement**

Standard deviation of price change per active block, serving as a proxy for realised volatility during the agent's active periods:

$$\sigma_{\Delta p}(\alpha) = \sqrt{\frac{1}{|\mathcal{T}^*(\alpha)|} \sum_{t \in \mathcal{T}^*(\alpha)} \bigl(\Delta p(t) - \overline{\Delta p}(\alpha)\bigr)^2}$$

---

### Revenue

By the pro-rata fee distribution of Uniswap v3, agent $\alpha$ earns fee revenue $f(\alpha, t) = s(\alpha, t) \cdot F \cdot V(t)$ at block $t$, where $F$ is the pool fee tier. Normalising by the agent's liquidity share $s(\alpha, t)$ gives:

$$\tilde{f}(\alpha, t) = \frac{f(\alpha, t)}{s(\alpha, t)} = F \cdot V(t)$$

**Mean fees**

Mean fee revenue per active block, normalised by the agent's liquidity share:

$$\bar{f}(\alpha) = \frac{1}{|\mathcal{T}^*(\alpha)|} \sum_{t \in \mathcal{T}^*(\alpha)} \tilde{f}(\alpha, t)$$

**SD fees**

Standard deviation of fee revenue per active block, normalised by the agent's liquidity share:

$$\sigma_f(\alpha) = \sqrt{\frac{1}{|\mathcal{T}^*(\alpha)|} \sum_{t \in \mathcal{T}^*(\alpha)} \bigl(\tilde{f}(\alpha, t) - \bar{f}(\alpha)\bigr)^2}$$

---