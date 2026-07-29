# Price Deal Classifier

A feed-forward neural network **implemented from scratch in NumPy** — forward pass, backpropagation, and the Adam optimizer are all hand-written, with no ML framework — that classifies a product price *in market context*:

| Class | Meaning |
|---|---|
| `0` `OVERPRICED` | You are paying more than you should |
| `1` `FAIR` | Market price, nothing special |
| `2` `BEST_PRICE` | A genuine deal — this is what we want to surface |

The softmax output for class 2, `P(BEST_PRICE)`, doubles as a **ranking score**, so the same model both labels and sorts offers.

---

## Table of contents

1. [The core idea](#1-the-core-idea)
2. [Project structure](#2-project-structure)
3. [Quickstart](#3-quickstart)
4. [Data schema](#4-data-schema)
5. [Feature engineering](#5-feature-engineering)
6. [How labels are generated](#6-how-labels-are-generated)
7. [The network, formula by formula](#7-the-network-formula-by-formula)
8. [Training](#8-training)
9. [Evaluation](#9-evaluation)
10. [Results](#10-results)
11. [Testing](#11-testing)
12. [Using it with real data](#12-using-it-with-real-data)
13. [Review log — what was fixed](#13-review-log--what-was-fixed)
14. [Limitations](#14-limitations)

---

## 1. The core idea

**A raw price carries no signal on its own.** R$ 300 is a steal for a laptop and a ripoff for a coffee mug. Feeding absolute currency values into a network forces it to memorize each category's price scale, and it collapses the moment you add a new category or inflation shifts the range.

So the network **never sees currency**. It sees six **dimensionless ratios** built from the price and its market references. Formally, for any scaling constant $\alpha > 0$ applied to every currency column:

$$\phi(\alpha \cdot \mathbf{r}) = \phi(\mathbf{r})$$

where $\phi$ is the feature map. This scale invariance is enforced by a test (`test_features_are_scale_invariant`) and is why one model covers an R$ 80 accessory and an R$ 4,500 notebook simultaneously.

---

## 2. Project structure

```
price_classifier.py        # data, features, network, training, demo
test_price_classifier.py   # 27 tests, no pytest required
README.md                  # this file
price_model.npz            # written by the demo run
```

`price_classifier.py` is organized in seven sections: schema → feature engineering → dataset → math helpers → network → diagnostics → scoring/demo.

---

## 3. Quickstart

```bash
pip install numpy

python price_classifier.py        # train + evaluate + score sample offers
python test_price_classifier.py   # 27/27 tests
pytest -q test_price_classifier.py   # also works under pytest
```

Scoring your own offers:

```python
from price_classifier import PriceNet, rank_offers

model, scaler = PriceNet.load("price_model.npz")

offers = [dict(
    name="Headset A", price=189.0, category_avg=350.0, competitor_min=240.0,
    hist_min=210.0, shipping=0.0, rating=4.6, list_price=249.0,
)]

for r in rank_offers(model, scaler, offers):
    print(r["name"], r["label"], f"{r['p_best']:.1%}")
```

---

## 4. Data schema

The `raw` matrix has seven columns, defined once in `RAW_COLUMNS` so every function shares one source of truth:

| Column | Description |
|---|---|
| `price` | The offer being evaluated |
| `category_avg` | Average price of the category |
| `competitor_min` | Cheapest competing offer right now |
| `hist_min` | Lowest price seen for this product in 90 days |
| `shipping` | Shipping charged on top of the price |
| `rating` | Review score, 1–5 |
| `list_price` | Manufacturer / crossed-out price |

`engineer_features` raises `ValueError` if the width does not match — a schema mismatch fails loudly instead of silently misaligning columns.

---

## 5. Feature engineering

Let the raw columns be $p$ (price), $\bar{c}$ (category avg), $m$ (competitor min), $h$ (historical min), $s$ (shipping), $q$ (rating), $\ell$ (list price), and $\varepsilon = 10^{-9}$ a division guard.

| # | Feature | Formula | Reads as |
|---|---|---|---|
| 0 | `price_vs_category_avg` | $x_0 = \dfrac{p}{\bar{c} + \varepsilon}$ | < 1 → below category average |
| 1 | `price_vs_competitor_min` | $x_1 = \dfrac{p}{m + \varepsilon}$ | < 1 → beats every competitor |
| 2 | `price_vs_hist_min` | $x_2 = \dfrac{p}{h + \varepsilon}$ | ≈ 1 → at the 90-day floor |
| 3 | `shipping_ratio` | $x_3 = \dfrac{s}{\bar{c} + \varepsilon}$ | 0 → free shipping |
| 4 | `discount_pct` | $x_4 = \mathrm{clip}\!\left(1 - \dfrac{p}{\ell + \varepsilon},\, 0,\, 1\right)$ | fraction off list price |
| 5 | `rating_norm` | $x_5 = \dfrac{q}{5}$ | quality guard |

Every one is a ratio of two currency amounts, so the units cancel — that is the scale invariance from §1.

### Standardization

Ratios still live on different ranges ($x_2 \in [0.6, 2.3]$ vs $x_5 \in [0.2, 1.0]$), which distorts gradient descent. Each column is centred and scaled:

$$z_j = \frac{x_j - \mu_j}{\sigma_j + \varepsilon}, \qquad \mu_j, \sigma_j \text{ computed on the TRAINING SET ONLY}$$

Fitting $\mu, \sigma$ on the full dataset is a classic leak — the test set's distribution bleeds into training. `Standardizer` is fit on train and reused verbatim at inference, and `test_scaler_is_not_fit_on_test_data` guards this.

---

## 6. How labels are generated

The synthetic ground truth is deliberately **non-linear**, so a hidden layer is genuinely necessary rather than decorative.

**Price advantage** — a weighted sum of how far below each reference the price sits:

$$A = 0.6\,(1 - x_0) + 1.0\,(1 - x_1) + 0.8\,(1 - x_2)$$

**Quality gate** — a logistic gate on the rating $q$:

$$G = \sigma\big(6\,(q - 3.6)\big), \qquad \sigma(t) = \frac{1}{1 + e^{-t}}$$

**Shipping penalty** — concave, so the first cent of shipping hurts more than the last:

$$S = 1.2\,\sqrt{x_3}$$

**Deal score:**

$$\text{score} = \underbrace{A \cdot G}_{\text{multiplicative interaction}} + \; 0.35\,x_4 \; - \; S \; + \; \eta, \qquad \eta \sim \mathcal{N}(0,\, 0.10^2)$$

The term $A \cdot G$ is the key: a cheap product with a bad rating is **not** a deal, because $G \to 0$ kills the price advantage. A linear model cannot express that product of two inputs — which is exactly why the MLP beats the logistic-regression baseline by ~6 points.

Labels come from quantile cuts on the score:

$$y = \begin{cases}
0 \;(\text{overpriced}) & \text{score} < Q_{0.35} \\
2 \;(\text{best price}) & \text{score} > Q_{0.80} \\
1 \;(\text{fair}) & \text{otherwise}
\end{cases}$$

The Gaussian noise $\eta$ is deliberate: it makes the label **partly unpredictable**, so accuracy lands at a believable ~92% instead of a suspicious 98%.

---

## 7. The network, formula by formula

Architecture: `6 → 24 (ReLU) → 12 (ReLU) → 3 (softmax)`.

### 7.1 He initialization

$$W^{[l]}_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{n^{[l-1]}}\right), \qquad b^{[l]} = 0$$

ReLU zeroes roughly half its inputs, halving the variance at each layer. The factor $2/n_{\text{in}}$ compensates, keeping activation variance stable so deep stacks neither vanish nor explode. Biases start at zero because there is no symmetry to break once the weights are random.

### 7.2 Forward pass

For layers $l = 1 \ldots L$:

$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}, \qquad
A^{[l]} = \begin{cases}
\max(0, Z^{[l]}) & l < L \quad \text{(ReLU)} \\
Z^{[l]} & l = L \quad \text{(logits)}
\end{cases}$$

with $A^{[0]} = Z$, the standardized input.

### 7.3 Softmax

$$\hat{y}_k = \frac{e^{z_k - \max_j z_j}}{\sum_{i} e^{z_i - \max_j z_j}}$$

Subtracting the row maximum is mathematically a no-op (numerator and denominator share the factor $e^{-\max}$) but prevents `exp()` overflow on large logits. `test_softmax_is_overflow_safe` feeds logits of 1000 to prove it.

### 7.4 Loss — categorical cross-entropy with L2

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik}\,\log(\hat{y}_{ik} + 10^{-12}) \;+\; \frac{\lambda}{2}\sum_{l}\lVert W^{[l]}\rVert_F^2$$

The $10^{-12}$ prevents $\log(0) = -\infty$. A uniform prediction over $K$ classes costs exactly $\ln K$, which is asserted in `test_cross_entropy_bounds`.

### 7.5 Backpropagation

The elegant part. For softmax composed with cross-entropy, the messy Jacobian of the softmax cancels against the $1/\hat{y}$ from the log, and the gradient at the output collapses to a subtraction:

$$\frac{\partial \mathcal{L}}{\partial Z^{[L]}} = \frac{\hat{Y} - Y}{n}$$

Then propagating backwards through $l = L \ldots 1$:

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = (A^{[l-1]})^{\top} \delta^{[l]} + \lambda W^{[l]}, \qquad
\frac{\partial \mathcal{L}}{\partial b^{[l]}} = \sum_i \delta^{[l]}_i$$

$$\delta^{[l-1]} = \left(\delta^{[l]} (W^{[l]})^{\top}\right) \odot \mathbf{1}\!\left[A^{[l-1]} > 0\right]$$

where $\odot$ is elementwise multiplication and the indicator is the ReLU derivative. Note it is evaluated on the **activation** rather than the pre-activation — valid because $\text{ReLU}(z) > 0 \iff z > 0$, and it saves storing $Z$.

### 7.6 Gradient check

Because backprop is hand-written, the demo verifies it on every run against a central finite difference:

$$\frac{\partial \mathcal{L}}{\partial w} \approx \frac{\mathcal{L}(w + \epsilon) - \mathcal{L}(w - \epsilon)}{2\epsilon}, \qquad \epsilon = 10^{-6}$$

The relative error

$$\frac{\left|g_{\text{num}} - g_{\text{analytic}}\right|}{\left|g_{\text{num}}\right| + \left|g_{\text{analytic}}\right|}$$

measures **3.86e-08**, far below the 1e-6 threshold. The central difference is used rather than the forward one because its error is $O(\epsilon^2)$ instead of $O(\epsilon)$.

### 7.7 Adam optimizer

Per-parameter adaptive learning rates from the first and second moments of the gradient:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$m$ and $v$ start at zero and are therefore biased toward zero early on, so both are corrected:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Defaults: $\eta = 8\times10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. Dividing by $\sqrt{\hat{v}}$ means features with consistently small gradients still take meaningful steps — which matters here because the six features differ a lot in influence (see the importance table in §10).

---

## 8. Training

- **Mini-batch gradient descent**, batch size 64, reshuffled each epoch.
- **L2 weight decay** $\lambda = 10^{-4}$, added directly to the weight gradient.
- **Early stopping**: validation accuracy is tracked every epoch; training halts after 25 epochs without improvement and the **best checkpoint is restored**, not the last one. Without this, the previous version drifted (loss 0.039 → 0.056 late in training) and shipped worse weights than it had found at epoch 60.
- **Determinism**: every RNG is explicitly seeded, verified by `test_training_is_deterministic_given_a_seed`.

---

## 9. Evaluation

Per class $c$, from the confusion matrix $C$ where $C_{ij}$ counts true $i$ predicted $j$:

$$\text{precision}_c = \frac{C_{cc}}{\sum_i C_{ic}}, \qquad
\text{recall}_c = \frac{C_{cc}}{\sum_j C_{cj}}, \qquad
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

**Permutation importance** — shuffle column $j$ in the validation set, breaking its relationship with the label while preserving its marginal distribution, and measure the damage:

$$\text{importance}_j = \text{acc}(X) - \text{acc}(X^{\text{shuffled } j})$$

**Baselines.** Two are reported, because "92% accuracy" is meaningless without them:
- *Majority class* (~45%) — asserted as a floor in the tests.
- *Multinomial logistic regression* — obtained for free as `PriceNet([6, 3])`, since a network with no hidden layer **is** softmax regression. It quantifies exactly what the hidden layers buy.

---

## 10. Results

```
[sanity] backprop vs numerical gradient, max rel. error = 3.86e-08

--- baseline: logistic regression (6 -> 3) ---
test accuracy: 0.8558

--- model: 6 -> 24 (ReLU) -> 12 (ReLU) -> 3 (softmax) ---
early stop at epoch 36 (best val acc 0.919)
test accuracy: 0.9192   (+6.33 pts over the linear baseline)

class        precision   recall      f1  support
OVERPRICED       0.959    0.940   0.949      400
FAIR             0.890    0.940   0.914      551
BEST_PRICE       0.925    0.839   0.880      249
```

Confusion matrix — note there is **no** `OVERPRICED ↔ BEST_PRICE` confusion. Every error is off-by-one into `FAIR`, which is the benign failure mode: the model never calls a ripoff a bargain.

```
              OVERPRICED        FAIR  BEST_PRICE
OVERPRICED           376          24           0
FAIR                  16         518          17
BEST_PRICE             0          40         209
```

Permutation importance:

```
price_vs_competitor_min     0.219  ######################
price_vs_hist_min           0.202  ####################
rating_norm                 0.162  ################
price_vs_category_avg       0.123  ############
shipping_ratio              0.092  #########
discount_pct                0.009  #
```

Live competitor pricing and the 90-day floor dominate. `discount_pct` contributes almost nothing — consistent with reality, where a crossed-out list price is mostly marketing theatre. **This is an actionable finding: drop the feature, or replace it with something the seller cannot inflate.**

Scoring unseen offers:

```
product          price        label   P(over)  P(fair)  P(best)
Headset A       189.00   BEST_PRICE     0.000    0.002    0.998
Notebook X     3150.00   BEST_PRICE     0.000    0.013    0.987
Cheap junk       45.00         FAIR     0.000    0.999    0.001
Notebook Y     4700.00         FAIR     0.000    1.000    0.000
Headset C       470.00   OVERPRICED     1.000    0.000    0.000
```

"Cheap junk" is the interesting row: at R$ 45 against a category average of R$ 80 it is the cheapest item on the list, yet its 2.1 rating routes it to `FAIR`. The quality gate $G$ from §6 was learned correctly — the network is expressing the interaction, not just the price.

---

## 11. Testing

27 tests, runnable standalone or under pytest.

| Group | Covers |
|---|---|
| Math helpers | Softmax sums to 1, overflow safety at logits of 1000, shift invariance, one-hot, cross-entropy equals $\ln K$ at uniform, confusion matrix trace |
| Features | **Scale invariance across ×0.01 → ×1000**, schema width validation, discount bounded to [0,1], monotonicity (cheaper ⇒ higher `p_best`) |
| Standardizer | Zero mean / unit variance, raises if `transform` precedes `fit`, survives a constant column |
| Network | **Backprop vs numerical gradient**, forward shapes, He variance ≈ $\sqrt{2/n}$, rejects degenerate architectures, loss decreases, beats the majority baseline by >20 pts, determinism under seed, L2 actually shrinks weights |
| Data hygiene | Split is a true partition with no row on both sides, **scaler not fit on test data** |
| Inference | `rank_offers` sorted correctly and labels valid, save/load round-trip reproduces predictions exactly |

```
27/27 passed
```

### Documentation tests

The formulas in this file are validated too, since broken math on the repo page is a real defect:

```bash
npm install katex
node validate_readme_math.js README.md
```

It extracts every `$...$` and `$$...$$` expression (86 of them), reproduces GitHub's pre-processing — the Markdown parser consumes backslash escapes, turning `\_` into a bare `_` before KaTeX ever sees it — and renders each one with `throwOnError`. That pre-processing step is precisely what broke `\text{BEST\_PRICE}`, and why class identifiers now appear as inline code rather than inside `\text{}`.

---

## 12. Using it with real data

Replace `generate_dataset()` with your catalogue. Everything downstream is agnostic to where the rows came from:

```python
raw = load_from_your_warehouse()   # (n, 7), columns in RAW_COLUMNS order
X = engineer_features(raw)
y = your_labels                    # 0 / 1 / 2
```

**Getting labels without a labelling team.** Weak supervision works well here: define `BEST_PRICE` as an offer that sat in the bottom decile of its category for at least 48 hours *and* converted above the category median. Noisy, but the network smooths noise, and it beats waiting on hand-annotation.

**Threshold, do not argmax.** In production you rarely want `argmax`. Pick a cutoff on `P(BEST_PRICE)` from a precision/recall curve: if a false "great deal" alert costs user trust, a 0.90 cutoff with 60% recall likely beats argmax at 84% recall.

**Retraining cadence.** Price distributions drift with seasonality. Monitor accuracy weekly and retrain when it decays; the `save`/`load` pair keeps $\mu, \sigma$ bundled with the weights, so an old scaler can never be paired with new weights.

---

## 13. Review log — what was fixed

The first version had nine defects. All are resolved.

| # | Defect | Fix |
|---|---|---|
| 1 | **`list_price` computed then never used.** The variable was created in `generate_dataset` and silently discarded. | Added to `RAW_COLUMNS` and consumed by `discount_pct`. |
| 2 | **`discount_pct` did not compute a discount.** It was $1 - p/(1.25\bar{c})$ — a clipped duplicate of feature 0, not a discount at all. It also contradicted its own docstring. | Now genuinely $1 - p/\ell$. |
| 3 | **Redundant feature.** `total_cost_ratio` $= (p+s)/\bar{c}$ was almost perfectly collinear with feature 0, so the pair carried one signal in two columns. | Replaced with `shipping_ratio` $= s/\bar{c}$, which isolates the shipping signal. |
| 4 | **Label leakage.** The label was a *linear* function of the exact features fed to the network, with noise of only $\sigma = 0.06$. The reported 97.9% measured almost nothing, and a hidden layer was unnecessary. | Non-linear ground truth (multiplicative gate + $\sqrt{\cdot}$ penalty), noise raised to $\sigma = 0.10$. Honest accuracy: 91.9%, with a logistic baseline proving the hidden layers earn their keep. |
| 5 | **Hard-coded currency threshold.** `shipping = np.where(price > 500, ...)` broke the scale-invariance premise the whole design rests on. | Free shipping is now a probability, and the cost is a fraction of `category_avg`. |
| 6 | **Dead code.** `FEATURE_NAMES` was defined and never referenced; `raw_tr`/`raw_te` and the returned `history` were unused. | `FEATURE_NAMES` drives permutation importance; unused returns made optional. |
| 7 | **`rank_offers` called twice** in `main`, running full inference again just to fetch the top row. | Called once, result reused. |
| 8 | **No overfitting protection.** Loss rose from 0.039 to 0.056 late in training and the *final* — not best — weights were kept. | L2 regularization, early stopping, best-checkpoint restore. |
| 9 | **`Standardizer.transform` before `fit`** raised a bare `AttributeError`. | Explicit `RuntimeError` with a clear message; attributes initialized to `None`. |

Also added: gradient checking, permutation importance, a logistic-regression baseline, `save`/`load`, input validation on `engineer_features`, and the 27-test suite.

---

## 14. Limitations

- **The data is synthetic.** Accuracy on a real catalogue will be lower — messier features, mislabelled rows, adversarial sellers. Treat 91.9% as a ceiling.
- **No temporal validation.** A random split leaks future information into the past. With real data, split by time.
- **Class imbalance is mild here (35/45/20).** Real "best price" events are rarer; you will likely need class weights in the loss.
- **No calibration.** The softmax outputs are confident (0.998) but not calibrated probabilities. If you surface them as "98% chance this is a deal", run Platt scaling or isotonic regression first.
- **Assumes the references are trustworthy.** If `competitor_min` is scraped from a marketplace where the seller controls competing listings, the strongest feature is also the easiest to game.
