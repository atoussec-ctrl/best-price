"""
Price Deal Classifier
=====================

A feed-forward neural network implemented from scratch with NumPy (forward pass,
backpropagation and Adam are all hand-written) that looks at a product price *in
market context* and classifies it into one of three classes:

    0 -> OVERPRICED   pay more than you should
    1 -> FAIR         market price, nothing special
    2 -> BEST_PRICE   a genuine deal -> this is what we want to surface

Core idea: a raw price carries no signal on its own. R$ 300 is a steal for a
laptop and a ripoff for a coffee mug. The network therefore never sees absolute
currency values -- it sees six *scale-invariant ratios* built from the price and
its market references. One model then covers every category.

Run the demo:      python price_classifier.py
Run the tests:     python test_price_classifier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
CLASS_NAMES = ["OVERPRICED", "FAIR", "BEST_PRICE"]

# Column order of the `raw` matrix. Every function that touches raw data uses
# this list, so the schema is defined exactly once.
RAW_COLUMNS = [
    "price",           # the offer being evaluated
    "category_avg",    # average price of the category
    "competitor_min",  # cheapest competing offer right now
    "hist_min",        # lowest price seen for this product in 90 days
    "shipping",        # shipping cost charged on top of the price
    "rating",          # review score, 1..5
    "list_price",      # manufacturer / crossed-out price
]

FEATURE_NAMES = [
    "price_vs_category_avg",    # price / category_avg
    "price_vs_competitor_min",  # price / competitor_min
    "price_vs_hist_min",        # price / hist_min
    "shipping_ratio",           # shipping / category_avg
    "discount_pct",             # 1 - price / list_price
    "rating_norm",              # rating / 5
]

N_FEATURES = len(FEATURE_NAMES)
N_CLASSES = len(CLASS_NAMES)


# --------------------------------------------------------------------------- #
# 1. Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(raw: np.ndarray) -> np.ndarray:
    """Map raw currency columns -> dimensionless ratios.

    Every output is a ratio, so a R$ 80 accessory and a R$ 4,500 notebook land in
    the same numeric range and a single model can serve both.
    """
    raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
    if raw.shape[1] != len(RAW_COLUMNS):
        raise ValueError(
            f"raw must have {len(RAW_COLUMNS)} columns {RAW_COLUMNS}, got {raw.shape[1]}"
        )

    price, category_avg, competitor_min, hist_min, shipping, rating, list_price = raw.T
    eps = 1e-9

    return np.column_stack([
        price / (category_avg + eps),
        price / (competitor_min + eps),
        price / (hist_min + eps),
        shipping / (category_avg + eps),
        np.clip(1.0 - price / (list_price + eps), 0.0, 1.0),
        rating / 5.0,
    ])


# --------------------------------------------------------------------------- #
# 2. Dataset
# --------------------------------------------------------------------------- #
def generate_dataset(n_samples: int = 6000, seed: int = 42):
    """Synthesize offers across four categories with very different price scales.

    The label is deliberately a NON-LINEAR function of the features (a
    multiplicative quality gate and a square-root shipping penalty), plus real
    noise. That is what makes a hidden layer worth having -- see the logistic
    regression baseline in `main()`.
    """
    rng = np.random.default_rng(seed)

    scales = np.array([80.0, 350.0, 1200.0, 4500.0])
    category_avg = rng.choice(scales, size=n_samples) * rng.uniform(0.85, 1.15, n_samples)

    competitor_min = category_avg * rng.uniform(0.80, 1.05, n_samples)
    hist_min = category_avg * rng.uniform(0.70, 1.00, n_samples)

    price = category_avg * rng.uniform(0.55, 1.60, n_samples)
    # Free shipping is common on expensive items -- expressed relative to the
    # category scale, never against a hard-coded currency threshold.
    free_ship = rng.random(n_samples) < 0.45
    shipping = np.where(free_ship, 0.0, category_avg * rng.uniform(0.01, 0.12, n_samples))
    list_price = price * rng.uniform(1.0, 1.45, n_samples)
    rating = np.clip(rng.normal(4.1, 0.6, n_samples), 1.0, 5.0)

    raw = np.column_stack([price, category_avg, competitor_min,
                           hist_min, shipping, rating, list_price])
    X = engineer_features(raw)

    r_cat, r_comp, r_hist, ship_ratio, discount, rating_norm = X.T

    # --- latent "deal quality" -------------------------------------------- #
    price_advantage = 0.6 * (1 - r_cat) + 1.0 * (1 - r_comp) + 0.8 * (1 - r_hist)
    quality_gate = 1.0 / (1.0 + np.exp(-6.0 * (rating_norm * 5.0 - 3.6)))
    shipping_penalty = 1.2 * np.sqrt(ship_ratio)

    score = (
        price_advantage * quality_gate      # interaction: cheap AND decent
        + 0.35 * discount
        - shipping_penalty
        + rng.normal(0.0, 0.10, n_samples)  # irreducible noise
    )

    y = np.full(n_samples, 1, dtype=np.int64)       # FAIR
    y[score < np.quantile(score, 0.35)] = 0         # OVERPRICED
    y[score > np.quantile(score, 0.80)] = 2         # BEST_PRICE
    return raw, X, y


# --------------------------------------------------------------------------- #
# 3. Math helpers
# --------------------------------------------------------------------------- #
def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: subtracting the row max never changes the
    result but keeps exp() from overflowing."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    out = np.zeros((np.asarray(y).size, n_classes))
    out[np.arange(np.asarray(y).size), y] = 1.0
    return out


def cross_entropy(probs: np.ndarray, y_onehot: np.ndarray) -> float:
    return float(-np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1)))


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(pred) == np.asarray(y)).mean())


class Standardizer:
    """Zero mean / unit variance. Fit on TRAIN ONLY, then reused at inference so
    that production traffic is transformed exactly like the training data."""

    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-9
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None:
            raise RuntimeError("Standardizer.transform() called before fit()")
        return (np.asarray(X, dtype=np.float64) - self.mu) / self.sigma

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_test_split(X, y, raw=None, test_size=0.2, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    if raw is None:
        return X[tr], X[te], y[tr], y[te]
    return X[tr], X[te], y[tr], y[te], raw[tr], raw[te]


def confusion_matrix(y_true, y_pred, n_classes: int = N_CLASSES) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def classification_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    lines = [f"{'class':<12}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}"]
    for c, name in enumerate(CLASS_NAMES):
        tp = cm[c, c]
        precision = tp / max(cm[:, c].sum(), 1)
        recall = tp / max(cm[c, :].sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        lines.append(f"{name:<12}{precision:>10.3f}{recall:>9.3f}{f1:>8.3f}{cm[c].sum():>9}")
    return "\n".join(lines), cm


# --------------------------------------------------------------------------- #
# 4. The network
# --------------------------------------------------------------------------- #
class PriceNet:
    """MLP: input -> [Dense + ReLU] * k -> Dense -> softmax.

    Trained with mini-batch Adam on the cross-entropy loss with optional L2
    weight decay. `PriceNet([6, 3])` (no hidden layer) is exactly multinomial
    logistic regression, which is used as the baseline in `main()`.
    """

    def __init__(self, layer_sizes: list[int], seed: int = 0) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least [n_inputs, n_outputs]")
        rng = np.random.default_rng(seed)
        self.layer_sizes = list(layer_sizes)
        self.n_classes = layer_sizes[-1]
        self.W, self.b = [], []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialization: Var(W) = 2/fan_in keeps activation variance
            # roughly constant through ReLU layers.
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))
        self._init_adam()

    # -- Adam state --------------------------------------------------------- #
    def _init_adam(self) -> None:
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    # -- Forward ------------------------------------------------------------ #
    def forward(self, X: np.ndarray):
        """Returns (probabilities, activations). activations[i] is the input of
        layer i, which backprop needs."""
        activations = [np.asarray(X, dtype=np.float64)]
        a = activations[0]
        last = len(self.W) - 1
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            a = z if i == last else np.maximum(z, 0.0)   # ReLU on hidden layers only
            activations.append(a)
        return softmax(a), activations

    # -- Backward ----------------------------------------------------------- #
    def backward(self, probs, activations, y_onehot, l2: float = 0.0):
        n = y_onehot.shape[0]
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        # For softmax + cross-entropy, dL/d(logits) collapses to (probs - target).
        delta = (probs - y_onehot) / n
        for i in reversed(range(len(self.W))):
            grads_W[i] = activations[i].T @ delta + l2 * self.W[i]
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                # ReLU'(z) is 1 where the ReLU output is positive, else 0.
                delta = (delta @ self.W[i].T) * (activations[i] > 0)
        return grads_W, grads_b

    # -- Optimizer ---------------------------------------------------------- #
    def adam_step(self, gW, gb, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        bias_c1 = 1 - b1 ** self.t
        bias_c2 = 1 - b2 ** self.t
        for i in range(len(self.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * gW[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * gW[i] ** 2
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * gb[i] ** 2

            self.W[i] -= lr * (self.mW[i] / bias_c1) / (np.sqrt(self.vW[i] / bias_c2) + eps)
            self.b[i] -= lr * (self.mb[i] / bias_c1) / (np.sqrt(self.vb[i] / bias_c2) + eps)

    # -- Loss --------------------------------------------------------------- #
    def loss(self, X, y_onehot, l2: float = 0.0) -> float:
        probs, _ = self.forward(X)
        penalty = 0.5 * l2 * sum(float((w ** 2).sum()) for w in self.W)
        return cross_entropy(probs, y_onehot) + penalty

    # -- Training loop ------------------------------------------------------ #
    def fit(self, X, y, epochs=120, batch_size=64, lr=8e-3, l2=1e-4,
            X_val=None, y_val=None, patience=25, verbose=True):
        """Mini-batch Adam with L2 and early stopping on validation accuracy."""
        y_onehot = one_hot(y, self.n_classes)
        n = X.shape[0]
        rng = np.random.default_rng(7)
        history, best_val, best_state, stale = [], -np.inf, None, 0

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                probs, acts = self.forward(X[idx])
                gW, gb = self.backward(probs, acts, y_onehot[idx], l2=l2)
                self.adam_step(gW, gb, lr=lr)

            train_probs, _ = self.forward(X)
            record = {
                "epoch": epoch,
                "loss": cross_entropy(train_probs, y_onehot),
                "acc": accuracy(train_probs.argmax(1), y),
            }
            if X_val is not None:
                val_probs, _ = self.forward(X_val)
                record["val_acc"] = accuracy(val_probs.argmax(1), y_val)
                if record["val_acc"] > best_val:
                    best_val, stale = record["val_acc"], 0
                    best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                else:
                    stale += 1
            history.append(record)

            if verbose and (epoch == 1 or epoch % 20 == 0):
                msg = f"epoch {epoch:>3} | loss {record['loss']:.4f} | train acc {record['acc']:.3f}"
                if X_val is not None:
                    msg += f" | val acc {record['val_acc']:.3f}"
                print(msg)

            if X_val is not None and stale >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (best val acc {best_val:.3f})")
                break

        if best_state is not None:   # restore the best checkpoint, not the last
            self.W, self.b = best_state
        return history

    # -- Inference ---------------------------------------------------------- #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # -- Persistence -------------------------------------------------------- #
    def save(self, path: str | Path, scaler: Standardizer | None = None) -> None:
        payload = {f"W{i}": w for i, w in enumerate(self.W)}
        payload.update({f"b{i}": b for i, b in enumerate(self.b)})
        if scaler is not None and scaler.mu is not None:
            payload["mu"], payload["sigma"] = scaler.mu, scaler.sigma
        payload["meta"] = np.array(json.dumps({"layer_sizes": self.layer_sizes}))
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str | Path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        model = cls(meta["layer_sizes"])
        model.W = [data[f"W{i}"] for i in range(len(model.W))]
        model.b = [data[f"b{i}"] for i in range(len(model.b))]
        model._init_adam()
        scaler = None
        if "mu" in data.files:
            scaler = Standardizer()
            scaler.mu, scaler.sigma = data["mu"], data["sigma"]
        return model, scaler


# --------------------------------------------------------------------------- #
# 5. Diagnostics
# --------------------------------------------------------------------------- #
def gradient_check(seed: int = 0, eps: float = 1e-6) -> float:
    """Verify backprop against a central finite-difference approximation.

        dL/dw  ~=  (L(w + eps) - L(w - eps)) / (2 * eps)

    Returns the max relative error. Anything below ~1e-6 means the analytical
    gradients are correct.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(12, N_FEATURES))
    y = rng.integers(0, N_CLASSES, size=12)
    y_onehot = one_hot(y)

    model = PriceNet([N_FEATURES, 7, 5, N_CLASSES], seed=seed)
    probs, acts = model.forward(X)
    gW, gb = model.backward(probs, acts, y_onehot, l2=0.0)

    worst = 0.0
    for layer in range(len(model.W)):
        for _ in range(15):
            i = rng.integers(0, model.W[layer].shape[0])
            j = rng.integers(0, model.W[layer].shape[1])
            original = model.W[layer][i, j]

            model.W[layer][i, j] = original + eps
            loss_plus = model.loss(X, y_onehot)
            model.W[layer][i, j] = original - eps
            loss_minus = model.loss(X, y_onehot)
            model.W[layer][i, j] = original

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = gW[layer][i, j]
            denom = max(abs(numerical) + abs(analytical), 1e-12)
            worst = max(worst, abs(numerical - analytical) / denom)
    return worst


def permutation_importance(model, X, y, seed: int = 0) -> list[tuple[str, float]]:
    """Shuffle one feature column at a time; the accuracy drop is its importance."""
    rng = np.random.default_rng(seed)
    base = accuracy(model.predict(X), y)
    out = []
    for j, name in enumerate(FEATURE_NAMES):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        out.append((name, base - accuracy(model.predict(Xp), y)))
    return sorted(out, key=lambda t: t[1], reverse=True)


# --------------------------------------------------------------------------- #
# 6. Scoring new offers
# --------------------------------------------------------------------------- #
def rank_offers(model: PriceNet, scaler: Standardizer, offers: list[dict]) -> list[dict]:
    """Score a list of offer dicts and rank them by P(BEST_PRICE)."""
    raw = np.array([[o[c] for c in RAW_COLUMNS] for o in offers], dtype=float)
    probs = model.predict_proba(scaler.transform(engineer_features(raw)))
    results = [
        {
            "name": o.get("name", "?"),
            "price": o["price"],
            "label": CLASS_NAMES[int(p.argmax())],
            "p_best": float(p[2]),
            "probs": p,
        }
        for o, p in zip(offers, probs)
    ]
    return sorted(results, key=lambda r: r["p_best"], reverse=True)


# --------------------------------------------------------------------------- #
# 7. Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 72)
    print("PRICE DEAL CLASSIFIER - feed-forward NN from scratch (NumPy)")
    print("=" * 72)

    err = gradient_check()
    print(f"\n[sanity] backprop vs numerical gradient, max rel. error = {err:.2e}")

    raw, X, y = generate_dataset(6000)
    X_tr, X_te, y_tr, y_te, _, _ = train_test_split(X, y, raw)

    scaler = Standardizer().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    print(f"train: {len(y_tr)}   test: {len(y_te)}   features: {N_FEATURES}")
    print("class balance (train):",
          {CLASS_NAMES[c]: int((y_tr == c).sum()) for c in range(N_CLASSES)})

    # Baseline: no hidden layer == multinomial logistic regression.
    print("\n--- baseline: logistic regression (6 -> 3) ---")
    baseline = PriceNet([N_FEATURES, N_CLASSES], seed=3)
    baseline.fit(X_tr_s, y_tr, epochs=120, lr=8e-3, X_val=X_te_s, y_val=y_te, verbose=False)
    base_acc = accuracy(baseline.predict(X_te_s), y_te)
    print(f"test accuracy: {base_acc:.4f}")

    print("\n--- model: 6 -> 24 (ReLU) -> 12 (ReLU) -> 3 (softmax) ---")
    model = PriceNet([N_FEATURES, 24, 12, N_CLASSES], seed=3)
    model.fit(X_tr_s, y_tr, epochs=200, batch_size=64, lr=8e-3, l2=1e-4,
              X_val=X_te_s, y_val=y_te)

    y_pred = model.predict(X_te_s)
    mlp_acc = accuracy(y_pred, y_te)
    print(f"\ntest accuracy: {mlp_acc:.4f}   "
          f"(+{(mlp_acc - base_acc) * 100:.2f} pts over the linear baseline)\n")

    report, cm = classification_report(y_te, y_pred)
    print(report)
    print("\nconfusion matrix (rows = true, cols = predicted)")
    print(f"{'':<12}" + "".join(f"{n:>12}" for n in CLASS_NAMES))
    for c, name in enumerate(CLASS_NAMES):
        print(f"{name:<12}" + "".join(f"{v:>12}" for v in cm[c]))

    print("\npermutation importance (accuracy drop when the column is shuffled)")
    for name, drop in permutation_importance(model, X_te_s, y_te):
        bar = "#" * int(round(drop * 100))
        print(f"  {name:<26}{drop:>7.3f}  {bar}")

    # ---- Scoring brand-new offers ----------------------------------------- #
    offers = [
        dict(name="Headset A", price=189.0, category_avg=350.0, competitor_min=240.0,
             hist_min=210.0, shipping=0.0, rating=4.6, list_price=249.0),
        dict(name="Headset B", price=330.0, category_avg=350.0, competitor_min=310.0,
             hist_min=295.0, shipping=25.0, rating=4.4, list_price=349.0),
        dict(name="Headset C", price=470.0, category_avg=350.0, competitor_min=330.0,
             hist_min=300.0, shipping=30.0, rating=4.8, list_price=499.0),
        dict(name="Notebook X", price=3150.0, category_avg=4500.0, competitor_min=3900.0,
             hist_min=3400.0, shipping=0.0, rating=4.3, list_price=4200.0),
        dict(name="Notebook Y", price=4700.0, category_avg=4500.0, competitor_min=4300.0,
             hist_min=4100.0, shipping=0.0, rating=4.9, list_price=4900.0),
        dict(name="Cheap junk", price=45.0, category_avg=80.0, competitor_min=62.0,
             hist_min=55.0, shipping=8.0, rating=2.1, list_price=79.0),
    ]

    ranked = rank_offers(model, scaler, offers)
    print("\n" + "=" * 72)
    print("SCORING NEW OFFERS  (ranked by P(BEST_PRICE))")
    print("=" * 72)
    print(f"{'product':<12}{'price':>10}{'label':>13}{'P(over)':>10}{'P(fair)':>9}{'P(best)':>9}")
    for r in ranked:
        p = r["probs"]
        print(f"{r['name']:<12}{r['price']:>10.2f}{r['label']:>13}"
              f"{p[0]:>10.3f}{p[1]:>9.3f}{p[2]:>9.3f}")

    best = ranked[0]
    print(f"\n>> Best price detected: {best['name']} at {best['price']:.2f} "
          f"(confidence {best['p_best']:.1%})")

    model.save("price_model.npz", scaler)
    reloaded, reloaded_scaler = PriceNet.load("price_model.npz")
    same = np.array_equal(reloaded.predict(reloaded_scaler.transform(X_te)), y_pred)
    print(f"\n[persistence] saved to price_model.npz, reload reproduces predictions: {same}")


if __name__ == "__main__":
    main()
