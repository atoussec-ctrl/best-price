"""
Test suite for price_classifier.py

Runs standalone (`python test_price_classifier.py`) or under pytest
(`pytest -q test_price_classifier.py`). No dependency beyond NumPy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from price_classifier import (
    CLASS_NAMES,
    FEATURE_NAMES,
    N_CLASSES,
    N_FEATURES,
    RAW_COLUMNS,
    PriceNet,
    Standardizer,
    accuracy,
    confusion_matrix,
    cross_entropy,
    engineer_features,
    generate_dataset,
    gradient_check,
    one_hot,
    rank_offers,
    softmax,
    train_test_split,
)


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def test_softmax_rows_sum_to_one():
    z = np.random.default_rng(0).normal(size=(50, N_CLASSES)) * 10
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all() and (p <= 1).all()


def test_softmax_is_overflow_safe():
    """Naive exp() on these logits overflows to inf/nan; ours must not."""
    p = softmax(np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1000.0, -1000.0]]))
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[1], 1 / 3)


def test_softmax_shift_invariance():
    z = np.random.default_rng(1).normal(size=(8, N_CLASSES))
    assert np.allclose(softmax(z), softmax(z + 17.3))


def test_one_hot():
    oh = one_hot(np.array([0, 2, 1]), 3)
    assert np.array_equal(oh, np.eye(3)[[0, 2, 1]])
    assert np.allclose(oh.sum(axis=1), 1.0)


def test_cross_entropy_bounds():
    target = one_hot(np.array([0, 1]), 3)
    perfect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    uniform = np.full((2, 3), 1 / 3)
    assert cross_entropy(perfect, target) < 1e-6
    # A uniform prediction over k classes costs exactly ln(k).
    assert abs(cross_entropy(uniform, target) - np.log(3)) < 1e-9


def test_confusion_matrix_and_accuracy():
    y_true = np.array([0, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.sum() == len(y_true)
    assert np.trace(cm) == 3
    assert abs(accuracy(y_pred, y_true) - 0.6) < 1e-12


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_features_are_scale_invariant():
    """THE core property: multiplying every currency column by any constant must
    not change a single feature. That is what lets one model serve a R$ 80 mug
    and a R$ 4,500 notebook."""
    raw = np.array([[189.0, 350.0, 240.0, 210.0, 12.0, 4.6, 249.0]])
    currency_cols = [RAW_COLUMNS.index(c) for c in
                     ["price", "category_avg", "competitor_min", "hist_min",
                      "shipping", "list_price"]]
    for factor in (0.01, 10.0, 1000.0):
        scaled = raw.copy()
        scaled[:, currency_cols] *= factor
        assert np.allclose(engineer_features(raw), engineer_features(scaled))


def test_features_shape_and_schema():
    raw, X, y = generate_dataset(200, seed=5)
    assert raw.shape == (200, len(RAW_COLUMNS))
    assert X.shape == (200, N_FEATURES)
    assert len(FEATURE_NAMES) == N_FEATURES
    assert np.isfinite(X).all()
    assert set(np.unique(y)).issubset({0, 1, 2})


def test_engineer_features_rejects_wrong_width():
    try:
        engineer_features(np.zeros((3, 5)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a 5-column raw matrix")


def test_discount_is_bounded():
    raw, X, _ = generate_dataset(500, seed=6)
    discount = X[:, FEATURE_NAMES.index("discount_pct")]
    assert (discount >= 0).all() and (discount <= 1).all()


def test_cheaper_offer_scores_higher():
    """Monotonicity sanity check against the trained model."""
    model, scaler = _trained_model()
    base = dict(name="x", category_avg=350.0, competitor_min=300.0, hist_min=280.0,
                shipping=0.0, rating=4.5, list_price=400.0)
    cheap = rank_offers(model, scaler, [dict(base, price=180.0)])[0]
    dear = rank_offers(model, scaler, [dict(base, price=430.0)])[0]
    assert cheap["p_best"] > dear["p_best"]


# --------------------------------------------------------------------------- #
# Standardizer
# --------------------------------------------------------------------------- #
def test_standardizer_zero_mean_unit_var():
    X = np.random.default_rng(2).normal(5, 3, size=(300, N_FEATURES))
    Xs = Standardizer().fit_transform(X)
    assert np.allclose(Xs.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Xs.std(axis=0), 1, atol=1e-6)


def test_standardizer_raises_before_fit():
    try:
        Standardizer().transform(np.zeros((2, N_FEATURES)))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when transform precedes fit")


def test_standardizer_handles_constant_column():
    """A zero-variance column must not produce inf/nan (the +1e-9 guard)."""
    X = np.ones((10, N_FEATURES))
    assert np.isfinite(Standardizer().fit_transform(X)).all()


# --------------------------------------------------------------------------- #
# Network internals
# --------------------------------------------------------------------------- #
def test_backprop_matches_numerical_gradient():
    """The single most important test: analytical gradients vs central
    finite differences."""
    assert gradient_check() < 1e-6


def test_forward_shapes_and_valid_distribution():
    model = PriceNet([N_FEATURES, 8, N_CLASSES], seed=0)
    probs, acts = model.forward(np.zeros((17, N_FEATURES)))
    assert probs.shape == (17, N_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert len(acts) == 3                    # input + hidden + logits
    assert acts[0].shape == (17, N_FEATURES)


def test_he_initialization_variance():
    model = PriceNet([256, 128, N_CLASSES], seed=0)
    assert abs(model.W[0].std() - np.sqrt(2 / 256)) < 0.01
    assert np.allclose(model.b[0], 0.0)


def test_rejects_degenerate_architecture():
    try:
        PriceNet([N_FEATURES])
    except ValueError:
        return
    raise AssertionError("expected ValueError for a single-element layer list")


def test_training_reduces_loss():
    raw, X, y = generate_dataset(800, seed=3)
    Xs = Standardizer().fit_transform(X)
    model = PriceNet([N_FEATURES, 12, N_CLASSES], seed=1)
    y_onehot = one_hot(y)
    before = model.loss(Xs, y_onehot)
    model.fit(Xs, y, epochs=40, verbose=False)
    assert model.loss(Xs, y_onehot) < before * 0.6


def test_model_beats_majority_class_baseline():
    model, scaler, X_te, y_te = _trained_model(with_test=True)
    majority = max((y_te == c).mean() for c in range(N_CLASSES))
    assert accuracy(model.predict(scaler.transform(X_te)), y_te) > majority + 0.20


def test_training_is_deterministic_given_a_seed():
    def run():
        raw, X, y = generate_dataset(500, seed=9)
        Xs = Standardizer().fit_transform(X)
        m = PriceNet([N_FEATURES, 10, N_CLASSES], seed=4)
        m.fit(Xs, y, epochs=15, verbose=False)
        return m.predict(Xs)
    assert np.array_equal(run(), run())


def test_l2_shrinks_weights():
    raw, X, y = generate_dataset(600, seed=11)
    Xs = Standardizer().fit_transform(X)
    norms = []
    for l2 in (0.0, 0.05):
        m = PriceNet([N_FEATURES, 16, N_CLASSES], seed=2)
        m.fit(Xs, y, epochs=40, l2=l2, verbose=False)
        norms.append(sum(float((w ** 2).sum()) for w in m.W))
    assert norms[1] < norms[0]


# --------------------------------------------------------------------------- #
# Data hygiene
# --------------------------------------------------------------------------- #
def test_train_test_split_is_a_partition():
    raw, X, y = generate_dataset(400, seed=8)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25)
    assert len(y_tr) == 300 and len(y_te) == 100
    # No row may appear on both sides.
    tr_rows = {r.tobytes() for r in X_tr}
    assert not any(r.tobytes() in tr_rows for r in X_te)


def test_scaler_is_not_fit_on_test_data():
    """Guards against the classic leak of fitting the scaler on the full set."""
    raw, X, y = generate_dataset(400, seed=8)
    X_tr, X_te, *_ = train_test_split(X, y)
    fit_on_train = Standardizer().fit(X_tr)
    fit_on_all = Standardizer().fit(X)
    assert not np.allclose(fit_on_train.mu, fit_on_all.mu)


def test_all_three_classes_are_represented():
    raw, X, y = generate_dataset(2000, seed=12)
    counts = [int((y == c).sum()) for c in range(N_CLASSES)]
    assert all(c > 100 for c in counts), counts
    assert len(CLASS_NAMES) == N_CLASSES


# --------------------------------------------------------------------------- #
# Inference / persistence
# --------------------------------------------------------------------------- #
def test_rank_offers_sorted_and_labelled():
    model, scaler = _trained_model()
    offers = [
        dict(name="deal", price=150.0, category_avg=350.0, competitor_min=300.0,
             hist_min=280.0, shipping=0.0, rating=4.7, list_price=400.0),
        dict(name="meh", price=349.0, category_avg=350.0, competitor_min=330.0,
             hist_min=310.0, shipping=20.0, rating=4.0, list_price=360.0),
        dict(name="bad", price=520.0, category_avg=350.0, competitor_min=330.0,
             hist_min=300.0, shipping=35.0, rating=4.2, list_price=540.0),
    ]
    ranked = rank_offers(model, scaler, offers)
    assert [r["p_best"] for r in ranked] == sorted((r["p_best"] for r in ranked), reverse=True)
    assert ranked[0]["name"] == "deal"
    assert all(r["label"] in CLASS_NAMES for r in ranked)


def test_save_load_roundtrip():
    model, scaler, X_te, y_te = _trained_model(with_test=True)
    expected = model.predict(scaler.transform(X_te))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.npz"
        model.save(path, scaler)
        loaded, loaded_scaler = PriceNet.load(path)
        assert np.array_equal(loaded.predict(loaded_scaler.transform(X_te)), expected)
        assert np.allclose(loaded_scaler.mu, scaler.mu)


# --------------------------------------------------------------------------- #
# Shared fixture + runner
# --------------------------------------------------------------------------- #
_CACHE: dict = {}


def _trained_model(with_test: bool = False):
    if "m" not in _CACHE:
        raw, X, y = generate_dataset(2500, seed=21)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, seed=21)
        scaler = Standardizer().fit(X_tr)
        model = PriceNet([N_FEATURES, 24, 12, N_CLASSES], seed=3)
        model.fit(scaler.transform(X_tr), y_tr, epochs=80, verbose=False)
        _CACHE["m"] = (model, scaler, X_te, y_te)
    model, scaler, X_te, y_te = _CACHE["m"]
    return (model, scaler, X_te, y_te) if with_test else (model, scaler)


def _run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    print(f"running {len(tests)} tests\n")
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    for name, exc in failed:
        print(f"  -> {name}: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
