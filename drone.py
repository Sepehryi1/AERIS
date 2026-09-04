from pathlib import Path
import re
import gc
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)

# ============================================================
# Configuration
# ============================================================
DATASET_DIR = Path("./")

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
LABELS_PATH = DATASET_DIR / "train_labels.csv"
OUTPUT_PATH = Path("./submission.csv")

# RAM-safe settings
CONTEXT = 2
MAX_POS_PER_IMAGE = 120
MAX_NEG_PER_IMAGE = 180

DEFAULT_THRESHOLD = 0.50
DEFAULT_SIGMA = 1.0
DEFAULT_MIN_DURATION = 0.08
DEFAULT_MAX_GAP = 0.10

assert TRAIN_DIR.exists(), f"Train directory not found: {TRAIN_DIR}"
assert TEST_DIR.exists(), f"Test directory not found: {TEST_DIR}"
assert LABELS_PATH.exists(), f"Labels not found: {LABELS_PATH}"

# ============================================================
# Load and validate labels
# ============================================================
labels = pd.read_csv(LABELS_PATH)

required = {"image_file", "audio_duration", "drone_count", "intervals"}
missing = required - set(labels.columns)
assert not missing, f"Missing columns: {missing}"

labels["image_file"] = labels["image_file"].astype(str)
labels["audio_duration"] = pd.to_numeric(labels["audio_duration"], errors="coerce")
labels["drone_count"] = pd.to_numeric(labels["drone_count"], errors="coerce").astype("Int64")
labels["intervals"] = labels["intervals"].fillna("").astype(str)

print("Labels:", labels.shape)
display(labels["drone_count"].value_counts().sort_index())
display(labels["audio_duration"].describe())

# ============================================================
# Parse intervals
# ============================================================
INTERVAL_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*$"
)

def parse_intervals(text):
    if text is None or str(text).strip() == "":
        return []
    out = []
    for part in str(text).split(";"):
        m = INTERVAL_RE.match(part)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if b > a:
                out.append((a, b))
    return sorted(out)

labels["parsed_intervals"] = labels["intervals"].map(parse_intervals)

mismatch = (
    labels["parsed_intervals"].map(len)
    != labels["drone_count"].astype(int)
).sum()

print("Interval/count mismatches:", mismatch)

# ============================================================
# Dataset metadata — lightweight
# ============================================================
def image_width(path):
    with Image.open(path) as im:
        return im.size[0]

meta_rows = []
for fname, dur in labels[["image_file", "audio_duration"]].itertuples(index=False):
    p = TRAIN_DIR / fname
    if p.exists():
        with Image.open(p) as im:
            meta_rows.append((fname, im.size[0], im.size[1], float(dur)))

meta = pd.DataFrame(
    meta_rows,
    columns=["image_file", "width", "height", "duration"]
)

print("Usable train images:", len(meta))
display(meta.describe())

# ============================================================
# Compact feature extraction
# ============================================================
def load_image_array(path):
    return np.asarray(
        Image.open(path).convert("RGB"),
        dtype=np.float32
    ) / 255.0


def compact_column_features(rgb):
    # Only compact statistics are retained.
    gray = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    )
    value = rgb.max(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)

    H, W = gray.shape
    edges = np.linspace(0, H, 9, dtype=int)

    feats = [
        gray.mean(axis=0),
        gray.std(axis=0),
        gray.max(axis=0),
        value.mean(axis=0),
        value.std(axis=0),
        sat.mean(axis=0),
        sat.std(axis=0),
    ]

    # 8 frequency bands
    for i in range(8):
        band = gray[edges[i]:edges[i+1], :]
        feats.append(band.mean(axis=0))

    X = np.stack(feats, axis=1).astype(np.float32)

    # spectral centroid + spread
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    energy = gray + 1e-6
    total = energy.sum(axis=0)

    centroid = (energy * yy).sum(axis=0) / total
    spread = np.sqrt(np.maximum(
        (energy * (yy - centroid[None, :]) ** 2).sum(axis=0) / total,
        0
    ))

    return np.column_stack([X, centroid, spread]).astype(np.float32)


def temporal_context(X, context=CONTEXT):
    W, F = X.shape
    padded = np.pad(X, ((context, context), (0, 0)), mode="edge")
    return np.concatenate(
        [padded[i:i+W] for i in range(2*context+1)],
        axis=1
    ).astype(np.float32)


def extract_features(path):
    rgb = load_image_array(path)
    base = compact_column_features(rgb)

    # First temporal derivative only.
    d1 = np.diff(base, axis=0, prepend=base[:1])
    X = np.concatenate([base, d1], axis=1)

    result = temporal_context(X, CONTEXT)

    del rgb, base, d1, X
    return result


def interval_to_column_labels(intervals, duration, width):
    y = np.zeros(width, dtype=np.int8)

    if duration <= 0:
        return y

    for a, b in intervals:
        x0 = int(np.floor(a / duration * width))
        x1 = int(np.ceil(b / duration * width))

        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))

        y[x0:x1] = 1

    return y

# ============================================================
# Image-level split
# ============================================================
label_map = {
    row["image_file"]: row
    for _, row in labels.iterrows()
    if (TRAIN_DIR / row["image_file"]).exists()
}

image_names = np.array(sorted(label_map.keys()))

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)

train_idx, val_idx = next(
    gss.split(image_names, groups=image_names)
)

train_images = image_names[train_idx]
val_images = image_names[val_idx]

print("Train images:", len(train_images))
print("Validation images:", len(val_images))

# ============================================================
# Memory-safe training matrix
# ============================================================
def balanced_indices(y, max_pos=120, max_neg=180, seed=42):
    r = np.random.default_rng(seed)

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    if len(pos) > max_pos:
        pos = r.choice(pos, max_pos, replace=False)

    if len(neg) > max_neg:
        neg = r.choice(neg, max_neg, replace=False)

    idx = np.concatenate([pos, neg])
    r.shuffle(idx)
    return idx


def build_sampled_training_matrix(image_names):
    X_parts = []
    y_parts = []

    for n, fname in enumerate(image_names, 1):
        row = label_map[fname]
        X = extract_features(TRAIN_DIR / fname)

        y = interval_to_column_labels(
            row["parsed_intervals"],
            float(row["audio_duration"]),
            X.shape[0]
        )

        idx = balanced_indices(
            y,
            MAX_POS_PER_IMAGE,
            MAX_NEG_PER_IMAGE,
            SEED
        )

        X_parts.append(X[idx])
        y_parts.append(y[idx])

        del X, y, idx

        if n % 100 == 0 or n == len(image_names):
            print(f"Preparing training data: {n}/{len(image_names)}")

    X_out = np.vstack(X_parts).astype(np.float32, copy=False)
    y_out = np.concatenate(y_parts).astype(np.int8, copy=False)

    del X_parts, y_parts
    gc.collect()

    return X_out, y_out


X_train, y_train = build_sampled_training_matrix(train_images)

print("Training shape:", X_train.shape)
print("Feature matrix RAM:",
      round(X_train.nbytes / 1024**2, 1), "MB")
print("Positive rate:", round(float(y_train.mean()), 4))

# ============================================================
# Train classifier
# ============================================================
model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=220,
    max_leaf_nodes=31,
    min_samples_leaf=25,
    l2_regularization=1.0,
    random_state=SEED
)

model.fit(X_train, y_train)

del X_train, y_train
gc.collect()

print("Model trained.")

# ============================================================
# Temporal post-processing
# ============================================================
def probability_to_intervals(
    prob,
    duration,
    threshold=0.5,
    sigma=1.0,
    min_duration=0.08,
    max_gap=0.10
):
    p = np.asarray(prob, dtype=np.float32)

    if sigma > 0:
        p = gaussian_filter1d(p, sigma=sigma)

    active = p >= threshold

    # Bridge only very small gaps.
    if max_gap > 0 and len(active) > 1:
        dt = duration / len(active)
        max_gap_bins = int(np.floor(max_gap / dt))

        if max_gap_bins > 0:
            i = 0
            while i < len(active):
                if active[i]:
                    i += 1
                    continue

                j = i
                while j < len(active) and not active[j]:
                    j += 1

                if (
                    i > 0
                    and j < len(active)
                    and (j - i) <= max_gap_bins
                ):
                    active[i:j] = True

                i = j

    intervals = []
    i = 0
    W = len(active)

    while i < W:
        if not active[i]:
            i += 1
            continue

        start = i

        while i < W and active[i]:
            i += 1

        end = i

        a = start / W * duration
        b = end / W * duration

        if b - a >= min_duration:
            intervals.append((a, b))

    return intervals


def temporal_iou(a, b):
    inter = max(
        0.0,
        min(a[1], b[1]) - max(a[0], b[0])
    )

    union = max(a[1], b[1]) - min(a[0], b[0])

    return inter / union if union > 0 else 0.0


def matched_tiou(pred, gt):
    if not pred or not gt:
        return 0.0, 0, len(pred), len(gt)

    matrix = np.array([
        [temporal_iou(p, g) for g in gt]
        for p in pred
    ])

    rows, cols = linear_sum_assignment(-matrix)

    matched = [
        matrix[r, c]
        for r, c in zip(rows, cols)
        if matrix[r, c] > 0
    ]

    tp = len(matched)
    fp = len(pred) - tp
    fn = len(gt) - tp

    denominator = tp + fp + fn

    score = (
        float(np.sum(matched) / denominator)
        if denominator
        else 1.0
    )

    return score, tp, fp, fn


def competition_score(pred_by_image, gt_by_image):
    numerator = 0.0
    denominator = 0
    rows = []

    for fname, gt in gt_by_image.items():
        score, tp, fp, fn = matched_tiou(
            pred_by_image.get(fname, []),
            gt
        )

        den = tp + fp + fn
        numerator += score * den
        denominator += den

        rows.append((fname, score, tp, fp, fn))

    final_score = (
        100.0 * numerator / denominator
        if denominator
        else 0.0
    )

    details = pd.DataFrame(
        rows,
        columns=["image_file", "score", "TP", "FP", "FN"]
    )

    return final_score, details

# ============================================================
# Validation — one image at a time
# ============================================================
val_rows = labels[
    labels["image_file"].isin(set(val_images))
].copy()

gt_val = {
    r["image_file"]: r["parsed_intervals"]
    for _, r in val_rows.iterrows()
}

duration_val = {
    r["image_file"]: float(r["audio_duration"])
    for _, r in val_rows.iterrows()
}

pred_cache_val = {}

for n, fname in enumerate(val_images, 1):
    X = extract_features(TRAIN_DIR / fname)

    pred_cache_val[fname] = (
        model.predict_proba(X)[:, 1]
        .astype(np.float32)
    )

    del X

    if n % 100 == 0 or n == len(val_images):
        print(f"Validation prediction: {n}/{len(val_images)}")

gc.collect()

# ============================================================
# Optimize post-processing parameters
# ============================================================
results = []

thresholds = np.arange(0.25, 0.76, 0.05)
sigmas = [0.0, 0.7, 1.0, 1.5]
min_durations = [0.04, 0.08, 0.12, 0.18]
max_gaps = [0.0, 0.05, 0.10, 0.15]

for threshold in thresholds:
    for sigma in sigmas:
        for min_duration in min_durations:
            for max_gap in max_gaps:

                pred = {}

                for fname, p in pred_cache_val.items():
                    pred[fname] = probability_to_intervals(
                        p,
                        duration_val[fname],
                        threshold,
                        sigma,
                        min_duration,
                        max_gap
                    )

                score, _ = competition_score(pred, gt_val)

                results.append(
                    (
                        score,
                        threshold,
                        sigma,
                        min_duration,
                        max_gap
                    )
                )

search_results = pd.DataFrame(
    results,
    columns=[
        "score",
        "threshold",
        "sigma",
        "min_duration",
        "max_gap"
    ]
).sort_values("score", ascending=False)

display(search_results.head(15))

best = search_results.iloc[0].to_dict()

print("Best parameters:")
print(best)

# ============================================================
# Validation score with best parameters
# ============================================================
best_threshold = float(best["threshold"])
best_sigma = float(best["sigma"])
best_min_duration = float(best["min_duration"])
best_max_gap = float(best["max_gap"])

pred_val = {}

for fname, p in pred_cache_val.items():
    pred_val[fname] = probability_to_intervals(
        p,
        duration_val[fname],
        best_threshold,
        best_sigma,
        best_min_duration,
        best_max_gap
    )

val_score, val_details = competition_score(
    pred_val,
    gt_val
)

print(f"Validation score: {val_score:.4f}")
display(val_details.sort_values("score").head(15))

del pred_val, pred_cache_val
gc.collect()

# ============================================================
# Estimate test duration from train
# ============================================================
duration_reg = LinearRegression()
duration_reg.fit(meta[["width"]], meta["duration"])

width_duration_map = (
    meta.groupby("width")["duration"]
    .median()
    .to_dict()
)

def estimate_duration_for_test(path):
    w = image_width(path)

    if w in width_duration_map:
        return float(width_duration_map[w])

    d = float(
        duration_reg.predict(
            np.array([[w]], dtype=np.float32)
        )[0]
    )

    return float(
        np.clip(
            d,
            labels["audio_duration"].min(),
            labels["audio_duration"].max()
        )
    )

print("Duration estimator ready.")

# ============================================================
# Final training — still memory safe
# ============================================================
X_final, y_final = build_sampled_training_matrix(image_names)

print("Final matrix:", X_final.shape)
print("RAM:",
      round(X_final.nbytes / 1024**2, 1), "MB")

final_model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=220,
    max_leaf_nodes=31,
    min_samples_leaf=25,
    l2_regularization=1.0,
    random_state=SEED
)

final_model.fit(X_final, y_final)

del X_final, y_final
gc.collect()

print("Final model trained.")

# ============================================================
# Predict test — one image at a time
# ============================================================
test_paths = sorted([
    p for p in TEST_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
])

print("Test images:", len(test_paths))

rows = []

for n, path in enumerate(test_paths, 1):
    duration = estimate_duration_for_test(path)

    X = extract_features(path)

    p = final_model.predict_proba(X)[:, 1]

    del X
    gc.collect()

    intervals = probability_to_intervals(
        p,
        duration,
        best_threshold,
        best_sigma,
        best_min_duration,
        best_max_gap
    )

    clean = []

    for a, b in intervals:
        a = max(0.0, min(duration, float(a)))
        b = max(0.0, min(duration, float(b)))

        if b > a:
            clean.append((a, b))

    rows.append({
        "image_file": path.name,
        "drone_count": len(clean),
        "intervals": ";".join(
            f"{a:.2f}-{b:.2f}"
            for a, b in clean
        )
    })

    del p, intervals, clean

    if n % 100 == 0 or n == len(test_paths):
        print(f"Test prediction: {n}/{len(test_paths)}")

submission = pd.DataFrame(
    rows,
    columns=["image_file", "drone_count", "intervals"]
)

submission.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH.resolve())
display(submission.head(20))

# ============================================================
# Final submission validation
# ============================================================
check = pd.read_csv(OUTPUT_PATH)

assert list(check.columns) == [
    "image_file",
    "drone_count",
    "intervals"
]

assert len(check) == len(test_paths)
assert check["image_file"].is_unique
assert (check["drone_count"] >= 0).all()

for _, row in check.iterrows():
    parsed = parse_intervals(row["intervals"])
    assert len(parsed) == int(row["drone_count"])

print("Submission format is valid.")
print("Rows:", len(check))
print("File:", OUTPUT_PATH.resolve())
