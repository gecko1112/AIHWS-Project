#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH = "data/Dataset/Country-Wise Data/Ireland_dataset.csv"
FEATURES = [
    "Ammonia (mg/l)",
    "Biochemical Oxygen Demand (mg/l)",
    "Dissolved Oxygen (mg/l)",
    "Orthophosphate (mg/l)",
    "pH (ph units)",
    "Temperature (cel)",
    "Nitrogen (mg/l)",
    "Nitrate (mg/l)",
]
TARGET = "CCME_WQI"
RANDOM_SEED = 42
TEST_SIZE = 0.1
VAL_SIZE  = 0.1


# In[5]:


# ── 1. Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df[TARGET].value_counts()}\n")

X = df[FEATURES].values
y_raw = df[TARGET].values


# In[8]:


# ── 2. Encode target ────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(y_raw)          # Excellent=0, Fair=1, Good=2, Marginal=3, Poor=4
num_classes = len(le.classes_)
print(f"Classes: {le.classes_}")


# In[33]:


# ── 3. Train / test split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_train
)
print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")


# In[34]:


# ── 4. Feature scaling ──────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)


# In[35]:


# ── 5. Class weights (handles imbalance) ────────────────────────────────────
class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights = dict(enumerate(class_weights_arr))
print(f"\nClass weights: { {le.classes_[k]: round(v, 2) for k, v in class_weights.items()} }\n")


# In[36]:


# ── 6. Build model ──────────────────────────────────────────────────────────
tf.random.set_seed(RANDOM_SEED)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURES),)),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# In[37]:


# ── 7. Callbacks ────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
]


# In[38]:


# ── 8. Train ────────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)


# In[39]:


# ── 9. Evaluate ─────────────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("F1-Score per class:")
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
for cls in le.classes_:
    print(f"{cls}: {report[cls]['f1-score']:.4f}")


# In[40]:


# ── 10. Training curves ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

epochs_ran = range(1, len(history.history["loss"]) + 1)

ax1.plot(epochs_ran, history.history["loss"], label="Train loss")
ax1.plot(epochs_ran, history.history["val_loss"], label="Val loss")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(epochs_ran, history.history["accuracy"], label="Train acc")
ax2.plot(epochs_ran, history.history["val_accuracy"], label="Val acc")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Plot saved to training_curves.png")


# In[42]:


import joblib

# ── 11. Save model + preprocessors ──────────────────────────────────────────
model.save("wqi_model.keras")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(le,     "label_encoder.joblib")
print("Saved: wqi_model.keras | scaler.joblib | label_encoder.joblib")


# # Feature Ablation Study
# 
# We use a Random Forest to rank feature importances, then iteratively remove the least important feature and re-train to observe the impact on macro F1-score.

# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

# ── Reload raw splits so we can index features freely ───────────────────────
df_ire = pd.read_csv(DATA_PATH)
X_raw = df_ire[FEATURES].values
y_raw_enc = le.transform(df_ire[TARGET].values)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y_raw_enc, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_raw_enc
)
X_tr, X_v, y_tr, y_v = train_test_split(
    X_tr, y_tr, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_tr
)

scaler_abl = StandardScaler()
X_tr_s = scaler_abl.fit_transform(X_tr)
X_te_s  = scaler_abl.transform(X_te)

# ── Train RF on all 8 features to get importance ranking ────────────────────
rf_full = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
rf_full.fit(X_tr_s, y_tr)

importances = rf_full.feature_importances_
feat_names  = np.array(FEATURES)

# Sort ascending — least important first (will be removed first)
sorted_idx = np.argsort(importances)
ranked_features = feat_names[sorted_idx]  # least → most important

print("Feature importance ranking (least → most important):")
for i, (name, imp) in enumerate(zip(ranked_features, importances[sorted_idx])):
    print(f"  {i+1}. {name:45s}  {imp:.4f}")


# In[10]:


# ── Feature importance bar chart ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

short_names = [n.split(" (")[0] for n in ranked_features]  # strip units for readability
bars = ax.barh(short_names, importances[sorted_idx], color="steelblue")
ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)")
ax.set_title("Random Forest Feature Importances — Ireland Dataset")
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
ax.set_xlim(0, importances.max() * 1.18)
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.show()
print("Saved to feature_importances.png")


# In[11]:


# ── Ablation: remove least important feature one by one ─────────────────────
# Start with all 8 features; each iteration drops the next least important.
# We use RF (fast) to measure macro F1 at each step.

feature_indices = list(range(len(FEATURES)))   # 0..7 in original order
removal_order   = sorted_idx.tolist()           # least → most important (index into FEATURES)

results = []
active_indices = list(range(len(FEATURES)))     # start with all

for step in range(len(FEATURES)):
    col_idx = [feature_indices[i] for i in active_indices]
    X_tr_sub = X_tr_s[:, col_idx]
    X_te_sub = X_te_s[:, col_idx]

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr_sub, y_tr)
    y_pred = rf.predict(X_te_sub)

    macro_f1 = f1_score(y_te, y_pred, average="macro")
    used_names = [FEATURES[i] for i in col_idx]
    removed    = FEATURES[removal_order[step - 1]] if step > 0 else "—"

    results.append({
        "n_features": len(active_indices),
        "removed":    removed,
        "macro_f1":   macro_f1,
        "features":   used_names,
    })
    print(f"n={len(active_indices):2d}  macro F1={macro_f1:.4f}  (just removed: {removed})")

    if active_indices:
        # Remove the least important remaining feature
        drop = removal_order[step]
        active_indices = [i for i in active_indices if i != drop]

results_df = pd.DataFrame(results)
print("\n", results_df[["n_features", "removed", "macro_f1"]].to_string(index=False))


# In[13]:


# ── Ablation curve plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(results_df["n_features"], results_df["macro_f1"], marker="o", linewidth=2, color="steelblue")
ax.set_xlabel("Number of Features")
ax.set_ylabel("Macro F1-Score")
ax.set_title("Feature Ablation — Ireland Dataset (RF classifier)")
ax.set_xticks(results_df["n_features"])
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle="--", alpha=0.5)

# Annotate each point with the feature that was just removed
for _, row in results_df.iterrows():
    label = row["removed"].split(" (")[0] if row["removed"] != "—" else "all"
    ax.annotate(
        label,
        xy=(row["n_features"], row["macro_f1"]),
        xytext=(0, 10), textcoords="offset points",
        ha="center", fontsize=7.5, rotation=30,
    )

plt.tight_layout()
plt.savefig("ablation_curve.png", dpi=150)
plt.show()
print("Saved to ablation_curve.png")


# ## Permutation Importance vs. MDI
# 
# Permutation importance measures how much macro F1 drops when a single feature's values are randomly shuffled on the **test set**. Unlike MDI, it is evaluated on held-out data and is not biased toward high-cardinality features.

# In[12]:


from sklearn.inspection import permutation_importance

# Use the RF trained on all 8 features
perm = permutation_importance(
    rf_full, X_te_s, y_te,
    scoring="f1_macro",
    n_repeats=10,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

perm_means = perm.importances_mean
perm_stds  = perm.importances_std

print("Permutation importance (mean drop in macro F1):")
perm_order = np.argsort(perm_means)
for i in perm_order:
    print(f"  {FEATURES[i]:45s}  {perm_means[i]:.4f} +/- {perm_stds[i]:.4f}")


# In[13]:


# ── Side-by-side comparison plot ─────────────────────────────────────────────
short = [n.split(" (")[0] for n in feat_names]  # strip units

# Normalise both to [0,1] for fair visual comparison
mdi_norm  = importances / importances.max()
perm_norm = perm_means  / perm_means.max() if perm_means.max() > 0 else perm_means

x = np.arange(len(FEATURES))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - width/2, mdi_norm,  width, label="MDI (normalised)",        color="steelblue",  alpha=0.85)
ax.bar(x + width/2, perm_norm, width, label="Permutation (normalised)", color="darkorange", alpha=0.85,
       yerr=perm_stds / perm_means.max(), capsize=4)

ax.set_xticks(x)
ax.set_xticklabels(short, rotation=30, ha="right")
ax.set_ylabel("Normalised Importance")
ax.set_title("Feature Importance: MDI vs. Permutation — Ireland Dataset")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("importance_comparison.png", dpi=150)
plt.show()
print("Saved to importance_comparison.png")

