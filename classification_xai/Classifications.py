
!pip install lime
!pip install alibi
!pip install shap

!unzip /content/dataset_augmentedBin.zip
!unzip /content/datasetOG_BIN.zip

import numpy as np
import cv2
import glob
from math import log10
from pathlib import Path
import pandas as pd

AUG_BASE  = Path("/content/sample_data/datasetAUG")
ORIG_BASE = Path("/content/sample_data/datasetOG")

CLASSES = {
    "comp":    {"aug": AUG_BASE / "comp",    "orig": ORIG_BASE / "Comp-cropped"},
    "noncomp": {"aug": AUG_BASE / "noncomp", "orig": ORIG_BASE / "Non-Comp-cropped"},
}

MATCH_BY_NAME = False

def compute_mae(a, b):
    return float(np.mean(np.abs(a - b)))

def compute_mse(a, b):
    return float(np.mean((a - b) ** 2))

def compute_psnr_from_norm_mse(mse_norm, max_pixel=255.0):
    if mse_norm == 0:
        return float('inf')
    return 10.0 * log10((max_pixel ** 2) / mse_norm)

EXTS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")

def list_images(d: Path):
    paths = []
    for e in EXTS:
        paths.extend(sorted(d.glob(e)))
    return sorted(paths)

rows = []
all_mae, all_mse, all_psnr = [], [], []

for cls_name, paths in CLASSES.items():
    aug_dir = paths["aug"]
    orig_dir = paths["orig"]

    aug_paths = list_images(aug_dir)
    orig_paths = list_images(orig_dir)

    if not aug_paths:
        print(f"Augmented folder empty: {aug_dir}")
    if not orig_paths:
        print(f"Original folder empty: {orig_dir}")

    pairs = []
    if MATCH_BY_NAME:
        orig_by_stem = {p.stem: p for p in orig_paths}
        for a in aug_paths:
            if a.stem in orig_by_stem:
                pairs.append((orig_by_stem[a.stem], a))
        if len(pairs) == 0:
            n = min(len(orig_paths), len(aug_paths))
            pairs = list(zip(orig_paths[:n], aug_paths[:n]))
    else:
        n = min(len(orig_paths), len(aug_paths))
        pairs = list(zip(orig_paths[:n], aug_paths[:n]))

    maes, mses, psnrs = [], [], []

    for orig_p, aug_p in pairs:
        o = cv2.imread(str(orig_p), cv2.IMREAD_COLOR)
        a = cv2.imread(str(aug_p),  cv2.IMREAD_COLOR)
        if o is None or a is None:
            continue

        o = cv2.cvtColor(o, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if a.shape != o.shape:
            a = cv2.resize(a, (o.shape[1], o.shape[0]), interpolation=cv2.INTER_LINEAR)

        mae_n = compute_mae(o, a)
        mse_n = compute_mse(o, a)
        psnr  = compute_psnr_from_norm_mse(mse_n, max_pixel=255.0)

        maes.append(mae_n)
        mses.append(mse_n)
        psnrs.append(psnr)

        all_mae.append(mae_n)
        all_mse.append(mse_n)
        all_psnr.append(psnr)

    rows.append({
        "Class": cls_name,
        "Pairs": len(maes),
        "MAE(norm)": np.mean(maes) if maes else np.nan,
        "MSE(norm)": np.mean(mses) if mses else np.nan,
        "PSNR(dB)":  np.mean([p for p in psnrs if np.isfinite(p)]) if psnrs else np.nan
    })

if all_mae:
    rows.append({
        "Class": "overall",
        "Pairs": len(all_mae),
        "MAE(norm)": np.mean(all_mae),
        "MSE(norm)": np.mean(all_mse),
        "PSNR(dB)":  np.mean([p for p in all_psnr if np.isfinite(p)])
    })

df = pd.DataFrame(rows)[["Class", "Pairs", "MAE(norm)", "MSE(norm)", "PSNR(dB)"]]
print("\nClass            MAE(norm)    MSE(norm)     PSNR(dB)")
print("----------------------------------------------------")
for r in rows:
    print(f"{r['Class']:<15} {r['MAE(norm)']:10.4f} {r['MSE(norm)']:12.6f} {r['PSNR(dB)']:12.2f}")

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19, ResNet50, InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


DATA_DIR    = "/content/sample_data/datasetAUG"
CLASSES     = ['comp','noncomp']
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 4
INPUT_SHAPE = IMG_SIZE + (3,)
EPOCHS      = 25


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)
train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    classes=CLASSES, class_mode='binary', subset='training', seed=42
)
val_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    classes=CLASSES, class_mode='binary', subset='validation', seed=42, shuffle=False
)

print(f"→ TRAIN: {train_gen.samples} images ({int(np.ceil(train_gen.samples/BATCH_SIZE))} steps)")
print(f"→   VAL: {val_gen.samples} images ({int(np.ceil(val_gen.samples/BATCH_SIZE))} steps)")


y_train = train_gen.classes
cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(cw))


def create_binary_model(backbone_fn):
    base = backbone_fn(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model(base.input, out)


def get_callbacks(name):
    return [
        EarlyStopping('val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(f'best_{name}.keras', 'val_accuracy', save_best_only=True),
        ReduceLROnPlateau('val_loss', factor=0.2, patience=3, min_lr=1e-6),
    ]


backbones = {'VGG19': VGG19, 'ResNet50': ResNet50, 'InceptionV3': InceptionV3}
models, histories, results = {}, {}, {}

for name, fn in backbones.items():
    print(f"\n=== TRAINING {name} ===")
    model = create_binary_model(fn)
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=get_callbacks(name),
        verbose=1
    )
    histories[name] = history.history
    best = load_model(f'best_{name}.keras')
    loss, acc = best.evaluate(val_gen, verbose=0)
    results[name] = {'val_loss': loss, 'val_acc': acc}
    models[name] = best


val_gen.reset()
iterator = iter(val_gen)
val_data, val_labels = [], []
for _ in range(len(val_gen)):
    x_batch, y_batch = next(iterator)
    val_data.append(x_batch)
    val_labels.append(y_batch)
val_data   = np.concatenate(val_data, axis=0)
y_true_bin = np.concatenate(val_labels, axis=0).astype(int)

probs_list    = [m.predict(val_data, verbose=0) for m in models.values()]
ensemble_prob = np.mean(probs_list, axis=0)
ensemble_pred = (ensemble_prob > 0.5).astype(int).flatten()

ensemble_acc = accuracy_score(y_true_bin, ensemble_pred)
results['Ensemble'] = {'val_acc': ensemble_acc}

print("\n=== RESULTS (comp vs noncomp) ===")
for k,v in results.items():
    print(f"{k:12s}  val_acc = {v['val_acc']:.2%}")

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from IPython.display import display

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EXPECTED_MODELS = ['VGG19', 'ResNet50', 'InceptionV3']    # names to look for / label in results
BINARY_VAL_DIR = Path("/content/sample_data/dataset_augumentedBin/val")  # fallback val dir for binary task
IMG_SIZE = (224,224)
BATCH = 32
CLASS_NAMES = ['comp', 'noncomp']   # binary class labels used in data generator & displays

def safe_load_models(expected=EXPECTED_MODELS):
    generator.reset()
    Xs, Ys = [], []
    for i in range(len(generator)):
        xb, yb = generator[i]
        Xs.append(xb)
        Ys.append(yb)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y

def get_pos_prob(pred):
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pred.flatten()
    if pred.ndim == 2:
        if pred.shape[1] == 1:
            return pred[:,0].flatten()
        elif pred.shape[1] == 2:
            return pred[:,1].flatten()
        else:
            return pred[:,-1].flatten()
    return pred.flatten()

models = safe_load_models()

if not models:
    raise RuntimeError("No models found. Place your trained models (best_VGG19.keras etc.) in the working dir or create a 'models' dict before running this cell.")

val_data = globals().get('val_data', None)
y_true_bin = globals().get('y_true_bin', None)

if val_data is None or y_true_bin is None:
    gen_cands = [n for n in ('val_gen','val_generator','val_datagen') if n in globals()]
    if gen_cands:
        print(f"Building val arrays from generator '{gen_cands[0]}' ...")
        val_gen = globals()[gen_cands[0]]
        val_data, y_arr = build_val_arrays_from_generator(val_gen)
        if y_arr.ndim == 2:
            y_true_bin = np.argmax(y_arr, axis=1)
        else:
            y_true_bin = y_arr.flatten()
    elif BINARY_VAL_DIR.exists():
        print(f"Loading validation data from {BINARY_VAL_DIR} ...")
        dg = ImageDataGenerator(rescale=1./255)
        vg = dg.flow_from_directory(str(BINARY_VAL_DIR), target_size=IMG_SIZE, batch_size=BATCH,
                                     class_mode='categorical', shuffle=False, classes=CLASS_NAMES)
        val_data, y_arr = build_val_arrays_from_generator(vg)
        if y_arr.ndim == 2:
            y_true_bin = np.argmax(y_arr, axis=1)
        else:
            y_true_bin = y_arr.flatten()
    else:
        raise RuntimeError("val_data / y_true_bin not found and no fallback val generator or folder available.")

print(f"Validation samples: {len(y_true_bin)}; class counts: {np.unique(y_true_bin, return_counts=True)}")

model_probs = {}
model_preds = {}

for name, m in models.items():
    try:
        p = m.predict(val_data, verbose=0)
        pos = get_pos_prob(p)
        preds = (pos > 0.5).astype(int)
        model_probs[name] = pos
        model_preds[name] = preds
        print(f"{name}: produced probs for {len(pos)} samples")
    except Exception as e:
        print(f"Prediction failed for {name}: {e}")

available_probs = list(model_probs.values())
if not available_probs:
    raise RuntimeError("No model probability arrays available to form an ensemble.")
ensemble_prob = np.mean(np.vstack(available_probs), axis=0)
ensemble_pred = (ensemble_prob > 0.5).astype(int)

results = globals().get('results', {})
for name in models.keys():
    acc = accuracy_score(y_true_bin, model_preds[name]) if name in model_preds else None
    results[name] = {'val_acc': float(acc) if acc is not None else None}
results['Ensemble'] = {'val_acc': float(accuracy_score(y_true_bin, ensemble_pred))}

names = list(results.keys())   # will contain model names + 'Ensemble'
print("Results (validation accuracies):")
for n in names:
    print(f" - {n}: {results[n]['val_acc']}")

accs = []
for n in names:
    a = results[n].get('val_acc')
    if a is None:
        print(f"Warning: results['{n}']['val_acc'] is None — using 0.0 for plotting. Consider recomputing accuracy.")
        a = 0.0
        results[n]['val_acc'] = a
    accs.append(a)

plt.figure(figsize=(8,5))
colors = ['#1f77b4','#2ca02c','#d62728','#9467bd'][:len(names)]
plt.bar(names, accs, color=colors)
plt.ylim(0,1)
plt.title("Model Comparison – Binary Validation Accuracy")
plt.ylabel("Accuracy")
for i,a in enumerate(accs):
    plt.text(i, a+0.02, f"{a:.1%}", ha='center')
plt.show()

for n in names:
    if n == 'Ensemble':
        preds = ensemble_pred
    else:
        preds = model_preds[n]

    cm = confusion_matrix(y_true_bin, preds)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"{n} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.show()

    rep = classification_report(y_true_bin, preds, target_names=CLASS_NAMES, output_dict=True)
    if 'accuracy' in rep:
        rep.pop('accuracy', None)
    df = pd.DataFrame(rep).transpose().round(3)
    print(f"\n=== {n} Classification Report ===")
    display(df)

plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
for n in names:
    if n == 'Ensemble':
        prob = ensemble_prob
    else:
        prob = model_probs[n]
    fpr, tpr, _ = roc_curve(y_true_bin, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc:.3f})")
plt.plot([0,1],[0,1], 'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(); plt.grid(True)

plt.subplot(1,2,2)
for n in names:
    if n == 'Ensemble':
        prob = ensemble_prob
    else:
        prob = model_probs[n]
    prec, rec, _ = precision_recall_curve(y_true_bin, prob)
    pr_auc = auc(rec, prec)
    plt.plot(rec, prec, label=f"{n} (AUC={pr_auc:.3f})")
plt.title("Precision–Recall Curves")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

histories = globals().get('histories', {})
for backbone in EXPECTED_MODELS:
    if backbone in histories:
        hist = histories[backbone]
        epochs = range(1, len(hist.get('loss',[])) + 1)
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(epochs, hist.get('loss',[]), label='Train Loss')
        plt.plot(epochs, hist.get('val_loss',[]), label='Val Loss', linestyle='--')
        plt.title(f"{backbone} Loss vs Epochs")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.subplot(1,2,2)
        plt.plot(epochs, hist.get('accuracy',[]), label='Train Acc')
        plt.plot(epochs, hist.get('val_accuracy',[]), label='Val Acc', linestyle='--')
        plt.title(f"{backbone} Accuracy vs Epochs")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"No history found for {backbone} — skipping loss/accuracy plot.")

metrics = ['Accuracy','F1','Kappa','MCC','Specificity']
radar_data = {}

for n in names:
    if n == 'Ensemble':
        preds = ensemble_pred
    else:
        preds = model_preds[n]

    acc = accuracy_score(y_true_bin, preds)
    f1m = f1_score(y_true_bin, preds)
    kappa = cohen_kappa_score(y_true_bin, preds)
    mcc = matthews_corrcoef(y_true_bin, preds)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    radar_data[n] = [acc, f1m, kappa, mcc, spec]

df_rad = pd.DataFrame(radar_data, index=metrics)
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)
for n in names:
    vals = df_rad[n].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, marker='o', label=n)
    ax.fill(angles, vals, alpha=0.15)
ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
ax.set_ylim(0,1)
plt.title("Binary Model Comparison Radar")
plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
plt.show()

print("Done. If anything is missing (e.g. histories or models), the cell printed warnings. Upload or define the missing items and rerun.")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import lime
from lime import lime_image
from tensorflow.keras.models import load_model
from skimage.segmentation import slic

def integrated_gradients(model, image, target_class_idx, steps=50):

    img_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)

    baseline = tf.zeros_like(img_tensor)

    alphas = tf.linspace(0.0, 1.0, steps + 1)

    path_gradients = []

    for alpha in alphas:
        interpolated_image = baseline + alpha * (img_tensor - baseline)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_image)
            predictions = model(interpolated_image)
            if target_class_idx == 0:
                target_score = 1 - predictions[0][0]  # P(noncomp) = 1 - P(comp)
            else:
                target_score = predictions[0][0]  # P(comp) = sigmoid output

        gradients = tape.gradient(target_score, interpolated_image)
        path_gradients.append(gradients[0])

    path_gradients = tf.stack(path_gradients)
    avg_gradients = tf.reduce_mean(path_gradients, axis=0)

    integrated_grads = (img_tensor[0] - baseline[0]) * avg_gradients

    return integrated_grads.numpy()

def create_shap_style_visualization_with_ig(model, image, class_names=None):

    if class_names is None:
        class_names = CLASSES  # ['comp', 'noncomp'] from your binary classification

    print("Computing Integrated Gradients (SHAP-style)...")

    prediction_raw = model.predict(image[np.newaxis, ...], verbose=0)[0][0]  # Single sigmoid value

    prob_comp = prediction_raw
    prob_noncomp = 1 - prediction_raw
    prediction = np.array([prob_noncomp, prob_comp])  # [P(noncomp), P(comp)]

    predicted_class = np.argmax(prediction)  # 0=noncomp, 1=comp

    ig_attributions = integrated_gradients(model, image, predicted_class)

    if len(ig_attributions.shape) == 3:  # Color image
        attribution_magnitude = np.sum(np.abs(ig_attributions), axis=2)
    else:  # Grayscale
        attribution_magnitude = np.abs(ig_attributions)

    try:
        segments = slic(image, n_segments=50, compactness=10, sigma=1, start_label=1)

        segment_importance = []
        segment_labels = []

        for segment_id in np.unique(segments):
            mask = segments == segment_id
            segment_attr = np.mean(attribution_magnitude[mask])
            segment_importance.append(segment_attr)
            segment_labels.append(f"Region {segment_id}")

        segment_importance = np.array(segment_importance)

        if np.max(segment_importance) > 0:
            segment_importance = segment_importance / np.max(segment_importance)

    except:
        h, w = attribution_magnitude.shape
        grid_size = 4
        region_h, region_w = h // grid_size, w // grid_size

        segment_importance = []
        segment_labels = []

        for i in range(grid_size):
            for j in range(grid_size):
                start_h, end_h = i * region_h, (i + 1) * region_h
                start_w, end_w = j * region_w, (j + 1) * region_w

                region_attr = np.mean(attribution_magnitude[start_h:end_h, start_w:end_w])
                segment_importance.append(region_attr)
                segment_labels.append(f"Grid {i}_{j}")

        segment_importance = np.array(segment_importance)
        if np.max(segment_importance) > 0:
            segment_importance = segment_importance / np.max(segment_importance)

    fig = plt.figure(figsize=(20, 12))

    plt.subplot(2, 4, 1)
    plt.imshow(image)
    plt.title(f"Original Image\nPredicted: {class_names[predicted_class]}\nConfidence: {prediction[predicted_class]:.3f}")
    plt.axis('off')

    plt.subplot(2, 4, 2)
    im = plt.imshow(attribution_magnitude, cmap='RdBu_r')
    plt.title("Feature Attribution Heatmap\n(Integrated Gradients)")
    plt.colorbar(im)
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(image, alpha=0.7)
    plt.imshow(attribution_magnitude, cmap='RdBu_r', alpha=0.5)
    plt.title("Attribution Overlay")
    plt.axis('off')

    plt.subplot(2, 4, 4)
    try:
        top_segments_idx = np.argsort(segment_importance)[::-1][:10]
        segment_viz = np.zeros_like(segments, dtype=float)

        for i, seg_idx in enumerate(top_segments_idx):
            if 'Region' in segment_labels[seg_idx]:
                seg_id = int(segment_labels[seg_idx].split()[-1])
                segment_viz[segments == seg_id] = segment_importance[seg_idx]

        plt.imshow(segment_viz, cmap='hot')
        plt.title("Top Contributing Regions")
        plt.colorbar()
        plt.axis('off')
    except:
        plt.imshow(attribution_magnitude, cmap='hot')
        plt.title("Attribution Magnitude")
        plt.colorbar()
        plt.axis('off')

    plt.subplot(2, 2, 3)

    n_top_features = min(12, len(segment_importance))
    top_indices = np.argsort(segment_importance)[::-1][:n_top_features]
    top_values = segment_importance[top_indices]
    top_labels = [segment_labels[i] for i in top_indices]

    colors = []
    mean_importance = np.mean(top_values) if len(top_values) > 0 else 0

    for val in top_values:
        if val > mean_importance:
            colors.append('#ff4444')  # Red for high importance
        else:
            colors.append('#4444ff')  # Blue for lower importance

    bars = plt.barh(range(len(top_values)), top_values, color=colors, alpha=0.8)

    plt.yticks(range(len(top_values)), top_labels)
    plt.xlabel('Feature Importance Score')
    plt.title(f'Top {n_top_features} Feature Importance\n{class_names[predicted_class]}', fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at top

    for i, (bar, value) in enumerate(zip(bars, top_values)):
        plt.text(value + 0.01*max(top_values) if max(top_values) > 0 else 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    plt.subplot(2, 2, 4)

    class_probs = prediction  # [P(noncomp), P(comp)]
    class_labels = class_names

    colors = ['#ff4444' if i == predicted_class else '#4444ff' for i in range(len(class_names))]

    bars = plt.barh(range(len(class_probs)), class_probs, color=colors, alpha=0.8)

    plt.yticks(range(len(class_probs)), class_labels)
    plt.xlabel('Prediction Probability')
    plt.title('Class Probabilities', fontweight='bold')
    plt.gca().invert_yaxis()

    for i, (bar, prob) in enumerate(zip(bars, class_probs)):
        plt.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Integrated Gradients analysis completed!")
    print(f"Max attribution: {np.max(attribution_magnitude):.6f}")
    print(f"Mean attribution: {np.mean(attribution_magnitude):.6f}")
    print(f"Top region importance: {np.max(segment_importance):.3f}")

def lime_explanation_clean(model, image, true_label, class_names=None):

    if class_names is None:
        class_names = CLASSES  # ['comp', 'noncomp']

    print("Running LIME explanation...")

    explainer = lime_image.LimeImageExplainer()

    def predict_fn(images):

        raw_preds = model.predict(images, verbose=0)

        binary_preds = np.column_stack([1 - raw_preds, raw_preds])
        return binary_preds

    explanation = explainer.explain_instance(
        image.astype('double'),
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=100
    )


    raw_pred = model.predict(image[np.newaxis, ...], verbose=0)[0][0]
    pred_probas = np.array([1 - raw_pred, raw_pred])  # [P(noncomp), P(comp)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))


    axes[0].imshow(image)
    axes[0].set_title(f"Original\nTrue: {class_names[true_label]}\nConf: {pred_probas[true_label]:.3f}")
    axes[0].axis('off')


    for i, class_idx in enumerate([0, 1]):  # noncomp, comp
        temp, mask = explanation.get_image_and_mask(
            class_idx,
            positive_only=True,
            num_features=8,
            hide_rest=False
        )

        axes[i+1].imshow(image)
        axes[i+1].imshow(mask, cmap='Greens', alpha=0.6)
        axes[i+1].set_title(f"LIME: {class_names[class_idx]}\nProb: {pred_probas[class_idx]:.3f}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

print("\n" + "="*70)
print("INTEGRATED GRADIENTS (SHAP-STYLE) + LIME ANALYSIS - BINARY CLASSIFICATION")
print("="*70)

sample_indices = [10, 25]
sample_images = val_data[sample_indices]
sample_true_labels = y_true_bin[sample_indices]  # Use y_true_bin from your binary code

for model_name in ['VGG19', 'ResNet50', 'InceptionV3']:
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")

    try:
        model = load_model(f'best_{model_name}.keras')

        for i, (img, true_label) in enumerate(zip(sample_images, sample_true_labels)):
            print(f"\n--- Image {i+1}/{len(sample_images)} ---")

            print(" Feature Attribution Analysis (SHAP-style):")
            create_shap_style_visualization_with_ig(model, img)

            print(" LIME Analysis:")
            lime_explanation_clean(model, img, true_label)

    except Exception as e:
        print(f" Error with model {model_name}: {str(e)}")

print("\n" + "="*70)
print(" ANALYSIS COMPLETE!")
print("="*70)

