# =====================================================
# Train Prompt Injection Detector from Hugging Face Dataset
# =====================================================

import torch
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
import warnings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =====================================================
# CONFIG
# =====================================================
DATASET_NAME = "Smooth-3/llm-prompt-injection-attacks"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./prompt_injection_model"

MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5

LABELS = [
    "BENIGN",
    "JAILBREAK",
    "INSTRUCTION_OVERRIDE",
    "ROLE_HIJACK",
    "DATA_EXFILTRATION"
]

# =====================================================
# STEP 1: LOAD DATASET FROM HUGGING FACE
# =====================================================
print("Loading dataset from Hugging Face...")

try:
    dataset = load_dataset(DATASET_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load dataset '{DATASET_NAME}': {e}")

# Validate splits exist
if "train" not in dataset or "validation" not in dataset:
    raise ValueError("Dataset must contain 'train' and 'validation' splits")

train_ds = dataset["train"]
val_ds = dataset["validation"]

print(f"Train samples: {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}")
print(f"Train columns: {train_ds.column_names}")

# Validate required columns
if "text" not in train_ds.column_names or "labels" not in train_ds.column_names:
    raise ValueError("Dataset must contain 'text' and 'labels' columns")

# Check for empty datasets
if len(train_ds) == 0 or len(val_ds) == 0:
    raise ValueError("Train or validation dataset is empty")

# =====================================================
# STEP 2: LABEL BINARIZATION
# =====================================================
print("\nEncoding labels...")

# Inspect label format
sample_labels = train_ds["labels"][0]
print(f"Sample label format: {sample_labels} (type: {type(sample_labels)})")

# Handle different label formats
def normalize_labels(labels):
    """Ensure labels are in list format"""
    if isinstance(labels, str):
        return [labels]
    elif isinstance(labels, list):
        return labels
    else:
        return [str(labels)]

# Normalize labels in datasets
train_labels = [normalize_labels(label) for label in train_ds["labels"]]
val_labels = [normalize_labels(label) for label in val_ds["labels"]]

mlb = MultiLabelBinarizer(classes=LABELS)
y_train = mlb.fit_transform(train_labels)
y_val = mlb.transform(val_labels)

print(f"Label shape: {y_train.shape}")
print(f"Label distribution:\n{y_train.sum(axis=0)}")

# Check for missing labels
if y_train.sum() == 0 or y_val.sum() == 0:
    warnings.warn("Warning: Some datasets have no positive labels")

# Replace label lists with multi-hot vectors
train_ds = train_ds.remove_columns(["labels"]).add_column(
    "labels", y_train.astype("float32").tolist()
)

val_ds = val_ds.remove_columns(["labels"]).add_column(
    "labels", y_val.astype("float32").tolist()
)

# =====================================================
# STEP 3: TOKENIZER & MODEL
# =====================================================
print("\nLoading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    problem_type="multi_label_classification"
).to(device)

# =====================================================
# STEP 4: TOKENIZATION
# =====================================================
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

print("\nTokenizing datasets...")
train_ds = train_ds.map(tokenize, batched=True, desc="Tokenizing train")
val_ds = val_ds.map(tokenize, batched=True, desc="Tokenizing validation")

train_ds.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "labels"]
)
val_ds.set_format(
    "torch",
    columns=["input_ids", "attention_mask", "labels"]
)

# =====================================================
# STEP 5: METRICS (MULTI-LABEL) - FIXED
# =====================================================
def compute_metrics(eval_pred):
    """Compute multi-label classification metrics"""
    logits, labels = eval_pred
    
    # Convert to numpy if needed
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Apply sigmoid and threshold
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs > 0.5).astype(int)
    
    # Compute metrics
    micro_f1 = f1_score(labels, preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    # Per-class F1
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    
    metrics = {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
    }
    
    # Add per-class metrics
    for i, label in enumerate(LABELS):
        metrics[f"f1_{label}"] = float(per_class_f1[i])
    
    return metrics

# =====================================================
# STEP 6: TRAINING ARGUMENTS
# =====================================================
# Handle different transformers versions
try:
    # Try newer version parameter name first
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=100,
        logging_dir=f"{OUTPUT_DIR}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=42
    )
except TypeError:
    # Fall back to older version parameter name
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=100,
        logging_dir=f"{OUTPUT_DIR}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=42,
        fp16=torch.cuda.is_available(),
    )

# =====================================================
# STEP 7: TRAINER
# =====================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# =====================================================
# STEP 8: TRAIN
# =====================================================
print("\nStarting training...")
trainer.train()

# =====================================================
# STEP 9: FINAL EVALUATION
# =====================================================
print("\nRunning final evaluation...")

preds = trainer.predict(val_ds)

logits = preds.predictions
true_labels = preds.label_ids

# Convert to numpy if needed
if isinstance(logits, torch.Tensor):
    logits = logits.cpu().numpy()
if isinstance(true_labels, torch.Tensor):
    true_labels = true_labels.cpu().numpy()

probs = 1 / (1 + np.exp(-logits))  # sigmoid
pred_labels = (probs > 0.5).astype(int)

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        pred_labels,
        target_names=LABELS,
        zero_division=0
    )
)

# =====================================================
# STEP 10: SAVE MODEL
# =====================================================
print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Training complete. Model saved to: {OUTPUT_DIR}")
print("\nTo use the model:")
print(f"  from transformers import AutoTokenizer, AutoModelForSequenceClassification")
print(f"  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")
print(f"  model = AutoModelForSequenceClassification.from_pretrained('{OUTPUT_DIR}')")