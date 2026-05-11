Here is a whole repo for all cool ML shenanighans and any cool mindblowing things that I will be dicovering

# :-0

# Deep Network Development Importants
## NumPy + PyTorch + Binary Classification Survival Guide

---

# 1. FIRST QUESTION:
# Is this Regression or Classification?

This is the MOST important first step in ML exams.

---

# REGRESSION

Use regression when predicting continuous values.

Examples:
- house price
- salary
- age
- temperature
- ratings/scores

Example outputs:

```python
7.5
102.3
250000
```

### Common Loss Functions

```python
nn.MSELoss()
nn.L1Loss()
```

### Typical Output Layer

```python
nn.Linear(hidden_size, 1)
```

### Important:
- No sigmoid
- Predicts real numbers

---

# CLASSIFICATION

Use classification when predicting categories/classes.

Examples:
- spam / not spam
- cat / dog
- fraud / not fraud
- premium / budget

---

# Binary Classification

Only TWO classes.

Examples:

```python
0 or 1
yes or no
premium or budget
```

### Common Loss Functions

```python
nn.BCELoss()
nn.BCEWithLogitsLoss()
```

### Output Layer

```python
nn.Linear(hidden_size, 1)
```

### Convert logits to probabilities

```python
prob = torch.sigmoid(output)
```

---

# Multi-class Classification

More than 2 classes.

Examples:
- digits 0-9
- animal species
- sentiment classes

### Common Loss

```python
nn.CrossEntropyLoss()
```

---

# QUICK DECISION TABLE

| Target Type | Problem Type | Loss |
|---|---|---|
| Continuous values | Regression | MSELoss |
| 0/1 labels | Binary Classification | BCEWithLogitsLoss |
| Integer classes 0,1,2... | Multi-class Classification | CrossEntropyLoss |

---

# 2. THE ORANGE JUICE EXAM IS:
# Binary Classification

Why?

Because:
- Budget → label 0
- Premium → label 1

The network predicts:

> Probability that a juice is Premium

So:
- output = probability
- labels = 0 or 1

This is NOT regression.

---

# 3. WHAT MODEL TYPE SHOULD YOU USE?

The professor explicitly asks for:

> Multi-layer Perceptron (MLP)

Meaning:
- Neural Network
- PyTorch
- Fully connected layers

NOT:
- Linear Regression
- Polynomial Regression
- Ridge/Lasso
- SVM
- KNN
- Random Forest

---

# 4. WHAT THE EXAM ACTUALLY TESTS

The exam is mostly testing:

- NumPy arrays
- shapes
- vectorized operations
- boolean masks
- PyTorch pipeline
- dataloaders
- training loops
- debugging

NOT advanced ML theory.

---

# 5. NUMPY ESSENTIALS

---

# Array Shape

```python
arr.shape
```

Example:

```python
(200, 14)
```

means:
- 200 rows
- 14 columns

---

# Select Columns

## First column

```python
arr[:, 0]
```

---

## Last 3 columns

```python
arr[:, -3:]
```

VERY IMPORTANT because scores are in the last 3 columns.

---

## All except last 3 columns

```python
arr[:, :-3]
```

VERY IMPORTANT.

---

# Concatenate Arrays

## Stack rows

```python
np.concatenate([a, b], axis=0)
```

axis=0 → rows

axis=1 → columns

---

# Mean and Min

## Row-wise mean

```python
arr.mean(axis=1)
```

---

## Row-wise minimum

```python
arr.min(axis=1)
```

---

# Boolean Masks

VERY IMPORTANT.

```python
mask = arr[:, 0] > 5
```

Returns:

```python
[True, False, True]
```

Use mask:

```python
filtered = arr[mask]
```

Inverse mask:

```python
filtered = arr[~mask]
```

---

# Reordering Columns

IMPORTANT EXAM PATTERN.

```python
indices = [ATTR2.index(name) for name in ATTR1]

dataset2_reordered = dataset2[:, indices]
```

This reorders dataset2 columns to match dataset1.

MEMORIZE THIS PATTERN.

---

# Shuffling

```python
perm = np.random.permutation(len(dataset))

dataset = dataset[perm]
```

---

# Train/Validation/Test Split

```python
n = len(dataset)

train_end = int(0.7 * n)
val_end = int(0.85 * n)

train = dataset[:train_end]
val = dataset[train_end:val_end]
test = dataset[val_end:]
```

---

# 6. HOW TO SOLVE TASK C
# (Category Creation)

---

# Step 1:
Get scores

```python
scores = dataset[:, -3:]
```

---

# Step 2:
Compute average scores

```python
avg_scores = scores.mean(axis=1)
```

---

# Step 3:
Compute minimum scores

```python
min_scores = scores.min(axis=1)
```

---

# Step 4:
Create premium mask

Premium rules:
- average > 5
- minimum >= 3

```python
premium_mask = (avg_scores > 5) & (min_scores >= 3)
```

---

# Step 5:
Remove score columns

```python
features = dataset[:, :-3]
```

---

# Step 6:
Split categories

```python
dataset_premium = features[premium_mask]

dataset_budget = features[~premium_mask]
```

That is basically the whole task.

---

# 7. PYTORCH ESSENTIALS

---

# Tensor Conversion

```python
torch.tensor(arr, dtype=torch.float32)
```

---

# TensorDataset + DataLoader

```python
from torch.utils.data import TensorDataset, DataLoader
```

---

# Create Dataset

```python
dataset = TensorDataset(X_tensor, y_tensor)
```

---

# Create DataLoader

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

---

# Expected Shapes

Inputs:

```python
(batch_size, 11)
```

Labels:

```python
(batch_size, 1)
```

---

# IMPORTANT:
Labels must be float32 for BCE loss.

---

# 8. SIMPLE MLP TEMPLATE

```python
import torch.nn as nn

class BinClModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)
```

---

# Best Loss Function

```python
criterion = nn.BCEWithLogitsLoss()
```

IMPORTANT:
If using BCEWithLogitsLoss:
- DO NOT add sigmoid in model

---

# Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

---

# 9. TRAINING LOOP TEMPLATE

```python
for epoch in range(num_epochs):

    model.train()

    for x_batch, y_batch in train_loader:

        optimizer.zero_grad()

        outputs = model(x_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()
```

---

# Validation Loop

```python
model.eval()

with torch.no_grad():

    for x_batch, y_batch in val_loader:

        outputs = model(x_batch)

        loss = criterion(outputs, y_batch)
```

---

# 10. BINARY ACCURACY

---

# Convert logits to probabilities

```python
probs = torch.sigmoid(outputs)
```

---

# Convert probabilities to predictions

```python
preds = (probs >= 0.5).float()
```

---

# Accuracy

```python
correct = (preds == labels).sum()

accuracy = correct / len(labels)
```

---

# 11. EARLY STOPPING

Purpose:
Stop training when validation loss stops improving.

---

# Basic Logic

```python
best_val_loss = float('inf')

patience_counter = 0
```

---

# If validation improves

```python
best_val_loss = val_loss

best_weights = model.state_dict()

patience_counter = 0
```

---

# Otherwise

```python
patience_counter += 1
```

---

# Stop condition

```python
if patience_counter >= patience:
    break
```

---

# Restore best weights

```python
model.load_state_dict(best_weights)
```

---

# 12. IMPORTANT DEBUGGING STRATEGY

PRINT EVERYTHING.

---

# Check Shapes

```python
print(arr.shape)
```

---

# Check First Rows

```python
print(arr[:5])
```

---

# Check Batch Shapes

```python
x, y = next(iter(loader))

print(x.shape)
print(y.shape)
```

---

# Check Labels

```python
print(torch.unique(labels))
```

---

# 13. COMMON SHAPE ERRORS

---

# WRONG

```python
(batch_size,)
```

---

# CORRECT

```python
(batch_size, 1)
```

Fix:

```python
labels = labels.unsqueeze(1)
```

---

# 14. LEAST CONFIDENT PREDICTIONS

Find probabilities closest to 0.5.

---

# Distance from 0.5

```python
distances = np.abs(preds - 0.5)
```

---

# Get 5 smallest

```python
indices = np.argsort(distances)[:5]
```

---

# 15. HISTOGRAMS

Example:

```python
plt.hist(premium_probs)

plt.hist(budget_probs)

plt.legend()

plt.show()
```

---

# 16. MOST IMPORTANT EXAM STRATEGY

---

# PRIORITY ORDER

## Step 1
Get arrays correct.

Check:
- shapes
- dtypes
- slices

---

## Step 2
Get labels correct.

Check:
- 0/1
- sizes

---

## Step 3
Get dataloader working.

Print one batch:

```python
x, y = next(iter(loader))

print(x.shape)
print(y.shape)
```

---

## Step 4
Get training loop running.

Even imperfect working code is better than elegant broken code.

---

## Step 5
Optimize later.

---

# 17. MOST IMPORTANT FUNCTIONS TO MEMORIZE

---

# NumPy

```python
np.concatenate()
np.random.permutation()
np.mean()
np.min()
```

---

# PyTorch

```python
TensorDataset
DataLoader

nn.Linear
nn.ReLU

torch.sigmoid

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

# 18. FINAL SURVIVAL RULES

---

## ALWAYS PRINT SHAPES

```python
print(x.shape)
```

---

## ALWAYS CHECK LABELS

```python
print(labels[:10])
```

---

## BCEWithLogitsLoss = NO SIGMOID IN MODEL

---

## Binary Classification:
- labels = 0/1
- sigmoid
- BCE loss

---

## Regression:
- continuous values
- MSE loss
- no sigmoid

---

# 19. FINAL MINDSET

Get a WORKING pipeline first.

The pipeline is:

```text
Data
→ Labels
→ DataLoader
→ Model
→ Loss
→ Train
→ Evaluate
```

If all those connect correctly,
you are already close to passing.
