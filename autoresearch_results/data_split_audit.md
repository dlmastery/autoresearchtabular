# AUTORESEARCHTABULAR — Data Split Audit Report

_Generated: 2026-04-26T14:50:28Z_

- **Overall:** **PASS**
- **Data mode:** higgs (Baldi 2014 frozen split)
- **subset_train_n:** None
- **Audit version:** 1.0.0
- **Split fingerprint:** `eee12999eeae3950c0c27295dd117286ff8a24999ba466a2d8b6fd8a0dda115c`

## Fold sizes

| fold | size |
|---|---|
| train | 10,000,000 |
| val | 500,000 |
| test | 500,000 |

## Auditors

### audit_split_disjoint — **PASS**
- (no violations)

<details><summary>details</summary>

```json
{
  "pairwise_intersections": {
    "train & val": 0,
    "train & test": 0,
    "val & test": 0
  }
}
```
</details>

### audit_split_protocol — **PASS**
- (no violations)

<details><summary>details</summary>

```json
{
  "train_n": 10000000,
  "train_expected": 10000000,
  "val_n": 500000,
  "val_expected": 500000,
  "test_n": 500000,
  "test_expected": 500000
}
```
</details>

### audit_class_balance — **PASS**
- (no violations)

<details><summary>details</summary>

```json
{
  "train_classes": [
    0,
    1
  ],
  "train_pos_prevalence": 0.53,
  "val_classes": [
    0,
    1
  ],
  "val_pos_prevalence": 0.5302,
  "test_classes": [
    0,
    1
  ],
  "test_pos_prevalence": 0.529
}
```
</details>

### audit_size_floors — **PASS**
- (no violations)

<details><summary>details</summary>

```json
{
  "train": 10000000,
  "val": 500000,
  "test": 500000
}
```
</details>

### audit_no_leakage_via_metadata — **PASS**
- (no violations)

<details><summary>details</summary>

```json
{
  "train_shape": [
    10000000,
    28
  ],
  "train_dtype": "float32",
  "val_shape": [
    500000,
    28
  ],
  "val_dtype": "float32",
  "test_shape": [
    500000,
    28
  ],
  "test_dtype": "float32"
}
```
</details>

### audit_feature_consistency — **PASS**
- (no violations)

<details><summary>details</summary>

```json
{
  "train_nan": 0,
  "train_inf": 0,
  "val_nan": 0,
  "val_inf": 0,
  "test_nan": 0,
  "test_inf": 0
}
```
</details>

### audit_reproducibility — **PASS**
- (no violations)

<details><summary>details</summary>

```json
{
  "fingerprints": [
    "eee12999eeae3950c0c27295dd117286ff8a24999ba466a2d8b6fd8a0dda115c",
    "eee12999eeae3950c0c27295dd117286ff8a24999ba466a2d8b6fd8a0dda115c"
  ],
  "sizes": [
    {
      "train": 10000000,
      "val": 500000,
      "test": 500000
    },
    {
      "train": 10000000,
      "val": 500000,
      "test": 500000
    }
  ]
}
```
</details>
