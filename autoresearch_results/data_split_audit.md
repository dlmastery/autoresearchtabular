# AUTORESEARCHTABULAR — Data Split Audit Report

_Generated: 2026-04-26T06:40:26Z_

- **Overall:** **PASS**
- **Data mode:** higgs (Baldi 2014 frozen split)
- **subset_train_n:** 1000000
- **Audit version:** 1.0.0
- **Split fingerprint:** `3c5edcc34086b3dba8406b7e5f4ede15e3549c1d6f5c190f24b8e34e780a0117`

## Fold sizes

| fold | size |
|---|---|
| train | 1,000,000 |
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
  "train_n": 1000000,
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
  "train_pos_prevalence": 0.5297,
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
  "train": 1000000,
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
    1000000,
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
    "3c5edcc34086b3dba8406b7e5f4ede15e3549c1d6f5c190f24b8e34e780a0117",
    "3c5edcc34086b3dba8406b7e5f4ede15e3549c1d6f5c190f24b8e34e780a0117"
  ],
  "sizes": [
    {
      "train": 1000000,
      "val": 500000,
      "test": 500000
    },
    {
      "train": 1000000,
      "val": 500000,
      "test": 500000
    }
  ]
}
```
</details>
