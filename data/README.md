# Expected data inputs

Participant-level data are deliberately excluded from this public repository.

The Kaggle notebook expects the following authorised files to be available through a private Kaggle Dataset:

| File | Role |
|---|---|
| `Development_corrected_with_Albumin.xlsx` | PoCOsteo development cohort |
| `External_with_Albumin.xlsx` | BEH Phase III external-validation cohort |

The notebook performs cohort-specific cleaning and maps source columns to a common harmonised schema. Identifiers are used only to verify that the analytic cohorts do not overlap and are not included among model inputs or exported analytic outputs.

Do not commit raw data, identifiers, fitted participant-level artifacts, API keys, or Kaggle credential files to this repository.

