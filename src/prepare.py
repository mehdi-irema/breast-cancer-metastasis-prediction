"""Build the modelling table from the raw clinical export (SPSS .sav).

The raw data are confidential patient records and are NOT part of this
repository. This module only encodes the cleaning rules, so the analysis can be
re-run by anyone with authorised access to the data.

Design choice: only *row-wise, rule-based* fills happen here (clinical logic
that uses no other patient's data). Anything that needs statistics estimated
from the data (medians, modes) is left as NaN and imputed later inside the
training pipeline, so test-set information never leaks into training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyreadstat

# Direct identifiers and free-text/date fields: dropped immediately, never used.
IDENTIFIERS = ["name", "tel", "Name.Doctor", "birthday", "date", "Family"]

FEATURES = [
    # demographics and reproductive history
    "place", "age.diag", "age.menarc", "age.menop", "education", "marital",
    "gravity", "lactation", "fpregnancy", "infertility", "reproduct", "mens",
    "ocpuse", "ocp", "hrt", "hystrectomy", "smoking", "no.biopsy",
    "personal.Hx", "personal.other", "weight", "height", "cc", "lifeevent",
    "hypothyroidism",
    # tumour and treatment
    "LN", "SN", "surgery", "G", "ER", "Her2", "P53", "Ki67", "chemo", "hormon",
]

# Columns whose missing values are filled by a median/mode *inside* the model
# pipeline (fitted on training folds only).
NUMERIC_TO_IMPUTE = ["age.diag", "weight", "height", "Ki67", "G", "ocp", "lactation"]
CATEGORICAL_TO_IMPUTE = ["education"]


def load_raw(path: str) -> pd.DataFrame:
    df, _ = pyreadstat.read_sav(path)
    return df.drop(columns=[c for c in IDENTIFIERS if c in df.columns])


def build_target(df: pd.DataFrame) -> pd.Series:
    """1 if any recurrence / metastasis event was recorded during follow-up."""
    event = pd.Series(False, index=df.index)
    for col in ["L.recurrence", "p.recurrence", "site", "therapyrecurrence", "contralatral"]:
        if col in df.columns:
            event |= df[col].notna()
    if "event" in df.columns:
        event |= df["event"].isin([1, 2])
    return event.astype(int).rename("metastasis_or_recurrence")


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # age is only used for clinical rules; fall back to age at diagnosis
    age = d["age"].fillna(d["age.diag"])

    # constant / "absent unless recorded" defaults (clinical convention)
    defaults = {"place": 1, "age.menarc": 13, "marital": 1, "gravity": 0,
                "fpregnancy": 0, "infertility": 0, "mens": 1, "ocpuse": 0,
                "hrt": 0, "hystrectomy": 0, "smoking": 0, "no.biopsy": 0,
                "personal.Hx": 0, "personal.other": 0, "cc": 1, "lifeevent": 0,
                "hypothyroidism": 0, "LN": 0, "SN": 0, "hormon": 0}
    for col, val in defaults.items():
        d[col] = d[col].fillna(val)

    # menopause: missing -> assume post-menopausal from age 50
    d["age.menop"] = d["age.menop"].fillna(pd.Series(np.where(age >= 50, 50, 0), index=d.index))
    d["age.menop"] = (d["age.menop"] != 0).astype(int)

    # reproductive status by age band when missing
    d["reproduct"] = d["reproduct"].fillna(
        pd.Series(np.select([age >= 50, age > 40], [3, 2], 1), index=d.index))

    # lactation / OCP: structurally zero when the precondition is absent
    d.loc[d["lactation"].isna() & (d["gravity"] == 2), "lactation"] = 0
    d.loc[d["lactation"].isna() & (d["fpregnancy"] == 0), "lactation"] = 0
    d.loc[d["ocp"].isna() & (d["ocpuse"] == 0), "ocp"] = 0

    d["fpregnancy"] = (d["fpregnancy"] != 0).astype(int)
    d["no.biopsy"] = (d["no.biopsy"] != 0).astype(int)

    # surgery type -> breast-conserving (0) vs mastectomy (1)
    surg = d["surgery"]
    d["surgery"] = np.select([surg.isin([1, 2, 5, 9, 10, 11, 12, 13]),
                              surg.isin([3, 4, 8])], [0, 1], np.nan)
    d["surgery"] = d["surgery"].fillna(0)

    # hormone receptors: ER and PR are strongly concordant, fill one from the other;
    # remaining unknowns get an explicit "unknown" level
    d["ER"] = d["ER"].fillna(d["PR"]).fillna(2)
    d["Her2"] = d["Her2"].fillna(4)
    d["P53"] = d["P53"].fillna(2)

    # chemotherapy: codes 8/15 are "unknown"; anything else non-zero = received
    chemo = d["chemo"].where(~d["chemo"].isin([8, 15]))
    d["chemo"] = (chemo.fillna(1) != 0).astype(int)
    d["hormon"] = (d["hormon"] != 0).astype(int)

    # a few SPSS fields arrive as text; blanks become missing
    return d[FEATURES].apply(pd.to_numeric, errors="coerce")


def load_dataset(path: str) -> tuple[pd.DataFrame, pd.Series]:
    raw = load_raw(path)
    return clean_features(raw), build_target(raw)
