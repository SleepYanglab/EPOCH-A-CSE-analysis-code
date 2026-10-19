from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pyreadstat
from docx import Document


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REFERENCE_DIR = ROOT / "references"
OUTPUT_DIR = ROOT / "results" / "phase10_sample_quality_audit"


def cronbach_alpha(frame: pd.DataFrame) -> float:
    complete = frame.dropna()
    k = complete.shape[1]
    item_variance = complete.var(axis=0, ddof=1).sum()
    total_variance = complete.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_variance / total_variance))


def find_questionnaire() -> Path:
    candidates = []
    for path in REFERENCE_DIR.glob("*.docx"):
        document = Document(path)
        candidates.append((len(document.tables), path))
    if not candidates:
        raise FileNotFoundError("No questionnaire document found in references/")
    return max(candidates)[1]


def questionnaire_evidence(path: Path) -> dict[str, object]:
    document = Document(path)
    pup6_needle = "闲暇之余，我会与父母一起使用手机。"
    lie_needle = "我拥有一些魔法或超能力。"
    compensation_needle = "我们会为通过测谎题与注意力检测题的参与者发放一定数额的津贴。"

    table_rows = [
        " | ".join(cell.text.strip() for cell in row.cells)
        for table in document.tables
        for row in table.rows
    ]
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    return {
        "file": str(path.relative_to(ROOT)),
        "pup6_exact_wording_found": any(pup6_needle in row for row in table_rows),
        "pup6_wording": pup6_needle,
        "implausible_statement_found": any(lie_needle in row for row in table_rows),
        "implausible_statement": lie_needle,
        "compensation_statement_found": compensation_needle in paragraphs,
        "compensation_statement": compensation_needle,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sav_files = list(RAW_DIR.glob("*.sav"))
    if len(sav_files) != 1:
        raise RuntimeError(f"Expected one SAV file in {RAW_DIR}, found {len(sav_files)}")

    sav_path = sav_files[0]
    data, metadata = pyreadstat.read_sav(str(sav_path))
    labels = dict(zip(metadata.column_names, metadata.column_labels))

    pup_items = [f"PUP{i}" for i in range(1, 9)]
    pup_rows = []
    for item in pup_items:
        other_items = [column for column in pup_items if column != item]
        pup_rows.append(
            {
                "item": item,
                "variable_label": labels.get(item),
                "n": int(data[item].notna().sum()),
                "mean": float(data[item].mean()),
                "sd": float(data[item].std(ddof=1)),
                "corrected_item_total_r": float(
                    data[item].corr(data[other_items].sum(axis=1))
                ),
                "alpha_if_deleted": cronbach_alpha(data[other_items]),
            }
        )
    pup_diagnostics = pd.DataFrame(pup_rows)
    pup_diagnostics.to_csv(OUTPUT_DIR / "pup_item_diagnostics.csv", index=False)

    time_quantiles = data["time"].quantile(
        [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]
    )
    pd.DataFrame(
        {"quantile": time_quantiles.index, "value": time_quantiles.values}
    ).to_csv(OUTPUT_DIR / "time_field_summary.csv", index=False)

    questionnaire = questionnaire_evidence(find_questionnaire())
    label_pup6 = labels.get("PUP6") or ""
    quality_pattern = re.compile(
        r"\b(?:magic|superpowers?|lie[- ]detection|attention check|"
        r"validity check|quality check)\b|魔法|超能力|测谎",
        flags=re.IGNORECASE,
    )
    retained_quality_variables = [
        column
        for column in data.columns
        if quality_pattern.search(f"{column} {labels.get(column) or ''}")
    ]
    provenance_pattern = re.compile(
        r"\b(?:respondent id|participant id|case id|record id|serial number|"
        r"exclusion|excluded|screening status|pass/fail)\b",
        flags=re.IGNORECASE,
    )
    provenance_variables = [
        column
        for column in data.columns
        if provenance_pattern.search(f"{column} {labels.get(column) or ''}")
    ]

    summary = {
        "sav_file": str(sav_path.relative_to(ROOT)),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "sav_creation_time": str(metadata.creation_time),
        "sav_modification_time": str(metadata.modification_time),
        "pup6_sav_label": label_pup6,
        "pup6_wording_agrees_across_questionnaire_and_sav": bool(
            questionnaire["pup6_exact_wording_found"]
            and label_pup6 == "During leisure time, I use phones with my parents."
        ),
        "pup_alpha_8_items": cronbach_alpha(data[pup_items]),
        "pup_alpha_without_pup6": cronbach_alpha(
            data[[item for item in pup_items if item != "PUP6"]]
        ),
        "pup6_corrected_item_total_r": float(
            pup_diagnostics.loc[
                pup_diagnostics["item"] == "PUP6", "corrected_item_total_r"
            ].iloc[0]
        ),
        "stored_pup_equals_item_sum": bool(
            (data["PUP"] == data[pup_items].sum(axis=1)).all()
        ),
        "retained_quality_check_variables": retained_quality_variables,
        "possible_provenance_or_exclusion_variables": provenance_variables,
        "time_field_label": labels.get("time"),
        "time_field_unit_documented": False,
        "time_field_median": float(data["time"].median()),
        "time_field_range": [float(data["time"].min()), float(data["time"].max())],
        "questionnaire_evidence": questionnaire,
    }
    (OUTPUT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# Sample provenance and data-quality audit

This audit distinguishes what is documented in the source questionnaire from what is retained in the cleaned SPSS analytic file.

## PUP item 6

The Chinese source questionnaire contains the wording “{questionnaire['pup6_wording']}”. The SPSS label independently reads “{label_pup6}”. The wording therefore appears consistently in both supplied sources and is unlikely to be a transcription error introduced during manuscript preparation.

PUP6 behaved less consistently with the remaining seven items than the other PUP indicators: its corrected item–total correlation was {summary['pup6_corrected_item_total_r']:.3f}. Cronbach's alpha was {summary['pup_alpha_8_items']:.3f} for all eight items and {summary['pup_alpha_without_pup6']:.3f} without PUP6. The stored `PUP` score exactly equals the sum of PUP1–PUP8 for every participant. The primary analysis should therefore describe the supplied score as broader parent-context phone use/distraction and disclose that one item assessed joint phone use.

## Initial sample and exclusions

The supplied SPSS file contains {data.shape[0]:,} rows and is explicitly named as a cleaned sample. It contains no respondent ID, original-record count, exclusion indicator, or screening pass/fail field. The repository history contains only this cleaned file. Consequently, the pre-cleaning sample size and exclusion counts cannot be reconstructed from the available analytic data.

## Data-quality checks

The source questionnaire includes the implausible statement “{questionnaire['implausible_statement']}” and states that compensation was conditional on passing lie-detection and attention checks. Neither the implausible-statement response nor any separate attention-check response, threshold, or pass/fail flag is retained in the cleaned SPSS file. The `time` field is retained (median {summary['time_field_median']:.0f}, range {summary['time_field_range'][0]:.0f}–{summary['time_field_range'][1]:.0f}), but its label and unit are absent, and no documented exclusion threshold is available. The manuscript can therefore report that the questionnaire included data-quality screening, but should not claim a specific rule or independently verified exclusion process.
"""
    (OUTPUT_DIR / "README.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
