#!/usr/bin/env python3
"""
Category Consistency Checker
============================
Checks the text (categorical) columns of a dataset for inconsistent category values.

Problems detected:
    1. Format variants    - the same value written differently (case, spaces, punctuation)
                            e.g. "Tehran" / "tehran" / " TEHRAN "   |   "IT" / "I.T."
    2. Synonym conflicts  - different values with the same meaning inside one column
                            e.g. "male" + "M"   |   "yes" + "Y" + "1"   |   "بله" + "yes"
    3. Possible typos     - very similar spellings where one is probably a mistake
                            e.g. "Tehrna" vs "Tehran"   |   "Berlni" vs "Berlin"

The data is NOT modified. The output is a report plus a suggested mapping
(old value -> suggested value) that you can review and apply yourself.

Usage:
    python category_consistency_checker.py data.csv
    python category_consistency_checker.py data.xlsx --sheet Sheet1 --json report.json
    python category_consistency_checker.py data.csv --synonyms my_synonyms.json --similarity 0.8

Custom synonyms file (JSON), domain -> concept -> variants:
    {"department": {"hr": ["HR", "Human Resources", "Personnel"],
                    "it": ["IT", "I.T.", "Tech"]}}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Values meaning "missing". Compared AFTER normalization:
# "N/A" -> "na", "-" -> "", " " -> ""
NA_KEYS = {"", "na", "nan", "null", "none", "nil", "unknown", "missing", "undefined", "empty", "nd"}

# Built-in concept dictionary: domain -> concept -> variants.
# A column is matched against one domain (e.g. "gender") only if most of its values belong to it.
BUILTIN_DOMAINS = {
    "gender": {
        "male": ["male", "m", "man", "boy", "مرد"],
        "female": ["female", "f", "woman", "girl", "زن"],
        "other": ["other", "x", "nonbinary", "nb"],
    },
    "boolean": {
        "true": ["true", "yes", "y", "t", "1", "بله"],
        "false": ["false", "no", "n", "f", "0", "خیر", "نه"],
    },
    "status": {
        "active": ["active", "enabled", "open"],
        "inactive": ["inactive", "disabled", "closed"],
    },
}

PERSIAN_FIX = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه"})


def normalize(value) -> str:
    """Lowercase, unify Persian letters, and drop spaces/punctuation: ' Male.' -> 'male'."""
    text = str(value).translate(PERSIAN_FIX).lower()
    return re.sub(r"[\W_]+", "", text)


def _to_domain(concepts: dict) -> dict:
    """Convert {concept: [variants]} into {concept: {normalized variants}} (concept name included)."""
    return {
        concept: {normalize(concept)} | {normalize(v) for v in variants}
        for concept, variants in concepts.items()
    }


def _concept_index(domain: dict) -> dict:
    return {variant: concept for concept, variants in domain.items() for variant in variants}


def most_common_form(forms: dict) -> str:
    return max(forms, key=forms.get)


def _pick_readable(forms: dict) -> str:
    """Most frequent form, preferring words over pure numbers ('yes' over '1')."""
    candidates = [f for f in forms if not f.isdigit()] or list(forms)
    return max(candidates, key=forms.get)


def is_candidate_column(series: pd.Series) -> bool:
    """Skip empty columns and numeric columns (IDs, amounts, ages, ...)."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return False
    numeric_ratio = pd.to_numeric(values, errors="coerce").notna().mean()
    return numeric_ratio < 0.9


def analyze_column(series: pd.Series, similarity: float, domains: dict):
    # Original strings are kept (no strip) so " Tehran " is reported as a format variant.
    key_forms: dict = {}
    for raw, count in series.dropna().astype(str).value_counts().items():
        key = normalize(raw)
        if key in NA_KEYS:
            continue
        key_forms.setdefault(key, {})[raw] = int(count)

    if not key_forms:
        return None

    total = sum(sum(forms.values()) for forms in key_forms.values())
    n_keys = len(key_forms)
    # Many distinct values (names, addresses, free text) -> not a category column.
    if n_keys > 50 and n_keys / total > 0.2:
        return None

    issues: dict = {}
    mapping: dict = {}

    # Detect which domain (gender / boolean / status) this column belongs to
    keys = set(key_forms)
    domain_name, domain = None, None
    best_coverage = 0.0
    for dname, concepts in domains.items():
        matched = keys & set(_concept_index(concepts))
        coverage = len(matched) / n_keys
        if len(matched) >= 2 and coverage >= 0.6 and coverage > best_coverage:
            domain_name, domain, best_coverage = dname, concepts, coverage

    # 2) Synonym conflicts: two or more different values for the same concept
    covered_raw: set = set()
    if domain is not None:
        synonym_conflicts = []
        for concept, variants in domain.items():
            present = [k for k in keys if k in variants]
            if len(present) >= 2:
                all_forms: dict = {}
                for k in present:
                    all_forms.update(key_forms[k])
                canon = _pick_readable(all_forms)
                covered_raw.update(all_forms)
                synonym_conflicts.append({
                    "domain": domain_name,
                    "concept": concept,
                    "forms": all_forms,
                })
                for raw in all_forms:
                    if raw != canon:
                        mapping.setdefault(raw, canon)
        if synonym_conflicts:
            issues["synonym_conflicts"] = synonym_conflicts

    # 2b) Format variants (same value, different spelling) not already covered above
    format_variants = []
    for key, forms in key_forms.items():
        if len(forms) > 1 and not set(forms) <= covered_raw:
            canon = most_common_form(forms)
            format_variants.append({"normalized": key, "forms": forms})
            for raw in forms:
                if raw != canon:
                    mapping.setdefault(raw, canon)
    if format_variants:
        issues["format_variants"] = format_variants

    # 3) Possible typos: very similar normalized values
    concept_of = _concept_index(domain) if domain is not None else {}
    ranked = sorted(key_forms.items(), key=lambda kv: -sum(kv[1].values()))
    possible_typos = []
    for i, (k1, f1) in enumerate(ranked):
        if len(k1) < 4:
            continue
        for k2, f2 in ranked[i + 1:]:
            if len(k2) < 4:
                continue
            # Two known values of the same domain (e.g. "active" / "inactive") are not typos.
            if k1 in concept_of and k2 in concept_of:
                continue
            ratio = SequenceMatcher(None, k1, k2).ratio()
            if ratio >= similarity:
                common = most_common_form(f1)
                rare = most_common_form(f2)
                possible_typos.append({
                    "likely_typo": rare,
                    "suggested": common,
                    "similarity": round(ratio, 2),
                    "counts": {common: sum(f1.values()), rare: sum(f2.values())},
                })
                mapping.setdefault(rare, common)
    if possible_typos:
        issues["possible_typos"] = possible_typos

    if not issues:
        return None
    return {"domain": domain_name, "issues": issues, "suggested_mapping": mapping}


def check_dataframe(df: pd.DataFrame, similarity: float, domains: dict) -> dict:
    checked, skipped, results = [], [], {}
    for col in df.columns:
        name = str(col)
        if not is_candidate_column(df[col]):
            skipped.append(name)
            continue
        checked.append(name)
        result = analyze_column(df[col], similarity, domains)
        if result:
            results[name] = result
    return {"columns_checked": checked, "columns_skipped": skipped, "results": results}


def load_table(path: str, sheet=None) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p, sheet_name=sheet if sheet is not None else 0)
    if ext == ".json":
        return pd.read_json(p)
    if ext == ".parquet":
        return pd.read_parquet(p)

    sep = "\t" if ext == ".tsv" else None  # None = let pandas detect the separator
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp1256", "latin1"):
        try:
            return pd.read_csv(p, sep=sep, engine="python", encoding=encoding, dtype=str)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def build_rows(report: dict) -> list:
    rows = []
    for col, res in report["results"].items():
        mapping = res["suggested_mapping"]
        issues = res["issues"]

        for s in issues.get("synonym_conflicts", []):
            forms = ", ".join(f"'{f}' ({c})" for f, c in sorted(s["forms"].items(), key=lambda kv: -kv[1]))
            fix = ", ".join(f"'{f}' → '{mapping[f]}'" for f in s["forms"] if f in mapping)
            rows.append((col, "Synonym conflict", f"Concept '{s['concept']}' is written as: {forms}", fix))

        for fv in issues.get("format_variants", []):
            forms = ", ".join(f"'{f}' ({c})" for f, c in sorted(fv["forms"].items(), key=lambda kv: -kv[1]))
            fix = ", ".join(f"'{f}' → '{mapping[f]}'" for f in fv["forms"] if f in mapping)
            rows.append((col, "Format variant", forms, fix))

        for t in issues.get("possible_typos", []):
            rows.append((
                col, "Possible typo",
                f"'{t['likely_typo']}' vs '{t['suggested']}' (similarity {t['similarity']})",
                f"'{t['likely_typo']}' → '{t['suggested']}'",
            ))
    return rows


def print_report(report: dict, rows: list, file_name: str, n_rows: int) -> None:
    summary = (
        f"{len(report['columns_checked'])} text column(s) checked, "
        f"{len(report['columns_skipped'])} skipped (numeric or empty), "
        f"{len(rows)} issue(s) in {len(report['results'])} column(s)."
    )

    if RICH_AVAILABLE:
        console = Console()
        console.print(f"[bold cyan]Category Consistency Check[/bold cyan]  {file_name}  ({n_rows:,} rows)")
        if rows:
            table = Table(box=box.SIMPLE_HEAVY, show_lines=True)
            for name in ("Column", "Issue", "Details", "Suggested fix"):
                table.add_column(name, overflow="fold")
            for row in rows:
                table.add_row(*[Text(str(x)) for x in row])
            console.print(table)
        else:
            console.print("[green]No category inconsistencies found.[/green]")
        console.print(summary)
        return

    print(f"Category Consistency Check: {file_name} ({n_rows:,} rows)")
    if rows:
        for col, issue, details, fix in rows:
            print(f"- [{col}] {issue}: {details}")
            if fix:
                print(f"    fix: {fix}")
    else:
        print("No category inconsistencies found.")
    print(summary)


def main() -> int:
    ap = argparse.ArgumentParser(description="Check text columns for inconsistent category values.")
    ap.add_argument("file", help="CSV / TSV / Excel / JSON / Parquet file")
    ap.add_argument("--sheet", default=None, help="Excel sheet name or index")
    ap.add_argument("--synonyms", default=None, help="JSON file with extra domains/concepts")
    ap.add_argument("--similarity", type=float, default=0.8,
                    help="Similarity threshold for possible typos (0-1, default 0.8)")
    ap.add_argument("--json", dest="json_out", default=None, help="Save the full report as JSON")
    args = ap.parse_args()

    sheet = int(args.sheet) if args.sheet and args.sheet.isdigit() else args.sheet
    try:
        df = load_table(args.file, sheet)
    except Exception as exc:
        print(f"Error: could not load the file: {exc}", file=sys.stderr)
        return 1

    domains = {name: _to_domain(concepts) for name, concepts in BUILTIN_DOMAINS.items()}
    if args.synonyms:
        extra = json.loads(Path(args.synonyms).read_text(encoding="utf-8"))
        for name, concepts in extra.items():
            domains.setdefault(name, {}).update(_to_domain(concepts))

    report = check_dataframe(df, args.similarity, domains)
    rows = build_rows(report)
    print_report(report, rows, Path(args.file).name, len(df))

    if args.json_out:
        out = {
            "file": Path(args.file).name,
            "rows": len(df),
            "columns_checked": report["columns_checked"],
            "columns_skipped": report["columns_skipped"],
            "results": report["results"],
        }
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON report saved to: {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
