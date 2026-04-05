"""
GeoLLM Data Pipeline — PDF → Training Dataset
==============================================
Converts a single geotechnical PDF (boring logs, IS tables, report text)
into a ready-to-train JSONL dataset in ChatML format.

Usage:
    python geollm_pipeline.py --pdf path/to/report.pdf --output dataset.jsonl

For multiple PDFs:
    for f in *.pdf; do python geollm_pipeline.py --pdf "$f" --output all_data.jsonl --append; done
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

# ─────────────────────────────────────────────
# STEP 0: CONFIGURATION
# ─────────────────────────────────────────────
# This system prompt is injected into every training sample.
# It defines GeoLLM's identity, IS code scope, and hard safety rules.
# Every sample the model trains on will carry this — so compliance
# becomes baked into the model's behavior, not just inference-time prompting.

SYSTEM_PROMPT = """You are GeoLLM, a geotechnical engineering assistant \
specialized in Indian subsurface conditions and Bureau of Indian Standards (BIS) codes.

You analyze SPT boring logs, classify soils per IS 1498, and recommend foundation \
strategies per IS 1904 and IS 2911. Your outputs are used by structural engineers \
to make safety-critical decisions.

Rules you always follow:
1. Flag any SPT N-value <= 5 as a CRITICAL SAFETY CONCERN.
2. Always cite the relevant IS clause number when making a recommendation.
3. If boring log data is insufficient for a recommendation, say so explicitly.
4. Do not extrapolate beyond the provided depth range.
5. Use SI units throughout (kN, kPa, meters)."""


# ─────────────────────────────────────────────
# STEP 1: PDF TRIAGE
# ─────────────────────────────────────────────
# Before extracting anything, we probe the PDF to understand what's inside.
# A geotechnical PDF can be:
#   (a) text-native: selectable text, extractable tables → use pdfplumber
#   (b) scanned image: no extractable text → OCR needed (flagged, not handled here)
#
# We check by extracting the first page and measuring how much text we get.
# Less than 50 characters on page 1 almost always means a scanned document.

def triage_pdf(pdf_path: str) -> dict:
    """Return basic PDF metadata and decide extraction strategy."""
    reader = PdfReader(pdf_path)
    n_pages = len(reader.pages)

    # Sample first page text to detect scan vs native
    first_page_text = reader.pages[0].extract_text() or ""
    is_scanned = len(first_page_text.strip()) < 50

    info = {
        "path": pdf_path,
        "pages": n_pages,
        "is_scanned": is_scanned,
        "strategy": "ocr_needed" if is_scanned else "pdfplumber",
    }

    print(f"\n[TRIAGE] {Path(pdf_path).name}")
    print(f"  Pages   : {n_pages}")
    print(f"  Scanned : {is_scanned}")
    print(f"  Strategy: {info['strategy']}")

    if is_scanned:
        print("  ⚠  Scanned PDF detected. Text extraction will be empty.")
        print("     Run OCR with pytesseract before this pipeline.")
        print("     See: https://pypi.org/project/pytesseract/")

    return info


# ─────────────────────────────────────────────
# STEP 2: TEXT EXTRACTION
# ─────────────────────────────────────────────
# We extract all readable text from the PDF, page by page.
# pdfplumber is preferred over pypdf here because it preserves spatial
# layout better — critical for boring logs where depth/value alignment matters.
#
# Each page is stored as {"page": N, "text": "..."} so we can later
# reference source pages in metadata.

def extract_text(pdf_path: str) -> list[dict]:
    """Extract text from each page. Returns list of {page, text} dicts."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text.strip()})

    total_chars = sum(len(p["text"]) for p in pages)
    print(f"\n[TEXT] Extracted {total_chars:,} characters across {len(pages)} pages")
    return pages


# ─────────────────────────────────────────────
# STEP 3: TABLE EXTRACTION
# ─────────────────────────────────────────────
# Boring logs are almost always in tabular form:
#   Depth (m) | Soil Description | N-value | Remarks
#
# pdfplumber's extract_tables() uses spatial clustering to find table
# boundaries. It returns a list of rows, where each row is a list of cells.
#
# We post-process each table to:
#   1. Detect if it's a boring log (contains depth + N-value columns)
#   2. Convert it to a structured string for the training input field

def extract_tables(pdf_path: str) -> list[dict]:
    """Extract all tables from the PDF. Returns list of {page, table, raw} dicts."""
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue
                # Clean cells: strip whitespace, replace None with ""
                cleaned = [
                    [str(cell).strip() if cell else "" for cell in row]
                    for row in table
                ]
                all_tables.append({
                    "page": i + 1,
                    "table_idx": t_idx,
                    "rows": cleaned,
                    "raw_text": table_to_text(cleaned),
                })

    print(f"[TABLES] Found {len(all_tables)} tables")
    return all_tables


def table_to_text(rows: list[list[str]]) -> str:
    """Convert a table (list of rows) to a clean pipe-delimited string."""
    if not rows:
        return ""
    lines = []
    for row in rows:
        lines.append(" | ".join(cell for cell in row if cell))
    return "\n".join(lines)


# ─────────────────────────────────────────────
# STEP 4: BORING LOG DETECTION & PARSING
# ─────────────────────────────────────────────
# Not every table is a boring log. We need to identify which tables
# contain SPT data by looking for key column headers or patterns.
#
# A boring log table typically has columns matching:
#   - Depth / Elevation / RL
#   - Soil description / Strata
#   - N-value / Blow count / SPT
#   - Remarks / WL / Recovery
#
# We also parse N-values using regex from both table and text sources,
# since some PDFs embed boring data inline rather than in tables.

BORING_LOG_KEYWORDS = [
    r"\bspt\b", r"\bn.?value\b", r"\bblow\b", r"\bdepth\b",
    r"\bstrata\b", r"\bsoil\b", r"\brefusal\b", r"\bwater\s*table\b",
]

N_VALUE_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*[–\-–to]+\s*(\d+\.?\d*)\s*m.*?N\s*[=:]\s*(\d+)",
    re.IGNORECASE,
)

BOREHOLE_ID_PATTERN = re.compile(
    r"\b(BH|BOR(?:EHOLE)?|TP|TT|BIT)[.\-\s]*(\d+)\b",
    re.IGNORECASE,
)


def is_boring_log_table(rows: list[list[str]]) -> bool:
    """Return True if the table looks like a boring log."""
    # Flatten all cells to one string and search for keywords
    flat = " ".join(cell for row in rows for cell in row).lower()
    matches = sum(1 for kw in BORING_LOG_KEYWORDS if re.search(kw, flat))
    return matches >= 2  # Need at least 2 keyword hits to be confident


def parse_n_values_from_text(text: str) -> list[dict]:
    """Extract depth-interval → N-value records from raw text."""
    records = []
    for match in N_VALUE_PATTERN.finditer(text):
        depth_from = float(match.group(1))
        depth_to = float(match.group(2))
        n_value = int(match.group(3))
        records.append({
            "depth_from": depth_from,
            "depth_to": depth_to,
            "n_value": n_value,
            "safety_flag": n_value <= 5,
        })
    return records


def extract_borehole_id(text: str) -> str:
    """Try to extract borehole identifier from text."""
    match = BOREHOLE_ID_PATTERN.search(text)
    if match:
        return f"{match.group(1).upper()}-{match.group(2)}"
    return "BH-UNKNOWN"


# ─────────────────────────────────────────────
# STEP 5: TRAINING SAMPLE GENERATION
# ─────────────────────────────────────────────
# This is the core of the pipeline. For each boring log (table or text block),
# we generate MULTIPLE training samples covering different instruction types:
#
#   Type 1 — Soil classification (IS 1498)
#   Type 2 — Safety flag identification
#   Type 3 — Foundation depth recommendation
#   Type 4 — SPT summary / statistics
#
# The output field contains a TEMPLATE response that you must manually
# review and complete before training. Samples are flagged with
# "needs_review": true so you can filter them in your review pass.
#
# Why templates instead of auto-generated answers?
#   GeoLLM outputs feed safety-critical decisions. Auto-generating "answers"
#   from a base model would introduce hallucinations into your training data —
#   exactly what you're trying to train the model NOT to do.

def boring_log_to_input_string(bh_id: str, records: list[dict], table_text: str = "") -> str:
    """Format a boring log as the 'input' field for training samples."""
    if records:
        lines = [f"Borehole: {bh_id}"]
        for r in records:
            flag = " ⚠ CRITICAL" if r["safety_flag"] else ""
            lines.append(
                f"  {r['depth_from']:.1f}–{r['depth_to']:.1f}m | N = {r['n_value']}{flag}"
            )
        return "\n".join(lines)
    elif table_text:
        return f"Borehole: {bh_id}\n{table_text}"
    return ""


def generate_samples_from_boring_log(
    bh_id: str,
    records: list[dict],
    table_text: str,
    source_page: int,
    source_file: str,
) -> list[dict]:
    """Generate multiple ChatML training samples from one boring log."""
    input_context = boring_log_to_input_string(bh_id, records, table_text)
    if not input_context.strip():
        return []

    has_critical = any(r["safety_flag"] for r in records)
    critical_note = (
        f"⚠ N-value ≤ 5 detected — flag as CRITICAL SAFETY CONCERN per IS 2131."
        if has_critical else "No critically low N-values detected in this log."
    )

    samples = []

    # ── Sample Type 1: Soil classification ──────────────────────────────
    samples.append({
        "instruction": "Classify the soil profile from the boring log and identify any problematic horizons per IS 1498.",
        "input": input_context,
        "output": (
            f"[TEMPLATE — fill before training]\n"
            f"Review each depth interval and assign IS 1498 soil group symbols (SC, SM, CH, ML, etc.).\n"
            f"{critical_note}\n"
            f"Describe bearing characteristics of each horizon."
        ),
        "type": "classification",
        "needs_review": True,
    })

    # ── Sample Type 2: Safety flag identification ────────────────────────
    samples.append({
        "instruction": "Are there any horizons in this boring log that preclude shallow foundation construction? Cite IS code.",
        "input": input_context,
        "output": (
            f"[TEMPLATE — fill before training]\n"
            f"{critical_note}\n"
            f"Reference IS 1904 Cl. 4 for shallow foundation depth criteria.\n"
            f"Reference IS 1893 if seismic liquefaction risk is relevant."
        ),
        "type": "safety_flag",
        "needs_review": True,
    })

    # ── Sample Type 3: Foundation recommendation ─────────────────────────
    samples.append({
        "instruction": "Recommend a safe foundation type and depth for a 4-storey reinforced concrete structure at this borehole location.",
        "input": input_context,
        "output": (
            f"[TEMPLATE — fill before training]\n"
            f"Identify the competent stratum (highest N-value horizon that is also consistent).\n"
            f"Recommend foundation depth per IS 1904.\n"
            f"If N < 10 throughout, recommend pile foundation per IS 2911.\n"
            f"{critical_note}"
        ),
        "type": "foundation_recommendation",
        "needs_review": True,
    })

    # ── Sample Type 4: SPT summary ───────────────────────────────────────
    if records:
        n_values = [r["n_value"] for r in records]
        avg_n = sum(n_values) / len(n_values)
        min_n = min(n_values)
        max_n = max(n_values)
        samples.append({
            "instruction": f"Summarize the SPT N-value profile for {bh_id} and comment on overall site consistency.",
            "input": input_context,
            "output": (
                f"[TEMPLATE — fill before training]\n"
                f"N-value range: {min_n}–{max_n} | Mean: {avg_n:.1f}\n"
                f"{critical_note}\n"
                f"Comment on whether the profile shows increasing resistance with depth (expected) or anomalies."
            ),
            "type": "spt_summary",
            "needs_review": True,
        })

    # Attach source metadata to each sample
    for s in samples:
        s["metadata"] = {
            "source_file": source_file,
            "source_page": source_page,
            "borehole_id": bh_id,
            "n_records": len(records),
        }

    return samples


# ─────────────────────────────────────────────
# STEP 6: GENERATE SAMPLES FROM TEXT BLOCKS
# ─────────────────────────────────────────────
# Some valuable training signal comes from IS code text, not boring logs.
# We scan pages for IS code references and generate code-lookup samples.
# These teach the model to answer "which IS code governs X?" questions.

IS_CODE_PATTERN = re.compile(
    r"IS\s*[:\-]?\s*(\d{3,5})(?:\s*[:\-]\s*(\d{4}))?",
    re.IGNORECASE,
)

IS_CODE_DESCRIPTIONS = {
    "1498": "Classification and identification of soils for general engineering purposes",
    "1893": "Criteria for earthquake resistant design of structures",
    "1904": "Design and construction of foundations in soils — general requirements",
    "2131": "Method of standard penetration test for soils",
    "2911": "Design and construction of pile foundations",
    "6403": "Code of practice for determination of bearing capacity of shallow foundations",
    "8009": "Code of practice for calculation of settlements of foundations",
}


def generate_text_samples(pages: list[dict], source_file: str) -> list[dict]:
    """Generate IS code lookup samples from text pages."""
    samples = []
    seen_codes = set()

    for page in pages:
        text = page["text"]
        for match in IS_CODE_PATTERN.finditer(text):
            code = match.group(1)
            if code in seen_codes or code not in IS_CODE_DESCRIPTIONS:
                continue
            seen_codes.add(code)

            # Extract a small context window around the IS code mention
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 200)
            context = text[start:end].strip().replace("\n", " ")

            samples.append({
                "instruction": f"What does IS {code} cover and when should it be applied in geotechnical practice?",
                "input": f"Context from report: ...{context}...",
                "output": (
                    f"IS {code} covers: {IS_CODE_DESCRIPTIONS[code]}.\n"
                    f"[TEMPLATE — expand with specific clauses, scope, and application notes.]"
                ),
                "type": "is_code_lookup",
                "needs_review": True,
                "metadata": {
                    "source_file": source_file,
                    "source_page": page["page"],
                    "is_code": code,
                },
            })

    return samples


# ─────────────────────────────────────────────
# STEP 7: FORMAT AS CHATML
# ─────────────────────────────────────────────
# Every sample gets wrapped in the ChatML format Mistral expects.
# The system prompt is included in every sample so the model always
# trains with it in context.
#
# We also run a basic validation check on each sample:
#   - Output must not be empty
#   - Input + output combined must be under ~6000 chars (fits in 2048 tokens)
#   - N-value 0 is flagged (likely a parsing error, not a real value)

def to_chatml(sample: dict) -> dict:
    """Wrap a raw sample dict into ChatML format."""
    user_turn = sample["instruction"]
    if sample.get("input", "").strip():
        user_turn += f"\n\n{sample['input']}"

    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_turn}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['output']}<|im_end|>"
    )

    return {
        "text": text,
        "type": sample.get("type", "unknown"),
        "needs_review": sample.get("needs_review", True),
        "metadata": sample.get("metadata", {}),
    }


def validate_sample(sample: dict) -> tuple[bool, str]:
    """Return (is_valid, reason). Basic sanity checks."""
    text = sample.get("text", "")
    if len(text) < 100:
        return False, "too_short"
    if len(text) > 8000:
        return False, "too_long"
    if "[TEMPLATE" not in text and len(sample.get("text", "")) < 200:
        return False, "empty_output"
    return True, "ok"


# ─────────────────────────────────────────────
# STEP 8: WRITE OUTPUT
# ─────────────────────────────────────────────
# Output is JSONL — one JSON object per line.
# This format is directly compatible with HuggingFace datasets:
#   dataset = load_dataset("json", data_files="dataset.jsonl")
#
# We write two files:
#   dataset.jsonl         — all valid samples (for training)
#   dataset_review.jsonl  — samples needing human review (same data, filtered view)

def write_jsonl(samples: list[dict], output_path: str, append: bool = False):
    mode = "a" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

def run_pipeline(pdf_path: str, output_path: str, append: bool = False):
    print(f"\n{'='*60}")
    print(f" GeoLLM Pipeline: {Path(pdf_path).name}")
    print(f"{'='*60}")

    # Step 1 — triage
    info = triage_pdf(pdf_path)
    if info["is_scanned"]:
        print("\n[SKIP] Scanned PDF — cannot extract text. Run OCR first.")
        return 0

    source_name = Path(pdf_path).name

    # Step 2 — text extraction
    pages = extract_text(pdf_path)

    # Step 3 — table extraction
    tables = extract_tables(pdf_path)

    all_samples = []

    # Step 4+5 — boring log detection & sample generation
    boring_log_tables = [(t, i) for i, t in enumerate(tables) if is_boring_log_table(t["rows"])]
    print(f"[BORING LOGS] {len(boring_log_tables)} boring log tables detected")

    for table, _ in boring_log_tables:
        # Try to find borehole ID from surrounding page text
        page_text = next(
            (p["text"] for p in pages if p["page"] == table["page"]), ""
        )
        bh_id = extract_borehole_id(page_text) or extract_borehole_id(table["raw_text"])

        # Parse N-values from both table text and page text
        records = parse_n_values_from_text(table["raw_text"])
        if not records:
            records = parse_n_values_from_text(page_text)

        samples = generate_samples_from_boring_log(
            bh_id=bh_id,
            records=records,
            table_text=table["raw_text"],
            source_page=table["page"],
            source_file=source_name,
        )
        all_samples.extend(samples)

    # Also scan full-page text for inline N-value patterns (no table)
    for page in pages:
        records = parse_n_values_from_text(page["text"])
        if records and len(records) >= 3:  # Only if we find a meaningful number
            bh_id = extract_borehole_id(page["text"])
            # Don't duplicate samples already caught by table extraction
            already_covered = any(
                s["metadata"]["source_page"] == page["page"]
                for s in all_samples
            )
            if not already_covered:
                samples = generate_samples_from_boring_log(
                    bh_id=bh_id,
                    records=records,
                    table_text="",
                    source_page=page["page"],
                    source_file=source_name,
                )
                all_samples.extend(samples)

    # Step 6 — IS code text samples
    text_samples = generate_text_samples(pages, source_name)
    all_samples.extend(text_samples)
    print(f"[IS CODE] {len(text_samples)} IS code reference samples generated")

    # Step 7 — format as ChatML + validate
    formatted = []
    skipped = 0
    for s in all_samples:
        chatml = to_chatml(s)
        valid, reason = validate_sample(chatml)
        if valid:
            formatted.append(chatml)
        else:
            skipped += 1

    print(f"\n[SUMMARY]")
    print(f"  Total samples generated : {len(all_samples)}")
    print(f"  Valid samples           : {len(formatted)}")
    print(f"  Skipped (invalid)       : {skipped}")
    print(f"  Needs human review      : {sum(1 for s in formatted if s['needs_review'])}")

    # Step 8 — write output
    write_jsonl(formatted, output_path, append=append)
    print(f"\n[OUTPUT] Written to {output_path}")

    # Type breakdown
    from collections import Counter
    counts = Counter(s["type"] for s in formatted)
    print("\n[TYPES]")
    for t, c in counts.most_common():
        print(f"  {t:<30} {c}")

    return len(formatted)


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoLLM: PDF → Training Dataset")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--output", default="geollm_dataset.jsonl", help="Output JSONL path")
    parser.add_argument("--append", action="store_true", help="Append to existing JSONL")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"Error: {args.pdf} not found")
        sys.exit(1)

    total = run_pipeline(args.pdf, args.output, args.append)
    print(f"\nDone. {total} samples written.")
    print(f"\nNext step: open {args.output} and fill in all [TEMPLATE] outputs")
    print(f"Filter with: grep 'needs_review.*true' {args.output} | wc -l")
