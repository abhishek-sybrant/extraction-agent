import os
import json
import time
import logging
import pandas as pd
from google import genai
from google.genai import types

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
MODEL_NAME = "gemini-2.5-flash"

logging.basicConfig(
    filename="extraction_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

api_key = ""
client = genai.Client(api_key=api_key)

# ==========================================
# EXTRACTION PROMPT
# ==========================================
GEMINI_PROMPT = """
You are an expert legal document data extraction engine for NCLT (National Company Law Tribunal) Cause Lists.

==================================================
TASK
==================================================

Extract ALL case/matter entries from the attached NCLT Cause List PDF.
Read EVERY page from first to last. Return structured JSON only. No markdown. No explanations.

==================================================
CRITICAL — PAGE BREAKS ARE NOT BLOCK BREAKS
==================================================

A case block (CP row + its IA sub-rows) can be split across multiple pages.
When a page ends mid-block, the IA rows continue at the TOP of the next page with NO repeated header.
You MUST treat the entire PDF as one continuous table, NOT page-by-page.

Example (Guwahati PDF, pages 3-4):
  Page 3 ends with:  108  CP(IB)/19/GB/2022  Main Case  ...  (block is NOT closed)
  Page 4 starts with (NO SR No, NO CP No — these are IA sub-rows of SR 108):
    IA(IBC)/79/GB/2025    For Further Consideration  ...
    IA(IBC)/143/GB/2025   For Further Consideration  ...
    IA(IBC)/4/GB/2026 In IA(IBC)/79/GB/2025   For Further Consideration  ...
  All three must be output as separate records.

==================================================
DOCUMENT STRUCTURE
==================================================

The PDF table has these columns (order may vary by bench):
  SR No | CP No | CA/IA No | Purpose | Section/Rule | Name of Parties | Counsel columns... | Remarks

Each page may contain entries from one or more LIST SECTION headings:
  CLARIFICATION LIST / PENDING FOR ADMISSION LIST / ORDINARY LIST / PRONOUNCEMENT LIST
  SUPPLEMENTARY LIST / PRIORITY LIST / ADMISSION LIST
  Use the exact heading text as-is. Carry forward until a new heading appears.

Within each section, entries follow one of these patterns:

PATTERN A — Guwahati / Kolkata style:
  CP row:   <SR_NO>  <CP_NO>  "Main Case"/"Main Matter"  <PURPOSE_OR_STATUS>  <SECTION>  <PARTIES>
  IA rows:  [blank]  [blank]  <IA_NO>                    <PURPOSE>            <SECTION>  <PARTIES>
  (IA rows continue on next page if block is cut by page break)

PATTERN B — Indore style (IA rows get their own SR Nos):
  CP row:   [blank]  <CP_NO>  "Main Matter"  <STATUS_NOTE>  <SECTION>  <PARTIES>
  IA rows:  <SR_NO>  [blank]  <IA_NO> in <CP_NO>  <PURPOSE>  <SECTION>  <PARTIES>

PATTERN C — Kochi special bench / order document:
  <SR_NO>  <CP_NO>  <IA_NO>  <PURPOSE>  <SECTION>  <PARTIES>

==================================================
GLOBAL FIELDS
==================================================

bench:
  From document header. E.g.: GUWAHATI BENCH, INDORE BENCH, KOCHI BENCH
  Look for "BENCH –", "BENCH:", or "NCLT ... BENCH" in the header text.

court:
  From document header.
  - If header says "COURT ROOM NO. 1" or "COURT NO. 1"  → use "COURT -I"
  - If header says "COURT ROOM NO. 2" or "COURT NO. 2"  → use "COURT -II"
  - If header says "COURT -I", "COURT -II", etc.         → use as-is
  - If NO court number is mentioned anywhere in the document → use "COURT -I" as default
  Do NOT leave this blank.

cause_list_date:
  Hearing date from header. Format as DD.MM.YYYY. E.g.: 06.05.2026

==================================================
OUTPUT FORMAT
==================================================

{
  "cases": [
    {
      "bench": "",
      "court": "",
      "cause_list_date": "",
      "list_type": "",
      "sr_no": "",
      "cp_no": "",
      "ca_ia_no": "",
      "purpose": "",
      "section_rule": "",
      "name_of_parties": "",
      "applicant_name": "",
      "respondent_name": "",
      "remarks": ""
    }
  ]
}

==================================================
FIELD RULES
==================================================

list_type:
  Exact section heading text. Carry forward across page breaks until a new heading appears.
  E.g.: "CLARIFICATION LIST", "PENDING FOR ADMISSION LIST", "ORDINARY LIST", "PRONOUNCEMENT LIST"

sr_no:
  Serial number from the SR No column for that row.
  - CP rows with a printed SR No: use it.
  - CP/Main Matter parent rows with NO SR No: leave blank.
  - IA sub-rows in Guwahati/Kolkata style: leave blank.
  - IA rows in Indore style that have their own SR No: use it.

cp_no:
  Value from the CP No column for that specific row only.
  - For IA sub-rows (Guwahati/Kolkata/Indore): leave blank — do NOT carry forward the parent's CP No.
  - Keep complex formats as-is: "TP 58 of 2019 [CP(IB) 131 of 2018]", "Co.Appeal/3(MP)2024"

ca_ia_no:
  - For CP main matter rows: the descriptor text in the CA/IA column.
    E.g.: "Main Case", "Main Matter", "Main Case (Final Motion)", "Main Case (1st Motion)", "MAIN CASE"
  - For IA/CA sub-rows: full IA/CA number including any "in CP/..." or "In IA/..." chain.
    E.g.: "IA(IBC)/59/GB/2025", "IA(IBC)/4/GB/2026 In IA(IBC)/79/GB/2025",
          "IA/266(MP)2026 in CP(IB)/26(MP)2024", "IA(C/ACT)/110/KOB/2025"
  - Preserve "in CP/..." context — do NOT remove it.
  - Strip only isolated parenthetical hearing dates like "(dtd. 01.01.2025)".

purpose:
  Content of the Purpose column verbatim for that row.
  - CP rows may have a hearing purpose OR a status/stage note:
    "For Clarification", "For Pronouncement", "Admitted 16-10-2024",
    "Main Matter Listed on 23-06-2026",
    "(Admitted as on 21-12-2018 & Liquidation vide order dtd. 19.12.2023)",
    "Remitted Back from NCLAT vide Order dated 25.08.2025 For Further",
    "Admitted vide order dated 28.10.2022, Liquidation vide order 13.10.2023 & Dissolution vide order 13.12.2024"
  - IA rows: "For Further Consideration", "For Hearing", "New Application", "FOR PRONOUNCEMENT OF ORDERS"
  - Never include counsel names.

section_rule:
  Section/Rule column value exactly as printed.
  E.g.: "U/s 7 of IBC, 2016", "60(5) r.w. Rule 11", "7 IBC", "252(1)", "U/R 32, R/W RULE 11 NCLT"

name_of_parties:
  Full party string. Use VS/V/S/Versus as-is for separation.
  If single party (no VS), that name is both name_of_parties and applicant_name.
  Do NOT include counsel names, IRP/RP/Liquidator names.

applicant_name:
  Party BEFORE the VS / V/S / Versus / V/s separator.

respondent_name:
  Party AFTER the VS / V/S / Versus / V/s separator. Blank if no respondent.

remarks:
  Value from Remarks column only. E.g.: "Ex-Party 28-08-2025", "RP Appointed".
  Blank if "-", empty, or absent. Never include counsel names or bar numbers.

==================================================
IGNORE COMPLETELY
==================================================

- Counsel/advocate names and bar/enrollment numbers
- IRP / RP / Liquidator / MP names (unless they are a named party)
- CORAM and bench member names
- Webex / VC / video conference instructions
- Page numbers, footnotes, email addresses, hearing times
- Registry notes, footer text ("Note: Although all efforts...")
- Copy-to lists at end of document
- "Cases filed in NCLAT against NCLT"

==================================================
QUALITY RULES
==================================================

1. Process ALL pages. A block cut by a page break continues on the next page.
2. Output ONE record per table row — both CP rows AND every IA/CA sub-row.
3. Every record must have name_of_parties populated.
4. CP rows must have cp_no; ca_ia_no must be the descriptor ("Main Case", "Main Matter", etc.).
5. IA sub-rows must have ca_ia_no; cp_no must be blank.
6. court must never be blank — default to "COURT -I" if not stated in the document.
7. list_type must never be blank — carry forward the last seen section heading.
8. Do not skip, merge, or duplicate any row.
9. Return only valid JSON — no trailing commas, no comments.
"""

# ==========================================
# MAIN PROCESSING LOGIC
# ==========================================
def process_pdf(file_path):
    # Step 1: Upload the whole PDF once
    print(f"  Uploading ... ", end="", flush=True)
    uploaded_file = None

    for attempt in range(1, 5):
        try:
            uploaded_file = client.files.upload(
                file=file_path,
                config={"mime_type": "application/pdf"},
            )
            print(f"done ({uploaded_file.name}).")
            logging.info(f"Uploaded {file_path} -> {uploaded_file.name}")
            break
        except Exception as e:
            err = str(e)
            if "503" in err or "UNAVAILABLE" in err or "429" in err:
                wait = 10 * attempt
                print(f"busy, retrying in {wait}s (attempt {attempt}/4)...")
                time.sleep(wait)
            else:
                print(f"upload error: {e}")
                logging.error(f"Upload failed {file_path}: {e}")
                return []
    else:
        print("Upload failed after 4 attempts, skipping.")
        logging.error(f"Upload gave up: {file_path}")
        return []

    # Step 2: Single API call with the uploaded PDF
    print(f"  Extracting ... ", end="", flush=True)

    for attempt in range(1, 5):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type="application/pdf",
                    ),
                    GEMINI_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )

            parsed = json.loads(response.text)
            cases = parsed.get("cases", [])
            if not isinstance(cases, list):
                cases = [cases]

            usage = response.usage_metadata
            in_tok  = usage.prompt_token_count     if usage else 0
            out_tok = usage.candidates_token_count if usage else 0
            print(f"{len(cases)} records extracted.  [{in_tok:,} in / {out_tok:,} out tokens]")
            logging.info(f"{file_path}: {len(cases)} records. Tokens: {in_tok} in / {out_tok} out.")
            break

        except json.JSONDecodeError as e:
            print("JSON parse error.")
            logging.error(f"JSON error {file_path}: {e}\nRaw: {response.text[:300]}")
            cases = []
            in_tok, out_tok = 0, 0
            break
        except Exception as e:
            err = str(e)
            if "503" in err or "UNAVAILABLE" in err or "429" in err:
                wait = 10 * attempt
                print(f"API busy, retrying in {wait}s (attempt {attempt}/4)...")
                time.sleep(wait)
            else:
                print(f"API error: {e}")
                logging.error(f"API error {file_path}: {e}")
                cases = []
                in_tok, out_tok = 0, 0
                break
    else:
        print("Failed after 4 attempts, skipping.")
        logging.error(f"Extraction gave up: {file_path}")
        cases = []
        in_tok, out_tok = 0, 0

    # Step 3: Delete the uploaded file from Gemini storage
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass

    return cases, in_tok, out_tok


def main():
    pdf_files = [f for f in os.listdir() if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the current directory.")
        return

    # Gemini 2.5 Flash pricing (non-thinking)
    PRICE_IN_PER_1M  = 0.15   # USD per 1M input tokens
    PRICE_OUT_PER_1M = 0.60   # USD per 1M output tokens

    print(f"Found {len(pdf_files)} PDF file(s).\n")
    final_data = []
    total_in_tok = 0
    total_out_tok = 0

    for idx, pdf in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] {pdf}")
        rows, in_tok, out_tok = process_pdf(pdf)
        final_data.extend(rows)
        total_in_tok  += in_tok
        total_out_tok += out_tok
        print(f"  Done — {len(rows)} rows.\n")

    if not final_data:
        print("No valid data extracted. Check extraction_log.txt for details.")
        return

    df = pd.DataFrame(final_data)

    column_map = {
        "sr_no":           "SR No",
        "cp_no":           "CP No",
        "ca_ia_no":        "CA/IA No",
        "purpose":         "Case Purpose",
        "section_rule":    "Section",
        "name_of_parties": "Name of Parties",
        "remarks":         "Remarks",
        "cause_list_date": "Date of Cause List",
        "bench":           "BENCH",
        "court":           "COURT",
        "applicant_name":  "Applicant Name",
        "respondent_name": "Respondent Name",
        "list_type":       "List Type",
    }

    for key in column_map:
        if key not in df.columns:
            df[key] = ""

    df = df[list(column_map.keys())].rename(columns=column_map).fillna("")

    excel_file = "nclt_cause_list.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Cause List", index=False)
        ws = writer.sheets["Cause List"]
        for col_cells in ws.columns:
            max_len = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in col_cells
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 60)

    cost_in  = total_in_tok  / 1_000_000 * PRICE_IN_PER_1M
    cost_out = total_out_tok / 1_000_000 * PRICE_OUT_PER_1M
    total_cost = cost_in + cost_out

    print(f"Extraction complete! {len(df)} total records saved to '{excel_file}'")
    print(f"\n{'='*50}")
    print(f"  Token Usage & Cost Summary ({MODEL_NAME})")
    print(f"{'='*50}")
    print(f"  Input  tokens : {total_in_tok:>12,}   @ ${PRICE_IN_PER_1M}/1M  = ${cost_in:.6f}")
    print(f"  Output tokens : {total_out_tok:>12,}   @ ${PRICE_OUT_PER_1M}/1M  = ${cost_out:.6f}")
    print(f"  {'─'*44}")
    print(f"  Total tokens  : {total_in_tok + total_out_tok:>12,}   Total cost   = ${total_cost:.6f}")
    print(f"{'='*50}\n")
    logging.info(f"Export complete: {len(df)} records -> {excel_file}. Tokens: {total_in_tok} in / {total_out_tok} out. Cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
