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

api_key = "AQ.Ab8RN6IAAcxfc6ybikO4QtDjdb9OhxCVRvpNX5ovLFChNpf2lA"
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
Return structured JSON only. No markdown. No explanations.

==================================================
DOCUMENT STRUCTURE — READ CAREFULLY
==================================================

The PDF table has these columns (order may vary by bench):
  SR No | CP No | CA/IA No | Purpose | Section/Rule | Name of Parties | Counsel columns... | Remarks

Each page may contain entries from one or more LIST SECTIONS (headings):
  PRONOUNCEMENT LIST
  SUPPLEMENTARY LIST
  PRIORITY LIST (NO ADJOURNMENT)
  ORDINARY LIST
  ADMISSION LIST
  CLARIFICATION LIST
  Any other named section heading

Within each section, entries are structured as MAIN MATTER BLOCKS:

PATTERN A — CP row with IAs below (Guwahati / Kolkata style):
  <SR_NO>  <CP_NO>   "Main Case" / "Main Matter"   <STATUS_OR_PURPOSE>   <SECTION>   <PARTIES>
                      <IA_NO>                        <PURPOSE>             <SECTION>   <PARTIES>
                      <IA_NO>                        <PURPOSE>             <SECTION>   <PARTIES>

PATTERN B — Parent Matter row (unlisted) with IA rows having SR Nos (Indore style):
  [blank]  <CP_NO>   "Main Matter"   <STATUS_NOTE>   <SECTION>   <PARTIES>
  <SR_NO>  [blank]   <IA_NO_with_"in CP..." context>   <PURPOSE>   <SECTION>   <PARTIES>
  <SR_NO>  [blank]   <IA_NO_with_"in CP..." context>   <PURPOSE>   <SECTION>   <PARTIES>

PATTERN C — IA listed independently (Kochi / special bench style):
  <SR_NO>  <CP_NO>   <IA_NO>   <PURPOSE>   <SECTION>   <PARTIES>

==================================================
GLOBAL FIELDS (extract once, apply to ALL records)
==================================================

bench:           From document header. E.g.: GUWAHATI BENCH, INDORE BENCH, KOCHI BENCH
court:           From document header. E.g.: COURT -I, COURT NO. 1, COURT -II
cause_list_date: Hearing date from header. E.g.: 06.05.2026

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
  The current section heading. Carry forward until a new heading appears.
  Examples: PRONOUNCEMENT LIST, SUPPLEMENTARY LIST, ORDINARY LIST, ADMISSION LIST, CLARIFICATION LIST
  Use the exact heading text from the PDF.

sr_no:
  The serial number printed in the SR No column.
  - For CP main matter rows that have a printed SR No: use it.
  - For CP/Main Matter parent rows with NO printed SR No: leave blank.
  - For IA rows: use the SR No if one is printed (Indore style), otherwise leave blank.

cp_no:
  The main case number from the CP No column.
  - Use the exact value printed in the CP No column for that row.
  - For IA rows in Guwahati/Kolkata style: leave blank (CP is already captured in the parent row).
  - For IA rows in Indore style: leave blank (parent CP is encoded in ca_ia_no as "IA/xxx in CP/xxx").
  - Do NOT carry forward the CP No to IA sub-rows.
  - Keep complex case numbers as-is: e.g., "TP 58 of 2019 [CP(IB) 131 of 2018]", "Co.Appeal/3(MP)2024".

ca_ia_no:
  - For CP main matter rows: use the descriptor from the CA/IA column as printed.
    E.g.: "Main Case", "Main Matter", "Main Case (Final Motion)", "Main Case (1st Motion)", "MAIN CASE"
  - For IA/CA sub-rows: use the full IA/CA number as printed, including any "in CP/..." or "in IA/..." context.
    E.g.: "IA(IBC)/59/GB/2025", "IA/266(MP)2026 in CP(IB)/26(MP)2024", "IA(C/ACT)/110/KOB/2025"
  - Do NOT strip the "in CP/..." context — it is important for tracing the parent case.
  - Strip only parenthetical dates like "(dtd. 01.01.2025)" if they appear.

purpose:
  The content of the Purpose column for that specific row.
  - For CP main matter rows: this may be a hearing purpose ("For Clarification", "For Pronouncement")
    OR a status/admission note ("Admitted 16-10-2024", "Main Matter Listed on 23-06-2026",
    "Admitted vide order dated 28.10.2022, Liquidation vide order 13.10.2023",
    "Remitted Back from NCLAT vide Order dated 25.08.2025 For Further").
    Use whatever text is in the Purpose column verbatim — do NOT clean or shorten.
  - For IA rows: hearing purpose like "For Further Consideration", "For Hearing", "New Application",
    "For Pronouncement", "FOR PRONOUNCEMENT OF ORDERS".
  - Never put counsel names or bar numbers in this field.

section_rule:
  The Section/Rule column value for that row.
  Use exactly as printed — short forms are valid: "7 IBC", "9 IBC", "252(1)", "Rule 11",
  "U/s 7 of IBC, 2016", "60(5) r.w. Rule 11", "Sec 12A r.w Reg 30A", "U/R 32, R/W RULE 11 NCLT".

name_of_parties:
  Full party string from the Name of Parties column, using VS/V/S/Versus as separator.
  E.g.: "Indian Bank (FC) Vs Prokash Datta (PG) to M/s Cleanopolis Energy Systems India Private. Limited."
  If a single party (no VS), use that party name for both name_of_parties and applicant_name.
  Do NOT include counsel names.

applicant_name:
  Party BEFORE the VS / V/S / Versus / V/s separator in name_of_parties.

respondent_name:
  Party AFTER the VS / V/S / Versus / V/s separator in name_of_parties. Blank if none.

remarks:
  Actual value from the Remarks column only.
  Examples: "Ex-Party 28-08-2025", "RP Appointed", "Liq Allowed".
  Blank if the cell is "-", empty, or absent.
  Never include counsel names, bar numbers, IRP/RP/Liquidator names, or hearing times.

==================================================
IGNORE COMPLETELY
==================================================

- Counsel names and bar/enrollment numbers (e.g., F/360/274/94, WB/568/2000)
- Names of IRP / RP / Liquidator / MP / Monitoring Committee members (unless they are a party)
- CORAM / bench member names
- Webex / VC / attendance instructions
- Page numbers, footnotes, email addresses, hearing times
- Registry notes and administrative text
- "Cases filed in NCLAT against NCLT"
- "PETITIONER IN PERSON (NA)"

==================================================
QUALITY RULES
==================================================

1. Output ONE record per row in the PDF table (both CP rows and IA rows).
2. Every record must have name_of_parties populated.
3. CP main matter rows must have cp_no populated; ca_ia_no should be the descriptor ("Main Case", "Main Matter", etc.).
4. IA sub-rows must have ca_ia_no populated; cp_no should be blank.
5. Split applicant_name and respondent_name correctly from name_of_parties on the VS separator.
6. Do not merge or skip any rows.
7. Do not duplicate records.
8. Return only valid JSON — no trailing commas, no comments.
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

            print(f"{len(cases)} records extracted.")
            logging.info(f"{file_path}: {len(cases)} records.")
            break

        except json.JSONDecodeError as e:
            print("JSON parse error.")
            logging.error(f"JSON error {file_path}: {e}\nRaw: {response.text[:300]}")
            cases = []
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
                break
    else:
        print("Failed after 4 attempts, skipping.")
        logging.error(f"Extraction gave up: {file_path}")
        cases = []

    # Step 3: Delete the uploaded file from Gemini storage
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass

    return cases


def main():
    pdf_files = [f for f in os.listdir() if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the current directory.")
        return

    print(f"Found {len(pdf_files)} PDF file(s).\n")
    final_data = []

    for idx, pdf in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] {pdf}")
        rows = process_pdf(pdf)
        final_data.extend(rows)
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

    print(f"Extraction complete! {len(df)} total records saved to '{excel_file}'")
    logging.info(f"Export complete: {len(df)} records -> {excel_file}")


if __name__ == "__main__":
    main()
