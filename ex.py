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

Each page may contain entries from one or more LIST SECTIONS:
  SUPPLEMENTARY
  PRIORITY (NO ADJOURNMENT)
  ORDINARY
  ADMISSION

Each list section contains MAIN MATTER BLOCKS. A main matter block looks like:

  <SR_NO>  <CP_NO>
            IBC Current Stage: <IBC_STAGE>
            <SECTION/RULE>    <PURPOSE>    <NAME_OF_PARTIES>    <COUNSEL...>    <REMARKS>
            <IA/CA_NO>        <SECTION/RULE>    <PURPOSE>    <NAME_OF_PARTIES>    ...

Rules:
- SR_NO appears only on the FIRST line of the block (with the CP_NO).
- One CP_NO can have ZERO or MULTIPLE IA/CA rows beneath it.
- If a CP has no CA/IA, output one record with ca_ia_no = "".
- If a CP has multiple CA/IA entries, output ONE record per CA/IA, repeating the CP details.
- A CP row itself (without a CA/IA) is also a valid record — include it with ca_ia_no = "".

==================================================
GLOBAL FIELDS (extract once, apply to ALL records)
==================================================

bench:    From document header. E.g.: KOLKATA BENCH, MUMBAI BENCH, NEW DELHI BENCH
court:    From document header. E.g.: Court-I, Court-II, Court No - II
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
      "ibc_stage": "",
      "ca_ia_no": "",
      "section_rule": "",
      "purpose": "",
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

list_type:  Section heading for this entry. Values: SUPPLEMENTARY | PRIORITY | ORDINARY | ADMISSION. Carry forward until a new heading appears.

sr_no:  Serial number at start of main matter block. E.g.: 1, 2, 101, 201. Blank for sub-IA rows without their own number.

cp_no:  Main case number. Strip IBC stage text and dates in parentheses. E.g.: C.P. (IB)/68(KB)2024, TP/38(KB)2026. Carry forward for CA/IA rows that don't restate it.

ibc_stage:  "IBC Current Stage:" value below CP_NO. E.g.: Admitted, Liquidation Approved, Pending Admission, Resolution Plan Approved. Blank if absent.

ca_ia_no:  Linked application number on sub-rows. E.g.: IA(I.B.C)/593(KB)2026, IA (LIQ.) PROGRESS REPORT/126(KB)2025. Strip any date in parentheses. Blank if none.

section_rule:  Full legal section/rule text for this row. E.g.: IBC under Sec 7, Section 60(5)/Rule 11, Reg. 15 of IBBI (Liq.) Regulation 2016. Keep complete wording.

purpose:  Exact hearing purpose. E.g.: Admission, Further Consideration, For Arguments, Reserved for Order. No counsel names.

name_of_parties:  Full party title using VS as separator. E.g.: STATE BANK OF INDIA VS SURATGARH BIKANER TOLL ROAD COMPANY PRIVATE LIMITED. If single party, use as both name_of_parties and applicant_name.

applicant_name:  Party BEFORE the VS/V/S/Versus separator.

respondent_name:  Party AFTER the VS/V/S/Versus separator. Blank if none.

remarks:  Actual remarks from the Remarks column only. E.g.: RP Appointed, Liq Allowed, Reserved for Order. Blank if cell is "-" or empty. No counsel names, bar numbers, timings.

==================================================
IGNORE COMPLETELY
==================================================

- Counsel names and bar registration numbers (F/360/274/94, WB/568/2000, etc.)
- IRP / RP / Liquidator / MP names
- CORAM / bench member names
- Webex / VC / attendance instructions
- Page numbers, footnotes, email addresses, hearing times
- Registry notes
- "Cases filed in NCLAT against NCLT"
- "PETITIONER IN PERSON (NA)"

==================================================
QUALITY RULES
==================================================

1. Every record must have cp_no and name_of_parties.
2. Split applicant_name and respondent_name correctly from name_of_parties.
3. Remove exact duplicate records.
4. Return only valid JSON — no trailing commas, no comments.
5. Do not omit any case entry.
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
        "ibc_stage":       "IBC Stage",
        "ca_ia_no":        "CA/IA No",
        "section_rule":    "Section/Rule",
        "purpose":         "Purpose",
        "name_of_parties": "Name of Parties",
        "applicant_name":  "Applicant Name",
        "respondent_name": "Respondent Name",
        "remarks":         "Remarks",
        "list_type":       "List Type",
        "cause_list_date": "Cause List Date",
        "bench":           "Bench",
        "court":           "Court",
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
