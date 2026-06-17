import os
import json
import logging
import pandas as pd
import PyPDF2
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

api_key = "AQ.Ab8RN6JdpOF-kH9Rk5DmNN8RFArX1ryfourH5QbRD4AUG7fZow"
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

# ==========================================
# EXTRACTION PROMPT
# ==========================================
GEMINI_PROMPT = """
You are an expert legal document data extraction engine for NCLT (National Company Law Tribunal) Cause Lists.

==================================================
TASK
==================================================

Extract ALL case/matter entries from the provided NCLT Cause List page text.
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

Rules for main matter blocks:
- The SR_NO appears only on the FIRST line of the block (with the CP_NO).
- One CP_NO can have ZERO or MULTIPLE IA/CA rows beneath it.
- If a CP has no CA/IA, output one record with ca_ia_no = "".
- If a CP has multiple CA/IA entries, output ONE record per CA/IA, repeating the CP details.
- Some pages show CA/IA entries that belong to a CP from the PREVIOUS page (no SR_NO visible). Use "" for sr_no in those cases, keep extracting them.
- A CP row itself (without a CA/IA) is also a valid record — include it with ca_ia_no = "".

==================================================
GLOBAL FIELDS (extract once, apply to ALL records)
==================================================

bench:
  Extract from the document header.
  Examples: KOLKATA BENCH, MUMBAI BENCH, NEW DELHI BENCH, CHENNAI BENCH

court:
  Extract from the document header.
  Examples: Court-I, Court-II, Court-III, Court No - II

cause_list_date:
  The hearing date from the document header.
  Example: 06.05.2026

==================================================
OUTPUT FORMAT
==================================================

Return ONLY this exact JSON structure:

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
FIELD-BY-FIELD RULES
==================================================

1. list_type
   The section heading that applies to this entry.
   Values: SUPPLEMENTARY | PRIORITY | ORDINARY | ADMISSION
   Carry forward the section heading to all records beneath it until a new heading appears.

--------------------------------------------------

2. sr_no
   The serial/item number printed at the start of a main matter block.
   Examples: 1, 2, 3, 101, 102, 201, 202
   For sub-IA rows that do not have their own serial number, use "" (blank).

--------------------------------------------------

3. cp_no
   The main case number. Found on the first line of each main matter block.
   Strip out any IBC stage text or dates in parentheses — put those in ibc_stage / next_hearing_note.
   Examples:
     C.P. (IB)/68(KB)2024
     C.P. (IB)/724(KB)2020
     TP/38(KB)2026
     C.P. (IB)/252(KB)2024
     C.P. (IB)/1370(KB)2018

   If a CA/IA row appears without a visible CP_NO on this page, carry forward the most recent CP_NO seen.

--------------------------------------------------

4. ibc_stage
   The "IBC Current Stage:" value printed below the CP_NO.
   Examples:
     Admitted
     Liquidation Approved
     Pending Admission
     Resolution Plan Approved
   If absent, use "".

--------------------------------------------------

5. ca_ia_no
   The linked application number. Found on sub-rows beneath the CP_NO row.
   Examples:
     IA(I.B.C)/593(KB)2026
     IA(I.B.C)/587(KB)2026
     IA (LIQ.) PROGRESS REPORT/126(KB)2025
     IVN.P (IBC)/4(KB)2025
     COMP.APPL/238(MB)2025
   If no CA/IA exists for a CP row, use "".
   Strip any date in parentheses from the number (e.g. "(11-05-2026)") — ignore it.

--------------------------------------------------

6. section_rule
   The full legal section/rule text for THIS specific row (CP row or CA/IA row).
   Examples:
     IBC under Sec 7
     IBC Under Sec 9
     Sec 433(e)/433(f) of CA 1956
     Section 60(5)/Rule 11
     Regulation 44(2), Liquidation process Regulation, 2016
     Reg. 15 of IBBI (Liq.) Regulation 2016
     Sec 42 Sections 60(5)/Regulation 13
   Keep the complete wording as it appears.

--------------------------------------------------

7. purpose
   The exact hearing purpose for this row.
   Examples:
     Admission
     Further Consideration
     For Arguments
     For Directions
     Reserved for Order
     Seeking approval of Resolution Plan
   Do NOT include counsel names or advocate bar registration numbers.

--------------------------------------------------

8. name_of_parties
   The full party title exactly as written.
   Examples:
     STATE BANK OF INDIA VS SURATGARH BIKANER TOLL ROAD COMPANY PRIVATE LIMITED
     Iserve Solutions & Services Pvt. Ltd. VS Capital Electronics And Appliances Ltd.
     Asian Distribution Trade Corporation VS ABC Products Limited
   Use the separator VS (without slashes) in the output.
   If only one party is named (no VS), capture that party as both name_of_parties and applicant_name.

--------------------------------------------------

9. applicant_name
   Party BEFORE the VS/V/S/Versus separator.
   Examples:
     STATE BANK OF INDIA
     Iserve Solutions & Services Pvt. Ltd.
     Asian Distribution Trade Corporation

--------------------------------------------------

10. respondent_name
    Party AFTER the VS/V/S/Versus separator.
    Examples:
      SURATGARH BIKANER TOLL ROAD COMPANY PRIVATE LIMITED
      Capital Electronics And Appliances Ltd.
      ABC Products Limited
    If no respondent is present (single party), use "".

--------------------------------------------------

11. remarks
    Any actual case remarks printed in the Remarks column.
    Examples:
      For Arguments
      RP Appointed
      R Plan Pending
      Liq Allowed
      Admitted
      Further Consideration
      Reserved for Order
    If the cell is blank or contains only "-", use "".
    Do NOT include counsel names, bar numbers, RP names, liquidator names, timings, or registry notes.

==================================================
WHAT TO IGNORE COMPLETELY
==================================================

Ignore these — do NOT put them in any field:
  - Name of Counsel (with bar registration numbers like F/360/274/94, WB/568/2000, etc.)
  - IRP / RP / Liquidator / MP names and roles
  - CORAM / bench member names
  - Webex / VC conference details
  - Attendance / joining instructions
  - Page numbers
  - Footnotes
  - Email addresses
  - Hearing times (10:30 A.M. etc.)
  - Registry notes
  - Text like "Cases filed in NCLAT against NCLT"
  - Text like "PETITIONER IN PERSON (NA)"

==================================================
QUALITY RULES
==================================================

Before returning the JSON:
1. Every record must have cp_no (carry forward from parent if needed).
2. Every record must have name_of_parties.
3. Split applicant_name and respondent_name from name_of_parties correctly.
4. Remove exact duplicate records.
5. Return only valid JSON — no trailing commas, no comments.
6. Do not omit any case entry. It is better to include uncertain entries than to skip them.
"""

# ==========================================
# MAIN PROCESSING LOGIC
# ==========================================
def process_pdf(file_path):
    all_rows = []

    try:
        reader = PyPDF2.PdfReader(file_path)
        total_pages = len(reader.pages)
        logging.info(f"Processing {file_path} - {total_pages} pages")
        print(f"  Total pages: {total_pages}")

        for i in range(total_pages):
            print(f"  Page {i+1}/{total_pages} ... ", end="", flush=True)
            page_text = reader.pages[i].extract_text()

            if i < total_pages - 1:
                next_text = reader.pages[i + 1].extract_text()
                page_text += "\n[NEXT PAGE CONTEXT — first 15 lines only]\n" + \
                             "\n".join(next_text.split("\n")[:15])

            if not page_text.strip():
                print("empty, skipped.")
                logging.warning(f"Page {i+1} has no extractable text, skipping.")
                continue

            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=f"{GEMINI_PROMPT}\n\nPAGE TEXT:\n{page_text}",
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                parsed = json.loads(response.text)
                cases = parsed.get("cases", [])
                if not isinstance(cases, list):
                    cases = [cases]

                all_rows.extend(cases)
                print(f"{len(cases)} records extracted.")
                logging.info(f"Page {i+1}: {len(cases)} records extracted.")

            except json.JSONDecodeError as e:
                print("JSON parse error.")
                logging.error(f"JSON parse error on page {i+1}: {e}\nRaw: {response.text[:300]}")
            except Exception as e:
                print(f"API error: {e}")
                logging.error(f"Gemini API error on page {i+1}: {e}")

    except Exception as e:
        print(f"  ERROR opening PDF: {e}")
        logging.error(f"Failed to open PDF {file_path}: {e}")

    return all_rows, total_pages


def main():
    pdf_files = [f for f in os.listdir() if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in the current directory.")
        return

    print(f"Found {len(pdf_files)} PDF file(s): {pdf_files}\n")
    final_data = []

    for idx, pdf in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] Processing: {pdf}")
        rows, pages = process_pdf(pdf)
        final_data.extend(rows)
        print(f"  Done — {len(rows)} rows from {pages} pages.\n")

    if not final_data:
        print("No valid data extracted. Check extraction_log.txt for details.")
        return

    df = pd.DataFrame(final_data)

    # JSON keys from the prompt → Excel column headers (in display order)
    column_map = {
        "sr_no":            "SR No",
        "cp_no":            "CP No",
        "ibc_stage":        "IBC Stage",
        "ca_ia_no":         "CA/IA No",
        "section_rule":     "Section/Rule",
        "purpose":          "Purpose",
        "name_of_parties":  "Name of Parties",
        "applicant_name":   "Applicant Name",
        "respondent_name":  "Respondent Name",
        "remarks":          "Remarks",
        "list_type":        "List Type",
        "cause_list_date":  "Cause List Date",
        "bench":            "Bench",
        "court":            "Court",
    }

    # Ensure every expected column exists (fill missing with blank)
    for json_key in column_map:
        if json_key not in df.columns:
            df[json_key] = ""

    # Reorder to desired columns, then rename for Excel
    df = df[list(column_map.keys())]
    df = df.rename(columns=column_map)

    # Replace NaN / None with blank string
    df = df.fillna("")

    excel_file = "nclt_cause_list.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Cause List", index=False)

        worksheet = writer.sheets["Cause List"]

        # Auto-fit column widths
        for column_cells in worksheet.columns:
            max_len = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column_cells
            )
            col_letter = column_cells[0].column_letter
            worksheet.column_dimensions[col_letter].width = min(max_len + 2, 60)

    print(f"\nExtraction complete! {len(df)} total records saved to '{excel_file}'")
    logging.info(f"Export complete: {len(df)} records written to {excel_file}")


if __name__ == "__main__":
    main()
