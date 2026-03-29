Note: This prompt was developed with some help from Gemini 1.5 Pro and run through Claude. Because this is a proof-of-concept, I decided to keep things simple and build the UI with Streamlit in two modular Python files. If we were taking this to production, I would upgrade the architecture to a React/Next.js frontend and a dedicated Python backend (like FastAPI or Django) to handle the scale.

**Prompt 1**
**System Context:**
I need to build a "DRHP Capital Structure Drafting Agent" in Python. The system takes 4 OCR-extracted MCA Form SH-7 documents (as markdown files) and generates an Authorised Share Capital change table. 

**Strict Constraints:**
1. **No Hallucinations:** Every number must trace back to the documents. If data is missing or unclear, the system must not guess.
2. **Deterministic Architecture:** Do not ask the LLM to generate the final table. Use the LLM strictly for structured JSON data extraction, and use native Python to format the legal strings and calculate math.

**Tech Stack:**
- Python, Pandas, Streamlit (for the UI)
- OpenAI Python SDK (configured to toggle between a local Ollama instance and production API)

**File Structure:**
Please split the code into two files: `ai_agent.py` (backend logic) and `app.py` (frontend UI).

**Requirements for `ai_agent.py`:**
1. **The LLM Extraction Function:** Create a function that takes document text and returns a JSON object. Use `response_format={"type": "json_object"}` and `temperature=0.0`.
2. **The Schema:** Extract the following keys. Allow `null` for any field if the LLM is unsure:
   - `meeting_date`: Format strictly as YYYY-MM-DD.
   - `from_amount`: Pure integer, remove commas.
   - `to_amount`: Pure integer, remove commas.
   - `face_value`: Pure integer.
   - `extracted_filenames`: List of strings. Force the LLM to scan the "Attachments" table at the bottom of the document and list the PDF filenames.
   - `gm_type`: String ("AGM" or "EGM"). Instruct the LLM to determine this by looking at the `extracted_filenames` list. (This Chain-of-Thought prevents hallucinations).
3. **Data Cleaning:** Use a native Python `safe_int` function to handle strings with commas (e.g., "1,50,000" -> 150000) and gracefully handle nulls.
4. **Table Formatting Logic:** - Write a Python function to calculate shares (`amount // face_value`).
   - Format the string using the Indian numbering system (e.g., "10,00,000 divided into 1,00,000 Equity Shares of 10 each").
   - If any data is null, output "[NEEDS VERIFICATION]".
5. **Table Assembly:** Sort the extracted events chronologically. Automatically generate an "On incorporation" row at the very top by looking at the `from_amount` of the earliest chronological event. 
   - Required DataFrame columns: "Particulars of Change", "Date of Shareholder's Meeting", "AGM/EGM", "From", "To", "Source Document".

**Requirements for `app.py`:**
1. Build a Streamlit UI with a wide layout.
2. Allow multiple `.md` or `.txt` file uploads.
3. On button click, process all files through the `ai_agent.py` pipeline and display the Pandas DataFrame using `st.dataframe` spanning the full width.
4. Include an `st.download_button` to download the table as a CSV.
5. Below the table, build an "Audit Trail & Traceability" section using `st.expander` to display the raw JSON dictionary extracted from each source file so the user can verify the LLM's work.


**Prompt 2: Upgrading to a Two-Pass Architecture for Mixed Document Batches**

**System Context:**
We need to upgrade the existing "DRHP Capital Structure Drafting Agent" to handle a mixed batch of uploads. Instead of assuming every file is an SH-7, the system might receive Board Resolutions, EGM Notices, or Drafts alongside the official SH-7 filings.

**Strict Constraints:**
1. **Context Window Protection:** For the classification pass, only send the first 2000 words of the document to the LLM to prevent context-overflow in local models.
2. **Chain-of-Thought (CoT) Routing:** Small models often suffer from "False Positives" (e.g., classifying an EGM Notice as an SH-7). To fix this, force the LLM to write out its reasoning *before* it outputs the classification boolean.
3. **Bulletproof JSON Parsing:** Local models often wrap JSON in markdown (e.g., ` ```json `). You must implement a Regex helper function to strip all conversational filler and strictly parse the dictionary.

**Upgrades for `ai_agent.py`:**
1. **Add `clean_and_parse_json(raw_text)`:** Use the `re` module to extract text strictly between `{` and `}` to prevent `JSONDecodeError`s. Apply this to all LLM responses.
2. **Add `extract_document_metadata` (Pass 1):** Create a new LLM function that reads the document snippet. Ask for the following JSON keys strictly in this order:
   - `reasoning`: A 1-2 sentence explanation of what the document actually is based on context.
   - `actual_document_type`: String (e.g., "EGM Notice", "Form SH-7", "Board Resolution").
   - `is_sh7`: Boolean. True ONLY if it contains "FORM NO. SH-7".
   - `meeting_date`: YYYY-MM-DD.
   - `status`: "Official Filing" or "Draft".
   *(Use `.get()` with safe defaults in Python to prevent KeyErrors if the LLM skips a field).*
3. **Keep `extract_sh7_data` (Pass 2):** Leave the strict ASC extraction logic unchanged, but make sure it uses the new regex JSON cleaner.

**Upgrades for `app.py`:**
1. **Smart Routing Loop:** Iterate through all uploaded files individually. Run Pass 1 (`extract_document_metadata`) on every file. If and ONLY if `is_sh7` is exactly `True`, pass that specific file to Pass 2 (`extract_sh7_data`).
2. **Add a Classification Ledger:** Above the DRHP table, render a new Pandas DataFrame (`st.dataframe`) showing the Pass 1 results for *all* uploaded files. 
   - Columns should be: "Filename", "Detected Document Type", "AI Reasoning", "Is SH-7?", and "Meeting Date". This provides total transparency into why the AI skipped certain documents.