Note: This prompt was developed with some help from Gemini 1.5 Pro and run through Claude. Because this is a proof-of-concept, I decided to keep things simple and build the UI with Streamlit in two modular Python files. If we were taking this to production, I would upgrade the architecture to a React/Next.js frontend and a dedicated Python backend (like FastAPI or Django) to handle the scale.


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