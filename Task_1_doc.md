# System Design Document: DRHP Capital Structure Drafting Agent

## 1. Problem Interpretation & Core Philosophy

The goal of this system is to take data from messy, unstructured SH-7 forms and supporting documents, and produce a clean, compliant *Authorised Share Capital Change* table. Early on, it became clear that LLMs shouldn't be used to do math, format complex tables, or sort through mixed document batches on their own.

**Extraction**
The LLM's only job here is to find and extract specific fields from source documents - nothing more. It doesn't generate final tables or run any calculations.

**Zero-Hallucination**
In financial and compliance work, a wrong number is worse than a missing one. So the system has a strict rule: if the model isn't sure, it returns `null`.

**Separation of Concerns**
The system is split into two parts:

- `ai_agent.py` — handles all LLM interactions and data cleaning
- `app.py` — manages the frontend, document routing, and display

This keeps the backend clean and independent, making it easy to plug into larger systems down the line.

---

## 2. The AI Extraction (Controlling the LLM)

**Two-Pass Architecture**

In practice, users rarely upload a clean batch of just SH-7 forms. Board Resolutions, EGM Notices, and other filings get mixed in. To deal with this without generating duplicate or incorrect rows, the system uses a two-pass pipeline:

- **Pass 1 – Classification:** Each document is evaluated individually to determine whether it's a valid Form SH-7.
- **Pass 2 – Extraction:** Only runs if Pass 1 confirms the document is a genuine SH-7. This prevents the system from trying to extract share capital data from the wrong files.

**Optimizing for Local Models & Context Windows**

To avoid memory issues and reduce processing time during Pass 1, the engine only sends the first 2,000 words of each document to the LLM - enough to capture the key headers and titles, without loading unnecessary boilerplate.

**Chain-of-Thought (CoT) for Hallucination Prevention**

Smaller local models tend to over-classify — for example, flagging an EGM Notice as an SH-7. To counter this, Pass 1 forces the model to output a `reasoning` field explaining what the document is *before* it outputs the `is_sh7` boolean. Writing out the reasoning first grounds the model's context, making it far less likely to return a false positive.

The same principle applies in Pass 2: the model must extract `extracted_filenames` before it classifies `gm_type` (AGM vs. EGM), so it actually reads the attachment table before deciding.

**Defensive Parsing**

Open-source LLMs often wrap their JSON output in markdown fences or add conversational filler, which breaks standard `json.loads()`. A custom regex parser removes all of that away and isolates just the JSON — keeping the pipeline stable regardless of how the model formats its response.

**AI Model**

The system uses a standard API interface, so switching models is straightforward:

- **Local models** (e.g., Gemma via Ollama) for free, private testing
- **Production APIs** (e.g., GPT-4o) with a simple API key swap

---

## 3. The Python Calculationg Logic

**Why Python Handles All Calculations**

LLMs make arithmetic mistakes. To eliminate that risk entirely, the LLM only extracts raw values (`from_amount`, `to_amount`, `face_value`) — Python does all the math.
```python
number_of_shares = amount // face_value
```

Number formatting (including the Indian numbering system, e.g., `10,00,000`) and legal phrasing are also handled entirely by Python, not the model.

**Handling the "On Incorporation" Row**

The output table requires an *"On incorporation"* row, but SH-7 forms only record *changes* to capital — not the founding state. To work around this, the code sorts extracted records chronologically, reads the `from_amount` and `face_value` from the earliest event, and uses those to backfill the incorporation row automatically.

**Graceful Degradation (Null Handling)**

When the LLM isn't confident about a value, it returns `null`. The code catches these and inserts `[NEEDS VERIFICATION]` in the final table — making it easy for a human reviewer to spot and fix anything uncertain.

---

## 4. Traceability & Auditability

Full auditability is non-negotiable in banking software.

**Classification Ledger**
The UI shows the LLM's chain-of-thought reasoning for every uploaded document — so it's always clear why a file was included or skipped.

**Audit Trail**
Every row in the final DRHP table links back to its source document. An expandable *Audit Trail* section at the bottom of the UI also displays the raw JSON output for each processed file, showing exactly where every number came from.