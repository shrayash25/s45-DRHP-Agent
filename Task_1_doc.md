# System Design Document: AI-Assisted Financial Data Extraction Pipeline

## 1. Problem Interpretation & Core Philosophy

The primary objective of this system is to accurately extract data from unstructured SH-7 documents and generate a compliant, Authorised share capital change table. During the design process, a key realization was that large language models (LLMs) are unreliable when tasked with numerical computation and complex table formatting.

### Extraction over Generation
The system is designed to treat the LLM strictly as a **structured data extractor**, not a generator. Instead of asking the model to produce final tables or perform calculations, it is only responsible for identifying and extracting key fields from the source documents.

### Zero-Hallucination Mandate
In financial and compliance contexts, incorrect data is more harmful than missing data. Therefore, the system enforces a strict **“do not guess” policy**. If the LLM is uncertain, it must return `null` instead of fabricating values.

### Separation of Concerns
The architecture separates the AI logic from the user interface:
- `ai_agent.py` handles all interactions with the LLM
- `app.py` manages the frontend and presentation layer  

This modular design ensures scalability and allows the backend to be integrated into larger enterprise systems without dependency on the UI.

---

## 2. The AI Extraction Engine (Controlling the LLM)

### Strict JSON Enforcement
The LLM is constrained to return a predefined JSON structure containing:
- `meeting_date`
- `from_amount`
- `to_amount`
- `face_value`
- `extracted_filenames`
- `gm_type`

This ensures consistency and simplifies downstream processing.

### Chain-of-Thought Strategy for AGM/EGM Detection
A key challenge was identifying whether a meeting was an AGM or EGM, as this information was not present in the main content but embedded within attachment filenames.

To solve this:
- The prompt first forces the LLM to extract `extracted_filenames`
- Only after processing filenames does the model determine `gm_type`

This controlled reasoning approach prevents blind guessing and improves accuracy.

### Model Agnosticism
The system is built using a standard API interface, allowing flexibility in model selection:
- Can run on local models (e.g., LLaMA via Ollama)
- Can switch to production-grade APIs (e.g., GPT-based models) with minimal configuration changes  

This ensures adaptability across environments.

---

## 3. The Deterministic Engine (Python Business Logic)

### Why Python Handles Calculations
LLMs are prone to arithmetic errors. To eliminate this risk:
- The LLM extracts only raw values (`from_amount`, `to_amount`, `face_value`)
- Python performs all mathematical operations  

Example:
```python
number_of_shares = amount // face_value