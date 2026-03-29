from openai import OpenAI
from typing import Optional
import json
import pandas as pd
from datetime import datetime
import re

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
USE_OLLAMA = True 

if USE_OLLAMA:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL_NAME = "gemma3:4b" # Adjust to your local model
else:
    client = OpenAI(api_key="YOUR_PROD_API_KEY")
    MODEL_NAME = "gpt-4o-mini"

# ==========================================
# HELPER: BULLETPROOF JSON PARSER
# ==========================================
def clean_and_parse_json(raw_text: str) -> dict:
    """Strips markdown backticks and conversational filler from local LLM outputs."""
    try:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            clean_text = match.group(0)
            return json.loads(clean_text)
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise Exception(f"JSON Parse Failed. Raw text was: {raw_text}")

# ==========================================
# 2. PASS 1: CHAIN-OF-THOUGHT CLASSIFICATION
# ==========================================
def extract_document_metadata(file_name: str, document_text: str) -> dict:
    """Evaluates tokens and uses Chain-of-Thought to prevent false positives."""
    
    snippet = " ".join(document_text.split()[:2000])
    
    prompt = f"""
    You are a highly strict document classifier. Read the text and identify the document.
    
    CRITICAL RULES:
    1. Look closely at the context. If the document is an "EGM Notice", "Extraordinary General Meeting", "Board Resolution", or "Minutes", it is DEFINITELY NOT an SH-7.
    2. A document is ONLY an SH-7 if it is the official MCA filing containing the exact phrase "FORM NO. SH-7".
    
    OUTPUT FORMAT: 
    Return ONLY a valid JSON object. DO NOT use markdown formatting.
    
    Document Text:
    {snippet}
    
    Return a JSON object with exactly these keys IN THIS EXACT ORDER:
    - reasoning: Briefly explain what this document is based on its contents (1-2 sentences).
    - actual_document_type: The specific type of document (e.g., "EGM Notice", "Board Resolution", "Form SH-7").
    - is_sh7: Boolean (true/false). Based strictly on your reasoning, is this the Form SH-7?
    - meeting_date: The date of the meeting mentioned, formatted as YYYY-MM-DD. (null if not found).
    - status: "Official Filing" if it looks like a submitted form/resolution, or "Draft" if unclear.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        raw_text = response.choices[0].message.content
        raw_json = clean_and_parse_json(raw_text)
        
        # We explicitly enforce the bool type just in case the LLM outputs "False" as a string
        is_sh7_flag = raw_json.get("is_sh7", False)
        if str(is_sh7_flag).lower() == "true":
            is_sh7_flag = True
        else:
            is_sh7_flag = False

        return {
            "source_file": file_name,
            "reasoning": raw_json.get("reasoning", "No reasoning provided"),
            "actual_document_type": raw_json.get("actual_document_type", "Unknown"),
            "is_sh7": is_sh7_flag, 
            "meeting_date": raw_json.get("meeting_date"),
            "status": raw_json.get("status", "Unknown")
        }
        
    except Exception as e:
        print(f"Pass 1 Error on {file_name}: {e}")
        return {
            "error": str(e), 
            "source_file": file_name, 
            "reasoning": "Extraction Error",
            "is_sh7": False,
            "actual_document_type": "Error",
            "meeting_date": None,
            "status": "Error"
        }

# ==========================================
# 3. PASS 2: STRICT SH-7 EXTRACTION
# ==========================================
def extract_sh7_data(file_name: str, document_text: str) -> dict:
    """Extracts the math and dates specifically for the ASC DRHP Table."""
    
    prompt = f"""
    You are an expert financial AI extracting data from an Indian MCA Form SH-7.
    Extract the exact values for the authorized share capital change.
    
    CRITICAL INSTRUCTIONS:
    1. If ANY value is missing, unclear, or you are not 100% confident, you MUST return null. DO NOT GUESS.
    2. To determine gm_type ("AGM" or "EGM"), scan the text or "Attachments" list for the filenames.
    
    OUTPUT FORMAT: 
    Return ONLY a valid JSON object. DO NOT use markdown formatting.
    
    Document Text:
    {document_text}
    
    Return a JSON object with the following keys exactly:
    - meeting_date: Date of the shareholder's meeting. Format strictly as YYYY-MM-DD. (null if missing)
    - from_amount: The 'Existing' total authorised capital in Rupees. Pure integer, no commas. (null if missing)
    - to_amount: The 'Revised' total authorised capital in Rupees. Pure integer, no commas. (null if missing)
    - face_value: The nominal amount per equity share. Pure integer. (null if missing)
    - extracted_filenames: Scan for an Attachments list. Return a list of all filenames ending in .pdf. (empty list [] if none)
    - gm_type: If any filename or text contains "EGM", output "EGM". If "AGM", output "AGM". (null if unclear)
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        raw_text = response.choices[0].message.content
        raw_json = clean_and_parse_json(raw_text)
        
        def safe_int(val):
            if val is None: return None
            try:
                return int(float(str(val).replace(',', '').strip()))
            except (ValueError, TypeError):
                return None

        return {
            'meeting_date': raw_json.get('meeting_date'),
            'from_amount': safe_int(raw_json.get('from_amount')),
            'to_amount': safe_int(raw_json.get('to_amount')),
            'face_value': safe_int(raw_json.get('face_value')),
            'extracted_filenames': raw_json.get('extracted_filenames', []),
            'gm_type': raw_json.get('gm_type'),
            'source_file': file_name
        }
    except Exception as e:
        print(f"Pass 2 Error on {file_name}: {e}")
        return {"error": str(e), "source_file": file_name}

# ==========================================
# 4. TABLE FORMATTING LOGIC
# ==========================================
def format_capital_string(amount: Optional[int], face_value: Optional[int]) -> str:
    """Formats the legal string, handling nulls gracefully."""
    if amount is None or face_value is None:
        return "[NEEDS VERIFICATION: Missing Data]"
        
    shares = amount // face_value
    
    def inr_format(n):
        s = str(n)
        if len(s) <= 3: return s
        return inr_format(s[:-3]) + ',' + s[-3:]
    
    return f"{inr_format(amount)} divided into {inr_format(shares)} Equity Shares of {face_value} each"

def build_drhp_table(extractions: list) -> pd.DataFrame:
    """Compiles the extracted JSONs into the final DataFrame."""
    valid_dates = [x for x in extractions if x.get('meeting_date') is not None]
    invalid_dates = [x for x in extractions if x.get('meeting_date') is None]
    
    valid_dates.sort(key=lambda x: x['meeting_date'])
    sorted_extractions = valid_dates + invalid_dates

    table_rows = []
    
    if sorted_extractions and sorted_extractions[0].get('from_amount') is not None:
        first_event = sorted_extractions[0]
        incorp_string = format_capital_string(first_event['from_amount'], first_event['face_value'])
        table_rows.append({
            "Particulars of Change": "On incorporation",
            "Date of Shareholder's Meeting": "-",
            "AGM/EGM": "-",
            "From": "-",
            "To": incorp_string,
            "Source Document": "Inferred from base state"
        })

    for data in sorted_extractions:
        if data.get('meeting_date'):
            try:
                display_date = datetime.strptime(data['meeting_date'], '%Y-%m-%d').strftime('%B %d, %Y')
            except ValueError:
                display_date = f"[INVALID DATE FORMAT: {data['meeting_date']}]"
        else:
            display_date = "[NEEDS VERIFICATION]"
            
        from_str = format_capital_string(data.get('from_amount'), data.get('face_value'))
        to_str = format_capital_string(data.get('to_amount'), data.get('face_value'))
        gm_type = data.get('gm_type')
        
        table_rows.append({
            "Particulars of Change": "Increase in authorized share capital",
            "Date of Shareholder's Meeting": display_date,
            "AGM/EGM": gm_type if gm_type else "[NEEDS VERIFICATION]",
            "From": from_str,
            "To": to_str,
            "Source Document": data['source_file']
        })

    return pd.DataFrame(table_rows)