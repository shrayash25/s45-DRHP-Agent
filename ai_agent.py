from openai import OpenAI
from typing import Optional
import json
import pandas as pd
from datetime import datetime

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
USE_OLLAMA = True 

if USE_OLLAMA:
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL_NAME = "gemma3:4b" 
else:
    client = OpenAI(api_key="YOUR_PROD_API_KEY")
    MODEL_NAME = "gpt-4o-mini"

# ==========================================
# 2. EXTRACTION PIPELINE
# ==========================================
def filter_relevant_text(document_text: str) -> str:
    """PASS 1: Condense large documents down to only the relevant parts."""
    if len(document_text.split()) < 1000:
        return document_text

    prompt = f"""
    You are a financial analyst reviewing a corporate filing. 
    Extract and return ONLY the exact text/paragraphs that contain information regarding:
    1. The Date of the shareholder meeting or resolution.
    2. Changes to Authorised Share Capital (Existing amounts, Revised amounts, number of shares, face value).
    3. The "Attachments" or "List of Attachments" table (crucial for finding PDF filenames).
    
    If you don't find relevant text, return "NO RELEVANT DATA FOUND".
    
    Document Text:
    {document_text}
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content


def extract_sh7_data(file_name: str, document_text: str) -> dict:
    """PASS 2: Strict JSON Extraction."""
    filtered_text = filter_relevant_text(document_text)
    
    if "NO RELEVANT DATA FOUND" in filtered_text:
        return {"error": "Document does not contain capital structure changes.", "source_file": file_name}

    prompt = f"""
    You are an expert financial AI extracting data from an Indian MCA Form SH-7.
    Extract the exact values for the authorized share capital change.
    
    CRITICAL INSTRUCTIONS:
    1. If ANY value is missing, unclear, or you are not 100% confident, you MUST return null for that field. DO NOT GUESS.
    2. To determine gm_type ("AGM" or "EGM"), you MUST scan the extracted_filenames.
    
    Filtered Document Text:
    {filtered_text}
    
    Return a JSON object with the following keys exactly:
    - meeting_date: The date of the shareholder's meeting. Format strictly as YYYY-MM-DD. (null if missing)
    - from_amount: The 'Existing' total authorised capital in Rupees. Return a pure integer no commas. (null if missing)
    - to_amount: The 'Revised' total authorised capital in Rupees. Return a pure integer no commas. (null if missing)
    - face_value: The nominal amount per equity share. Return a pure integer. (null if missing)
    - extracted_filenames: Scan the text for an Attachments list. Return a list of all filenames ending in .pdf. (empty list [] if none)
    - gm_type: If any filename in 'extracted_filenames' contains "EGM", output "EGM". If it contains "AGM", output "AGM". (null if unclear)
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    try:
        raw_json = json.loads(response.choices[0].message.content)
        
        # Native Python data cleanup (Replacing Pydantic)
        def safe_int(val):
            if val is None: return None
            try:
                # Strips commas and handles cases where the LLM returns strings like "150,000"
                return int(float(str(val).replace(',', '').strip()))
            except (ValueError, TypeError):
                return None

        cleaned_data = {
            'meeting_date': raw_json.get('meeting_date'),
            'from_amount': safe_int(raw_json.get('from_amount')),
            'to_amount': safe_int(raw_json.get('to_amount')),
            'face_value': safe_int(raw_json.get('face_value')),
            'extracted_filenames': raw_json.get('extracted_filenames', []),
            'gm_type': raw_json.get('gm_type'),
            'source_file': file_name
        }
        
        return cleaned_data
        
    except Exception as e:
        return {"error": str(e), "source_file": file_name}

# ==========================================
# 3. TABLE FORMATTING LOGIC
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