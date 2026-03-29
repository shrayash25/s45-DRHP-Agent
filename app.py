import streamlit as st
import pandas as pd
from ai_agent import extract_document_metadata, extract_sh7_data, build_drhp_table, USE_OLLAMA, MODEL_NAME

# ==========================================
# STREAMLIT FRONTEND
# ==========================================
st.set_page_config(page_title="DRHP Capital Structure Agent", layout="wide")

st.title("📄 DRHP Capital Structure Drafting Agent")
st.markdown("Upload a batch of mixed company filings. The system extracts metadata individually to reliably isolate SH-7s before generating the Capital Structure table.")

st.sidebar.header("System Configuration")
st.sidebar.info(f"**Current Mode:** {'Local (Ollama)' if USE_OLLAMA else 'Production (API)'}\n\n**Model:** `{MODEL_NAME}`")

uploaded_files = st.file_uploader("Upload Mixed Filings (.md, .txt)", accept_multiple_files=True, type=['md', 'txt'])

if uploaded_files:
    st.info(f"📁 **{len(uploaded_files)} Documents Ingested and Ready for Processing.**")

if st.button("Extract Metadata & Process Documents", type="primary") and uploaded_files:
    with st.spinner(f"Running individual document evaluation with {MODEL_NAME}..."):
        metadata_ledger = []
        asc_extractions = []
        
        for file in uploaded_files:
            text = file.read().decode("utf-8")
            
            # --- PASS 1: Strict Evaluation ---
            doc_metadata = extract_document_metadata(file.name, text)
            metadata_ledger.append(doc_metadata)
            
            # --- PASS 2: Smart Routing ---
            if doc_metadata.get("is_sh7") is True and "error" not in doc_metadata:
                extracted_data = extract_sh7_data(file.name, text)
                if "error" not in extracted_data:
                    asc_extractions.append(extracted_data)
        
        st.success("Individual Processing Complete!")
        
        # --- DISPLAY 1: METADATA LEDGER ---
        st.subheader("1. Ingested Document Classification Ledger")
        st.markdown("Displays the AI's Chain-of-Thought reasoning for every uploaded file.")
        
        meta_df = pd.DataFrame(metadata_ledger)
        if not meta_df.empty:
            # Reorder columns to show reasoning first
            meta_df = meta_df[['source_file', 'actual_document_type', 'reasoning', 'is_sh7', 'meeting_date']]
            meta_df.columns = ['Filename', 'Detected Document Type', 'AI Reasoning', 'Is SH-7?', 'Meeting Date']
            st.dataframe(meta_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # --- DISPLAY 2: DRHP CAPITAL STRUCTURE TABLE ---
        st.subheader("2. Draft Authorised Share Capital History")
        if asc_extractions:
            drhp_df = build_drhp_table(asc_extractions)
            st.dataframe(drhp_df, use_container_width=True, hide_index=True)
            
            csv = drhp_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download DRHP Table as CSV",
                data=csv,
                file_name='drhp_capital_structure.csv',
                mime='text/csv',
            )
        else:
            st.warning("No Form SH-7 documents were identified during the metadata extraction pass.")
            
        st.divider()
        
        # --- DISPLAY 3: AUDIT TRAIL ---
        st.subheader("🔍 Audit Trail (SH-7s Only)")
        st.markdown("Raw JSON math extractions from the isolated SH-7 forms.")
        for extraction in asc_extractions:
            with st.expander(f"View source extraction for: {extraction['source_file']}"):
                st.json(extraction)