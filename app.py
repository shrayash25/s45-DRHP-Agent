import streamlit as st
from ai_agent import extract_sh7_data, build_drhp_table, USE_OLLAMA, MODEL_NAME

# ==========================================
# STREAMLIT FRONTEND
# ==========================================
st.set_page_config(page_title="DRHP Capital Structure Agent", layout="wide")

st.title("📄 DRHP Capital Structure Drafting Agent")
st.markdown("Upload **Form SH-7** documents to automatically generate the Authorised Share Capital history table with complete traceability.")

st.sidebar.header("System Configuration")
st.sidebar.info(f"**Current Mode:** {'Local (Ollama)' if USE_OLLAMA else 'Production (API)'}\n\n**Model:** `{MODEL_NAME}`")

uploaded_files = st.file_uploader("Upload SH-7 Documents (.md, .txt)", accept_multiple_files=True, type=['md', 'txt'])

if st.button("Generate Capital Structure Table", type="primary") and uploaded_files:
    with st.spinner(f"Running 2-Pass Extraction with {MODEL_NAME}..."):
        all_extractions = []
        
        for file in uploaded_files:
            text = file.read().decode("utf-8")
            extracted_data = extract_sh7_data(file.name, text)
            
            if "error" not in extracted_data:
                all_extractions.append(extracted_data)
            else:
                st.error(f"Failed to parse {file.name}: {extracted_data['error']}")
        
        if all_extractions:
            st.success("Extraction Complete!")
            df = build_drhp_table(all_extractions)
            
            st.subheader("Draft Authorised Share Capital History")
            
            # Display full-width interactive table
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # CSV DOWNLOAD BUTTON
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Table as CSV",
                data=csv,
                file_name='drhp_capital_structure.csv',
                mime='text/csv',
            )
            
            st.divider()
            
            st.subheader("🔍 Audit Trail & Traceability")
            st.markdown("Click below to view the raw structured JSON extracted from each source document.")
            for extraction in all_extractions:
                with st.expander(f"View source extraction for: {extraction['source_file']}"):
                    st.json(extraction)