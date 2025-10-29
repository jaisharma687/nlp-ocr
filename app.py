import os
import streamlit as st
from inference import extract_calendar_json, build_context_lines, load_model, answer_question_from_context


st.set_page_config(page_title="Academic Calendar QA (SLM)", page_icon="📅", layout="centered")
st.title("📅 Academic Calendar Interpretation (Compact SLM)")
st.caption("Answers are generated from the uploaded file only. No external LLMs used.")

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model("model/saved_model.pt")

model, tokenizer = get_model()

uploaded = st.file_uploader("Upload a calendar PDF or image", type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"]) 

if uploaded:
    temp_dir = ".tmp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded.name)
    with open(file_path, "wb") as f:
        f.write(uploaded.read())

    with st.spinner("Extracting table and OCR..."):
        table_data = extract_calendar_json(file_path)
        context_lines = build_context_lines(table_data)

    st.success("File processed. Ask a question about this calendar.")
    st.text_area("Preview (first lines)", "\n".join(context_lines[:10]), height=160)

    question = st.text_input("Your question", placeholder="e.g., When is the last date to drop a course?")
    if question:
        with st.spinner("Generating answer from the uploaded file context..."):
            answer = answer_question_from_context(question, context_lines, model, tokenizer)
        st.subheader("Answer")
        st.write(answer)

else:
    st.info("Upload a file to get started.")
