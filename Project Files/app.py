import streamlit as st
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import GooglePalm
from langchain_community.embeddings import GooglePalmEmbeddings
import google.generativeai as genai  # ✅ Google AI API
import docx
from dotenv import load_dotenv
from fpdf import FPDF

# Load API Key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY is missing! Please set it in a .env file or environment variables.")
    st.stop()

# ✅ Set API Key for Google AI
genai.configure(api_key=GOOGLE_API_KEY)

# ✅ Extract text from PDFs with page-wise logging
def extract_text_from_pdfs(pdf_docs):
    text_ = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            extracted = page.extract_text()
            if extracted:
                text_.append(f"Page {i+1}:\n{extracted}\n")
                st.write(f"📄 Extracted from Page {i+1}:")
                st.write(extracted[:500])  # Display first 500 chars
    return "\n".join(text_)

# ✅ Extract text from Word documents
def extract_text_from_word(docx_files):
    text = ""
    for doc in docx_files:
        doc_reader = docx.Document(doc)
        for para in doc_reader.paragraphs:
            text += para.text + "\n"
    return text

# ✅ Process price lists
def process_price_lists(pdf_docs):
    return extract_text_from_pdfs(pdf_docs)

# ✅ Generate text embeddings using Google's `text-embedding-gecko`
def get_text_embedding(text):
    model = "models/embedding-001"  # Google's embedding model
    response = genai.embed_content(model=model, content=text, task_type="retrieval_document")
    return response["embedding"]

# ✅ Research Paper Summarization with Summary Length Options
def summarize_research_paper(text, summary_length="medium"):
    if not text.strip():
        return "❌ No valid text to summarize."

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")  

    length_prompt = {
        "short": "Give a very brief summary (2-3 sentences).",
        "medium": "Give a concise summary with key points (5-7 sentences).",
        "detailed": "Provide a detailed summary (10+ sentences) with insights."
    }

    prompt = f"""
    You are an AI assistant specializing in summarizing research papers.
    {length_prompt[summary_length]}

    **Research Paper:**  
    {text[:4000]}  # Limit input size to avoid token overflow

    Provide a structured summary with key points.
    """

    try:
        response = model.generate_content(prompt)  # ✅ Correct method

        return response.text if response else "No summary generated."

    except Exception as e:
        st.error(f"❌ Error in summarizing research paper: {str(e)}")
        return "Summarization failed."

# ✅ Save summary as PDF
def save_summary_as_pdf(summary_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)

    file_path = "summary.pdf"
    pdf.output(file_path)
    return file_path

# ✅ Resume Matching with AI
def match_resumes(pdf_docs, job_description):
    extracted_resumes = extract_text_from_pdfs(pdf_docs)

    if not extracted_resumes.strip():
        st.error("❌ No text extracted from resumes.")
        return "No resumes found."

    model = genai.GenerativeModel('models/gemini-1.5-flash')  # ✅ Faster option

    prompt = f"""
    You are an AI assistant specializing in resume screening.
    Match the following job description with the most relevant resumes from the extracted text:

    **Job Description:**  
    "{job_description}"

    **Extracted Resume Data:**  
    {extracted_resumes[:3000]}  # Limit characters to avoid token overflow

    Provide a ranked list of the best-matching candidates and explain why they are a good fit.
    """

    try:
        response = model.generate_content(prompt)  # ✅ Correct method for Gemini API

        return response.text if response else "No matching resumes found."

    except Exception as e:
        st.error(f"❌ Error in resume matching: {str(e)}")
        return "Resume matching failed."

# ✅ Main Streamlit App
def main():
    st.set_page_config(page_title="DocuQuery: AI-Powered PDF Knowledge Assistant")
    st.header("DocuQuery: AI-Powered PDF Knowledge Assistant")

    option = st.sidebar.selectbox("Choose a Feature", ["Price List Analyzer", "Research Paper Summarizer", "Resume Matcher for Hiring"])
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
    job_desc = ""

    if option == "Resume Matcher for Hiring":
        job_desc = st.text_area("Enter Job Description")

    summary_length = "medium"
    if option == "Research Paper Summarizer":
        summary_length = st.sidebar.radio("Summary Length", ["short", "medium", "detailed"], index=1)

    if st.button("Process"):
        if not pdf_docs:
            st.error("❌ Please upload PDF files.")
            return
        
        with st.spinner("Processing..."):
            if option == "Price List Analyzer":
                extracted_text = process_price_lists(pdf_docs)
                st.write("📑 Extracted Price List:")
                st.write(extracted_text[:1000])  # Display first 1000 chars
            
            elif option == "Research Paper Summarizer":
                extracted_text = extract_text_from_pdfs(pdf_docs)
                summary = summarize_research_paper(extracted_text, summary_length)
                st.write("📑 Research Paper Summary:")
                st.write(summary.replace("•", "🔹"))  # Bullet formatting
                
                # ✅ PDF Download Option
                pdf_file_path = save_summary_as_pdf(summary)
                with open(pdf_file_path, "rb") as file:
                    st.download_button(label="📥 Download Summary as PDF", data=file, file_name="Research_Summary.pdf", mime="application/pdf")

            elif option == "Resume Matcher for Hiring":
                if not job_desc.strip():
                    st.error("❌ Please enter a job description.")
                    return
                match_result = match_resumes(pdf_docs, job_desc)
                st.write("📌 Matched Resumes:")
                st.write(match_result)

if __name__ == "__main__":
    main()
