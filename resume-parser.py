# pip install streamlit google.generativeai PyPDF2

import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader

# Replace with your API Key
# URL to get API Key - https://aistudio.google.com/prompts/new_chat
# Get API Key and Paste here
GOOGLE_API_KEY = "xxxxxxxxxxxxxxxxx"
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash-001")

st.title("Resume Analyzer with AI")

resume_file = st.file_uploader("Upload your Resume (pdf)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

# Extract text from pdf
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def analyze_resume(resume_text, jd_text):
    prompt = f"""
    You are a resume analyzer.
    Analyze the following resume and compare it with job description
    Resume: {resume_text}
    Job Descriptio: {jd_text}
    Provide:
    - A list of mistakes in the resume
    - Matching score (0-100)
    - Skill gaps
    - Brief summary of the resume
    - Suggestions for improvement
    """
    respone = model.generate_content(prompt)
    return respone.text

if st.button("Analyze Resume"):
    if resume_file and job_description:
        resume_text = extract_text_from_pdf(resume_file)
        result = analyze_resume(resume_file, job_description)
        st.subheader("Analysis Result")
        st.markdown(result)
    else:
        st.warning("Please upload both resume and job desciption")
