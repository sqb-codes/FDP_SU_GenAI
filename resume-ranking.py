# pip install streamlit google.generativeai pdfminer.six docx2txt spacy nltk scikit-learn

import streamlit as st
import google.generativeai as genai
import pdfminer.high_level
import docx2txt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

GOOGLE_API_KEY = "AIzaSyAlLoonYbwmFBu4s3YykRKMGGZNoa9VZeE"
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash-001")

def extract_pdf(file_path):
    if file_path.name.endswith(".pdf"):
        return pdfminer.high_level.extract_text(file_path)
    elif file_path.name.endswith(".docx"):
        return docx2txt.process(file_path)
    
    return None

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return " ".join(set(keywords))  # remove duplicates - type cast into set

def extract_resume_details(text):
    prompt = f"""
    Extract key details from the following resume:
    {text}
    Identify Name, Email, Phone, Skills and work Experience as JSON
    """
    response = genai.generate_text(prompt)
    return response.text

# calculate simialrity between job description and resumes
def calc_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    return cosine_similarity(vectors[0], vectors[1][0][0])


st.set_page_config(layout="wide")
st.title("AI-powered resume screening & ranking")

# Two column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    job_desc = st.text_area("Enter Job Description Here", height=300)

with col2:
    st.subheader("Ranked Resumes")

uploaded_files = st.file_uploader("Upload Resumes (PDF/DOCX)", 
                                  type=["pdf", "docx"],
                                  accept_multiple_files=True)


if st.button("Rank Resumes"):
    if not job_desc:
        st.error("Please enter a job description")
    elif not uploaded_files:
        st.error("Please upload at least one resume")
    else:
        result = []
        with col2:
            for file in uploaded_files:
                resume_text = extract_pdf(file)
                resume_keywords = extract_keywords(resume_text)
                job_keywords = extract_keywords(job_desc)

                match_score = calc_similarity(resume_keywords, job_keywords)
                result.append((file.name, match_score, resume_keywords))

            result.sort(key=lambda x: x[1], reverse=True)

            for rank, (name, score, keywords) in enumerate(result, 1):
                st.markdown(f"{rank}, {name} - Match score: `{score}`")
                with st.expander(f"View Extracted Details : {name}"):
                    st.write("Extracted Keywords & Skills : ")
                    st.write(keywords)
