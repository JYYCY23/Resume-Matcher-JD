import streamlit as st
import pdfplumber
import docx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# Cache the vectorizer to improve performance
@st.cache_resource
def get_vectorizer():
    return TfidfVectorizer(stop_words='english')

# Improved text extraction with error handling
def extract_text(file):
    try:
        if file.name.endswith(".pdf"):
            with pdfplumber.open(file) as pdf:
                text = " ".join([page.extract_text() or "" for page in pdf.pages])
                return text.strip()
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            return " ".join([para.text for para in doc.paragraphs]).strip()
        else:
            return ""
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return ""

# Function to create downloadable DataFrame
def create_downloadable_data(results):
    df = pd.DataFrame(results, columns=["Rank", "Document", "Match Score"])
    df.set_index("Rank", inplace=True)
    return df.to_csv().encode('utf-8')

# Streamlit UI
st.set_page_config(page_title="AI Resume Matcher", page_icon="📄", layout="wide")
st.title("📄 AI Resume Matcher")
st.markdown("""
    <style>
        .reportview-container {margin-top: -2em;}
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        .stToolbar {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# File upload section
with st.expander("📤 Upload Resumes and Job Description", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        resumes = st.file_uploader("Upload resumes (PDF/DOCX)", 
                                 type=["pdf", "docx"], 
                                 accept_multiple_files=True,
                                 help="Upload multiple resumes in PDF or DOCX format")
    with col2:
        job_description = st.text_area("Paste Job Description:", 
                                     height=200,
                                     help="Enter the job description text to compare against resumes")

# Processing and results section
if resumes and job_description and st.button("🚀 Start Matching"):
    with st.spinner("🔍 Analyzing documents..."):
        # Initialize vectorizer
        vectorizer = get_vectorizer()
        
        # Process documents with progress bar
        progress_bar = st.progress(0)
        valid_resumes = []
        resume_texts = []
        
        for i, resume in enumerate(resumes):
            progress_bar.progress((i+1)/len(resumes), f"Processing {resume.name}...")
            text = extract_text(resume)
            if text:
                valid_resumes.append(resume)
                resume_texts.append(text)
        
        if not resume_texts:
            st.error("No valid text extracted from uploaded resumes!")
            st.stop()
        
        # Vectorize documents
        try:
            job_desc_texts = [job_description] + resume_texts
            tfidf_matrix = vectorizer.fit_transform(job_desc_texts)
            similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        except Exception as e:
            st.error(f"Error in document processing: {str(e)}")
            st.stop()
        
        # Create results dataframe
        results = []
        for name, score in zip([r.name for r in valid_resumes], similarity_scores):
            results.append({
                "Document": name,
                "Match Score": f"{score * 100:.2f}%",
                "Raw Score": score
            })
        
        # Sort results by score
        sorted_results = sorted(results, key=lambda x: x['Raw Score'], reverse=True)
        
        # Display results
        st.success(f"✅ Analysis complete! Processed {len(sorted_results)} resumes")
        st.subheader("📊 Matching Results")
        
        # Create two columns layout
        left_col, right_col = st.columns([3, 1])
        
        with left_col:
            # Display results in a table format
            for idx, result in enumerate(sorted_results, 1):
                score_color = "green" if result['Raw Score'] > 0.5 else "orange" if result['Raw Score'] > 0.3 else "red"
                st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; margin: 10px 0; 
                                box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h4 style="margin:0;">🏅 #{idx} {result['Document']}</h4>
                        <h2 style="color: {score_color}; margin: 5px 0;">{result['Match Score']}</h2>
                        <progress value="{result['Raw Score']}" max="1" style="width: 100%; height: 8px;"></progress>
                    </div>
                """, unsafe_allow_html=True)
        
        with right_col:
            # Add download button
            st.markdown("### 📥 Download Results")
            csv_data = create_downloadable_data(
                [(i+1, res['Document'], res['Match Score']) for i, res in enumerate(sorted_results)]
            )
            
            st.download_button(
                label="Export as CSV",
                data=csv_data,
                file_name="resume_matching_results.csv",
                mime="text/csv",
                help="Download results in CSV format"
            )
            
            # Add score interpretation guide
            st.markdown("""
                ### 🧠 Score Guide
                - 🟢 70-100%: Strong match
                - 🟠 40-69%: Moderate match
                - 🔴 0-39%: Weak match
            """)

elif not resumes or not job_description:
    st.warning("⚠️ Please upload resumes and enter a job description to start analysis!")
