import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="Patient Similarity Search",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .patient-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .similarity-score {
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_patient_data():
    """Load and process patient data"""
    try:
        # Load the data
        df = pd.read_excel('Patient_Analysis_Data.xls', header=4)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Rename columns to standard format
        column_mapping = {
            'Patient': 'Patient',
            'Diagonosis': 'Diagnosis',  # Fix the typo
            'Chief Complaint': 'Chief_Complaint',
            'Past History': 'Past_History',
            'Medicine Prescribed': 'Treatment',
            'Doctor Advice': 'Doctor_Advice',
            'MR No': 'MR_No',
            'Appointment Date': 'Appointment_Date'
        }
        df = df.rename(columns=column_mapping)
        
        # Create combined text field for similarity matching
        df['Combined_Text'] = (
            df['Patient'].astype(str) + ' ' +
            df['Diagnosis'].astype(str) + ' ' +
            df['Chief_Complaint'].astype(str) + ' ' +
            df['Past_History'].astype(str) + ' ' +
            df['Treatment'].astype(str) + ' ' +
            df['Doctor_Advice'].astype(str)
        )
        
        return df, True, "Data loaded successfully"
        
    except Exception as e:
        return None, False, f"Error loading data: {str(e)}"

def preprocess_text(text):
    """Preprocess text for better similarity matching"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return ' '.join(tokens)

@st.cache_data
def create_similarity_model(df):
    """Create TF-IDF model for similarity search"""
    # Preprocess the combined text
    df['Processed_Text'] = df['Combined_Text'].apply(preprocess_text)
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8
    )
    
    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(df['Processed_Text'])
    
    return vectorizer, tfidf_matrix

def find_similar_patients(query_text, df, vectorizer, tfidf_matrix, top_k=10):
    """Find similar patients based on query text"""
    try:
        # Preprocess the query
        processed_query = preprocess_text(query_text)
        
        # Transform query using the same vectorizer
        query_vector = vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get top similar patients
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                patient_info = {
                    'patient': df.iloc[idx]['Patient'],
                    'diagnosis': df.iloc[idx]['Diagnosis'],
                    'treatment': df.iloc[idx]['Treatment'],
                    'chief_complaint': df.iloc[idx]['Chief_Complaint'],
                    'past_history': df.iloc[idx]['Past_History'],
                    'doctor_advice': df.iloc[idx]['Doctor_Advice'],
                    'appointment_date': df.iloc[idx]['Appointment_Date'],
                    'mr_no': df.iloc[idx]['MR_No'],
                    'similarity_score': float(similarities[idx])
                }
                results.append(patient_info)
        
        return results
        
    except Exception as e:
        st.error(f"Error in similarity search: {str(e)}")
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Patient Similarity Search</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Medical Case Similarity Analysis")
    
    # Load data
    with st.spinner("Loading patient data..."):
        df, success, message = load_patient_data()
    
    if not success:
        st.error(f"❌ {message}")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📊 Analytics", "📋 Data Overview", "ℹ️ About"])
    
    with tab1:
        st.markdown("### Search for Similar Patients")
        
        # Create similarity model
        with st.spinner("Building similarity model..."):
            vectorizer, tfidf_matrix = create_similarity_model(df)
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "Enter patient symptoms, diagnosis, or medical history:",
                placeholder="e.g., fever and cough, chest pain, diabetes medication, headache nausea...",
                height=100
            )
        
        with col2:
            st.markdown("**Search Options**")
            top_k = st.slider("Number of results:", 1, 20, 10)
            min_similarity = st.slider("Minimum similarity:", 0.0, 1.0, 0.1, 0.05)
        
        if st.button("🔍 Search Similar Patients", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Searching for similar patients..."):
                    results = find_similar_patients(query, df, vectorizer, tfidf_matrix, top_k)
                
                # Filter by minimum similarity
                results = [r for r in results if r['similarity_score'] >= min_similarity]
                
                if results:
                    st.success(f"✅ Found {len(results)} similar patient(s)")
                    
                    # Display results
                    for i, result in enumerate(results, 1):
                        similarity_pct = (result['similarity_score'] * 100)
                        
                        with st.expander(f"Patient {i}: {result['patient']} - {similarity_pct:.1f}% Match", expanded=True):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**👤 Patient:** {result['patient']}")
                                if result['mr_no'] and str(result['mr_no']) != 'nan':
                                    st.markdown(f"**📋 MR Number:** {result['mr_no']}")
                                if result['appointment_date'] and str(result['appointment_date']) != 'nan':
                                    st.markdown(f"**📅 Appointment Date:** {result['appointment_date']}")
                                
                                st.markdown(f"**🩺 Diagnosis:** {result['diagnosis']}")
                                
                                if result['chief_complaint'] and str(result['chief_complaint']) != 'nan':
                                    st.markdown(f"**⚠️ Chief Complaint:** {result['chief_complaint']}")
                                
                                if result['past_history'] and str(result['past_history']) != 'nan':
                                    st.markdown(f"**📚 Past History:** {result['past_history']}")
                                
                                if result['treatment'] and str(result['treatment']) != 'nan':
                                    st.markdown(f"**💊 Treatment:** {result['treatment']}")
                                
                                if result['doctor_advice'] and str(result['doctor_advice']) != 'nan':
                                    st.markdown(f"**👨‍⚕️ Doctor Advice:** {result['doctor_advice']}")
                            
                            with col2:
                                st.markdown(f'<div class="similarity-score">{similarity_pct:.1f}% Match</div>', unsafe_allow_html=True)
                else:
                    st.warning("No similar patients found with the given criteria. Try adjusting the minimum similarity threshold.")
            else:
                st.warning("Please enter a search query.")
    
    with tab2:
        st.markdown("### Data Analytics")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", f"{len(df):,}")
        
        with col2:
            unique_diagnoses = df['Diagnosis'].nunique()
            st.metric("Unique Diagnoses", f"{unique_diagnoses:,}")
        
        with col3:
            # Count non-null treatments
            treatments_count = df['Treatment'].notna().sum()
            st.metric("Patients with Treatment", f"{treatments_count:,}")
        
        with col4:
            # Count non-null complaints
            complaints_count = df['Chief_Complaint'].notna().sum()
            st.metric("Patients with Complaints", f"{complaints_count:,}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top 10 Diagnoses")
            diagnosis_counts = df['Diagnosis'].value_counts().head(10)
            fig = px.bar(
                x=diagnosis_counts.values,
                y=diagnosis_counts.index,
                orientation='h',
                title="Most Common Diagnoses",
                labels={'x': 'Number of Patients', 'y': 'Diagnosis'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Diagnosis Distribution")
            diagnosis_counts = df['Diagnosis'].value_counts()
            fig = px.pie(
                values=diagnosis_counts.head(10).values,
                names=diagnosis_counts.head(10).index,
                title="Top 10 Diagnoses Distribution"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Data Overview")
        
        # Data info
        st.markdown(f"**Dataset Information:**")
        st.info(f"""
        - **Total Records:** {len(df):,}
        - **Columns:** {len(df.columns)}
        - **Date Range:** {df['Appointment_Date'].min()} to {df['Appointment_Date'].max()}
        - **Unique Patients:** {df['Patient'].nunique():,}
        """)
        
        # Show sample data
        st.markdown("**Sample Data:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column information
        st.markdown("**Column Information:**")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)
    
    with tab4:
        st.markdown("### About This Application")
        
        st.markdown("""
        #### 🏥 Patient Similarity Search System
        
        This application uses **Machine Learning** to find patients with similar medical histories based on:
        - Medical diagnoses
        - Chief complaints
        - Past medical history
        - Treatment plans
        - Doctor advice
        
        #### 🔧 Technical Details
        
        - **Algorithm:** TF-IDF Vectorization + Cosine Similarity
        - **Text Processing:** NLTK for preprocessing and stopword removal
        - **Framework:** Streamlit for the web interface
        - **Data Source:** Patient Analysis Data (13,281 records)
        
        #### 🎯 How It Works
        
        1. **Text Preprocessing:** Cleans and normalizes medical text
        2. **Feature Extraction:** Converts text to numerical vectors using TF-IDF
        3. **Similarity Calculation:** Uses cosine similarity to find similar cases
        4. **Ranking:** Returns results sorted by similarity score
        
        #### 💡 Usage Tips
        
        - Enter symptoms, diagnoses, or medical history in the search box
        - Adjust the number of results and minimum similarity threshold
        - Use specific medical terms for better results
        - Try different combinations of symptoms and conditions
        
        #### 📊 Data Statistics
        
        - **Total Patients:** 13,281
        - **Unique Diagnoses:** Multiple medical conditions
        - **Time Period:** Comprehensive medical records
        - **Fields:** Patient info, diagnosis, complaints, history, treatment, advice
        """)

if __name__ == "__main__":
    main()
