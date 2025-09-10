import nltk
nltk.download('punkt')
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store data and model
patient_data = None
vectorizer = None
tfidf_matrix = None

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

def load_and_process_data(file_path):
    """Load Excel file and process the data"""
    global patient_data, vectorizer, tfidf_matrix
    
    try:
        # Read Excel file with proper header
        if 'Patient_Analysis_Data.xls' in file_path:
            # Use header=4 for the actual patient data file
            df = pd.read_excel(file_path, header=4)
        else:
            # For other files, try to detect the header
            df = pd.read_excel(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check if this is the patient analysis data format
        if 'Patient' in df.columns and 'Diagonosis' in df.columns:
            # This is the patient analysis data format
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
            
            # Create a combined text field for similarity matching using all relevant fields
            df['Combined_Text'] = (
                df['Patient'].astype(str) + ' ' +
                df['Diagnosis'].astype(str) + ' ' +
                df['Chief_Complaint'].astype(str) + ' ' +
                df['Past_History'].astype(str) + ' ' +
                df['Treatment'].astype(str) + ' ' +
                df['Doctor_Advice'].astype(str)
            )
        else:
            # Check if required columns exist for standard format
            required_columns = ['Patient', 'Diagnosis', 'Treatment']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                # Try to find similar column names
                suggestions = {}
                for missing in missing_columns:
                    similar_cols = [col for col in df.columns if missing.lower() in col.lower()]
                    if similar_cols:
                        suggestions[missing] = similar_cols[0]
                
                if suggestions:
                    # Rename columns based on suggestions
                    df = df.rename(columns=suggestions)
                else:
                    return False, f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}"
            
            # Create a combined text field for similarity matching
            df['Combined_Text'] = (
                df['Patient'].astype(str) + ' ' +
                df['Diagnosis'].astype(str) + ' ' +
                df['Treatment'].astype(str)
            )
        
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
        
        patient_data = df
        return True, "Data loaded successfully"
        
    except Exception as e:
        return False, f"Error loading data: {str(e)}"

def find_similar_patients(query_text, top_k=5):
    """Find similar patients based on query text"""
    global patient_data, vectorizer, tfidf_matrix
    
    if patient_data is None or vectorizer is None or tfidf_matrix is None:
        return []
    
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
            if similarities[idx] > 0:  # Only include patients with some similarity
                patient_info = {
                    'patient': patient_data.iloc[idx]['Patient'],
                    'diagnosis': patient_data.iloc[idx]['Diagnosis'],
                    'treatment': patient_data.iloc[idx]['Treatment'],
                    'similarity_score': float(similarities[idx]),
                    'index': int(idx)
                }
                
                # Add additional fields if they exist in the patient analysis data
                if 'Chief_Complaint' in patient_data.columns:
                    patient_info['chief_complaint'] = patient_data.iloc[idx]['Chief_Complaint']
                if 'Past_History' in patient_data.columns:
                    patient_info['past_history'] = patient_data.iloc[idx]['Past_History']
                if 'Doctor_Advice' in patient_data.columns:
                    patient_info['doctor_advice'] = patient_data.iloc[idx]['Doctor_Advice']
                if 'Appointment_Date' in patient_data.columns:
                    patient_info['appointment_date'] = patient_data.iloc[idx]['Appointment_Date']
                if 'MR_No' in patient_data.columns:
                    patient_info['mr_no'] = patient_data.iloc[idx]['MR_No']
                
                results.append(patient_info)
        
        return results
        
    except Exception as e:
        print(f"Error in similarity search: {str(e)}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and file.filename.endswith(('.xlsx', '.xls')):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the uploaded file
        success, message = load_and_process_data(file_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'patient_count': len(patient_data)
            })
        else:
            return jsonify({'success': False, 'message': message})
    
    return jsonify({'success': False, 'message': 'Invalid file format. Please upload an Excel file.'})

@app.route('/search', methods=['POST'])
def search_similar():
    data = request.get_json()
    query = data.get('query', '')
    
    if not query:
        return jsonify({'success': False, 'message': 'Please enter a search query'})
    
    if patient_data is None:
        return jsonify({'success': False, 'message': 'Please upload a data file first'})
    
    results = find_similar_patients(query)
    
    return jsonify({
        'success': True,
        'results': results,
        'query': query
    })

@app.route('/data_info')
def data_info():
    if patient_data is None:
        return jsonify({'success': False, 'message': 'No data loaded'})
    
    return jsonify({
        'success': True,
        'patient_count': len(patient_data),
        'columns': list(patient_data.columns)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
