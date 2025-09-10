# Patient Similarity Search - ML Model with UI

A web application that uses machine learning to find patients with similar medical history based on diagnosis and treatment data.

## Features

- **Excel File Upload**: Upload patient data in Excel format (.xlsx, .xls)
- **AI-Powered Similarity Search**: Uses TF-IDF vectorization and cosine similarity to find similar patients
- **Modern Web Interface**: Beautiful, responsive UI with drag-and-drop file upload
- **Real-time Results**: Instant similarity matching with confidence scores
- **Data Preprocessing**: Automatic text cleaning and normalization

## Supported Data Formats

### Patient Analysis Data Format (Patient_Analysis_Data.xls)
The application is optimized for your patient analysis data with these columns:
- **Appointment Date**: Date of appointment
- **Patient**: Patient name
- **MR No**: Medical record number
- **Diagonosis**: Medical diagnosis (note: typo in original data)
- **Chief Complaint**: Primary complaint/symptoms
- **Past History**: Previous medical history
- **Medicine Prescribed**: Medications and treatments
- **Doctor Advice**: Additional medical advice

### Standard Format
For other Excel files, the following columns are required:
- **Patient**: Patient name or identifier
- **Diagnosis**: Medical diagnosis or condition
- **Treatment**: Treatment plan or medications

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and go to: `http://localhost:5000`

## Usage

1. **Upload Data**: 
   - Click "Upload Patient Data" or drag and drop your Excel file
   - The system will process and load your patient data

2. **Search for Similar Patients**:
   - Enter patient history, symptoms, or diagnosis in the search box
   - Click "Find Similar Patients" to get results
   - Results show similarity percentage and patient details

3. **View Results**:
   - Similar patients are ranked by similarity score
   - Each result shows patient name, diagnosis, and treatment
   - Higher percentage indicates better match

## Sample Data

A sample Excel file (`sample_data.xlsx`) is included with 20 example patients for testing.

## Technical Details

- **Backend**: Flask (Python)
- **ML Algorithm**: TF-IDF Vectorization + Cosine Similarity
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Text Processing**: NLTK for text preprocessing and stopword removal

## API Endpoints

- `POST /upload`: Upload Excel file
- `POST /search`: Search for similar patients
- `GET /data_info`: Get information about loaded data

## Example Search Queries

- "Patient with high blood pressure and heart problems"
- "Diabetes treatment with metformin"
- "Chronic pain management"
- "Mental health conditions requiring therapy"
- "Fever and cough symptoms"
- "Chest pain and breathing difficulties"
- "Headache and nausea"
- "Joint pain and swelling"

## Your Data

The application is now configured to work with your `Patient_Analysis_Data.xls` file containing **13,282 patient records** with comprehensive medical information including:
- Patient demographics
- Medical diagnoses
- Chief complaints
- Past medical history
- Prescribed medications
- Doctor recommendations

The ML model will analyze all these fields to find the most similar patients based on your search query.
