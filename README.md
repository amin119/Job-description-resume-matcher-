# Job Description & Resume Matcher 🔍

A Python-based tool that automatically matches resumes with job descriptions using machine learning, streamlining the recruitment process.

![Screenshot](screenshot.png) *(Add a screenshot later if available)*

## Features ✨
- **Job Description Analysis** - Process any job description text
- **Multi-Resume Upload** - Supports PDF, DOCX, and TXT formats (bulk upload)
- **Smart Matching** - Machine learning-powered similarity scoring
- **Ranked Results** - Displays top candidates with match percentages

## How It Works ⚙️
1. Recruiter inputs a job description
2. Upload multiple resumes (5+ recommended)
3. System analyzes and ranks candidates by fit
4. View results with similarity scores

## Technologies Used 🛠️
- **Python** (Flask backend)
- **scikit-learn** (TF-IDF vectorization & cosine similarity)
- **Bootstrap 5** (Modern responsive UI)
- **NLTK** (Text processing)

## Installation 📦
```bash
# Clone the repository
git clone https://github.com/yourusername/job-resume-matcher.git

# Navigate to project
cd job-resume-matcher

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
