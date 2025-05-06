from flask import Flask, request, render_template
import os
import PyPDF2
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_text(file_path)
    else:
        return None

@app.route('/')
def matchresume():
    return render_template('matchresume.html')

@app.route("/matcher", methods=['GET','POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description')
        resume_files = request.files.getlist('resumes')
        
        # Validate inputs
        if not job_description or not resume_files or all(f.filename == '' for f in resume_files):
            return render_template('matchresume.html', 
                                error="Please provide both a job description and at least one resume file.")
        
        # Process files
        results = []
        for resume_file in resume_files:
            if resume_file.filename == '':
                continue
                
            # Save file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            
            # Extract text
            text = extract_text(filename)
            if not text:
                continue
                
            # Calculate similarity
            vectorizer = TfidfVectorizer().fit_transform([job_description, text])
            vectors = vectorizer.toarray()
            similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
            
            results.append({
                'filename': resume_file.filename,
                'score': round(similarity * 100, 2)
            })
        
        if not results:
            return render_template('matchresume.html', 
                                error="Could not process any resumes (supported formats: PDF, DOCX, TXT).")
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return render_template('matchresume.html',
                            results=results,
                            success="Matching completed successfully!")
    
    return render_template('matchresume.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)