import PyPDF2
import re

# Ideal skills mapping for different career paths (a simple knowledge base)
CAREER_SKILLS = {
    'software_engineering': ['python', 'java', 'c++', 'javascript', 'sql', 'react', 'node', 'algorithms', 'data structures', 'git'],
    'machine_learning': ['python', 'scikit-learn', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'sql', 'machine learning', 'deep learning'],
    'data_science': ['python', 'r', 'sql', 'pandas', 'numpy', 'matplotlib', 'tableau', 'statistics', 'machine learning'],
    'web_development': ['html', 'css', 'javascript', 'react', 'node', 'express', 'mongodb', 'sql', 'git'],
}

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text.lower() # Convert to lower case for easier matching

def analyze_resume(pdf_path, target_role='machine_learning'):
    text = extract_text_from_pdf(pdf_path)
    
    if not text.strip():
        return {
            "score": 0,
            "found_skills": [],
            "missing_skills": CAREER_SKILLS.get(target_role, []),
            "message": "Could not extract text from the PDF."
        }
        
    ideal_skills = CAREER_SKILLS.get(target_role, CAREER_SKILLS['software_engineering'])
    
    found_skills = []
    missing_skills = []
    
    for skill in ideal_skills:
        # Simple word boundary regex to find exactly the skill name
        # \b ensures we match 'r' only as a word, not inside 'train'
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found_skills.append(skill)
        else:
            missing_skills.append(skill)
            
    # Calculate score based on percentage of ideal skills found
    score = (len(found_skills) / len(ideal_skills)) * 100 if ideal_skills else 0
    
    return {
        "score": round(score, 2),
        "found_skills": found_skills,
        "missing_skills": missing_skills,
        "target_role": target_role
    }

if __name__ == "__main__":
    # Test stub
    pass
