def generate_recommendations(profile_data, prediction_prob, resume_analysis):
    suggestions = []
    
    cgpa = float(profile_data.get("CGPA", 0))
    internships = int(profile_data.get("Internships", 0))
    projects = int(profile_data.get("Projects", 0))
    
    score = resume_analysis.get('score', 0)
    missing_skills = resume_analysis.get('missing_skills', [])
    target_role = resume_analysis.get('target_role', 'software_engineering')
    
    # Heuristic 1: Academic Performance
    if cgpa < 7.0:
        suggestions.append("Your CGPA is below average (7.0). Work on increasing your academic scores or compensate heavily with advanced certifications.")
    elif cgpa >= 8.5:
        suggestions.append("Great CGPA! You have a solid academic foundation. Maintain this to cross major company thresholds.")
        
    # Heuristic 2: Real-world Experience
    if internships == 0:
        suggestions.append("Lack of internships is a major flag. Apply for summer internships or contribute to open-source projects aggressively to build experience.")
    elif internships >= 2:
        suggestions.append("Strong internship experience! Make sure to highlight the impact of your contributions on your resume.")
        
    # Heuristic 3: Projects
    if projects < 2:
        suggestions.append("You have fewer than 2 projects. Build full-stack or end-to-end ML projects to demonstrate practical knowledge.")
        
    # Heuristic 4: Placement Probability ML Output
    if prediction_prob < 50:
        suggestions.append(f"Your model-predicted placement probability is {prediction_prob:.1f}%. Immediate action is required to upskill.")
    elif prediction_prob >= 80:
        suggestions.append(f"Excellent placement probability ({prediction_prob:.1f}%). Focus on interview preparation and advanced data structures.")
        
    # Heuristic 5: Resume NLP Feedback
    if score < 50:
        suggestions.append(f"Your resume matches only {score}% of the ideal keywords for {target_role.replace('_', ' ')}. Reword your resume to better reflect the job description.")
    
    if len(missing_skills) > 0:
        skills_str = ", ".join(missing_skills[:3])
        suggestions.append(f"Consider learning or adding these missing key skills: {skills_str}.")

    # Fallback
    if len(suggestions) < 3:
        suggestions.append("Participate in mock interviews and practice coding tasks on LeetCode/HackerRank daily.")
        
    return suggestions
