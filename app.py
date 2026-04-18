import streamlit as st
import os
import pickle
import plotly.express as px

from nlp.resume_analyzer import analyze_resume
from engine.recommender import generate_recommendations
from rag.chatbot import CareerChatbot

st.set_page_config(page_title="Career Intelligence System", page_icon="🚀", layout="wide")

# Cache the ML models and Chatbot to prevent reloading on each interaction
@st.cache_resource
def load_ml_resources():
    model_path = os.path.join('ml', 'model.pkl')
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['scaler']
    except FileNotFoundError:
        st.error("ML Model not found. Please train the model first.")
        return None, None

@st.cache_resource
def load_chatbot():
    kb_path = os.path.join('rag', 'career_knowledge.txt')
    try:
        return CareerChatbot(data_path=kb_path)
    except Exception as e:
        st.error(f"Failed to load Chatbot: {e}")
        return None

model, scaler = load_ml_resources()
chatbot = load_chatbot()

st.title("🚀 AI-Powered Career Intelligence System")
st.markdown("A unified platform for Placement Prediction, Resume Scoring, and Career Guidance.")

# Initialize session state for multi-step workflow
if 'profile' not in st.session_state:
    st.session_state.profile = {}
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'resume_analysis' not in st.session_state:
    st.session_state.resume_analysis = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []

tabs = st.tabs(["📊 Profile Predictor", "📄 Resume Analyzer", "💡 Recommendations", "💬 Career Bot"])

with tabs[0]:
    st.header("Step 1: Academic Profile & Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)
        internships = st.number_input("Internships", 0, 10, 1)
        projects = st.number_input("Projects", 0, 10, 2)
        workshops = st.number_input("Workshops/Certifications", 0, 10, 1)
        aptitude = st.slider("Aptitude Test Score", 0, 100, 75)
        
    with col2:
        soft_skills = st.slider("Soft Skills Rating", 0.0, 5.0, 4.0)
        ssc = st.slider("10th Marks (%)", 0, 100, 80)
        hsc = st.slider("12th Marks (%)", 0, 100, 75)
        eca_val = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        training_val = st.selectbox("Placement Training", ["Yes", "No"])
        
    if st.button("Predict Placement Chance"):
        # We need a native profile object for the recommender
        profile = {
            "CGPA": cgpa, "Internships": internships, "Projects": projects, 
            "Workshops/Certifications": workshops, "AptitudeTestScore": aptitude,
            "SoftSkillsRating": soft_skills, "ExtracurricularActivities": eca_val,
            "PlacementTraining": training_val, "SSC_Marks": ssc, "HSC_Marks": hsc
        }
        st.session_state.profile = profile
        
        if model and scaler:
            # Format exactly as model expects
            features = [
                float(cgpa),
                int(internships),
                int(projects),
                int(workshops),
                float(aptitude),
                float(soft_skills),
                1 if eca_val == 'Yes' else 0,
                1 if training_val == 'Yes' else 0,
                float(ssc),
                float(hsc)
            ]
            X_scaled = scaler.transform([features])
            prob = model.predict_proba(X_scaled)[0][1] * 100
            
            st.session_state.prediction = round(prob, 2)
            
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Pie(
                labels=['Placed Chance', 'Risk'], 
                values=[float(st.session_state.prediction), 100.0 - float(st.session_state.prediction)],
                marker=dict(colors=['#00CC96', '#EF553B'])
            )])
            fig.update_layout(title="Placement Probability Breakdown")
            st.plotly_chart(fig, use_container_width=True)
            
            if st.session_state.prediction >= 70:
                st.success(f"Great! High chance of placement: {st.session_state.prediction}%")
            elif st.session_state.prediction >= 50:
                st.warning(f"Moderate chance of placement: {st.session_state.prediction}%")
            else:
                st.error(f"Low chance of placement: {st.session_state.prediction}%. Needs immediate action.")
        else:
            st.error("Model resources failed to load.")

with tabs[1]:
    st.header("Step 2: Resume Analyzer")
    role = st.selectbox("Target Role", ["software_engineering", "machine_learning", "data_science", "web_development"])
    uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    
    if st.button("Analyze Resume"):
        if uploaded_file is not None:
            with st.spinner("Analyzing..."):
                try:
                    # Save PDF to disk temporarily
                    temp_folder = os.path.join(os.path.dirname(__file__), 'tmp')
                    os.makedirs(temp_folder, exist_ok=True)
                    temp_path = os.path.join(temp_folder, uploaded_file.name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Direct function call
                    analysis = analyze_resume(temp_path, target_role=role)
                    st.session_state.resume_analysis = analysis
                    
                    # Cleanup
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    score = analysis.get('score', 0)
                    st.metric("Resume Score", f"{score}/100")
                    
                    st.subheader("Found Skills")
                    st.write(", ".join(analysis.get('found_skills', [])))
                    
                    st.subheader("Missing Minimum Skills")
                    st.write(", ".join(analysis.get('missing_skills', [])))

                except Exception as e:
                    st.error(f"Analysis Exception: {e}")
        else:
            st.warning("Please upload a file.")

with tabs[2]:
    st.header("Step 3: Core Recommendations")
    if st.button("Generate Final Advice"):
        if st.session_state.prediction is None or st.session_state.resume_analysis is None:
            st.warning("Please complete Step 1 (Predictor) and Step 2 (Resume) first.")
        else:
            with st.spinner("Generating Intelligence..."):
                try:
                    # Direct function call
                    recs = generate_recommendations(
                        st.session_state.profile, 
                        st.session_state.prediction, 
                        st.session_state.resume_analysis
                    )
                    st.session_state.recommendations = recs
                    
                    for i, rec in enumerate(recs):
                        st.info(f"{i+1}. {rec}")
                except Exception as e:
                    st.error(f"Engine Exception: {e}")

with tabs[3]:
    st.header("RAG Chatbot Advice")
    st.markdown("Ask anything about standard career transitions, skills, or placement rules.")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help your career today?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("E.g., How to improve placement in ML?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Thinking..."):
            if chatbot:
                try:
                    # Direct function call
                    answer = chatbot.ask_question(prompt)
                    response_text = answer.get("answer", "No answer found")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.chat_message("assistant").write(response_text)
                except Exception as e:
                    st.error(f"Chat Exception: {e}")
            else:
                st.error("Chatbot is not loaded.")
