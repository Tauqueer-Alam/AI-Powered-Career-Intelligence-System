# 🚀 AI-Powered Career Intelligence System

An advanced, all-in-one Streamlit monolithic application designed to help students and professionals navigate their career paths through intelligent predictions, automated resume parsing, and actionable AI-driven guidance.

---

## 🌟 Key Features

### 📊 Academic Profile & Prediction
A machine learning module built using `scikit-learn` (Random Forest) that calculates a user's probability of securing a placement based on quantitative (CGPA, Marks) and qualitative (Internships, Extracurriculars) factors. It instantly outputs a beautifully tailored probability visual.

### 📄 Resume Analyzer
A Natural Language Processing tool that automatically parses uploaded PDF resumes (using `PyPDF2`). It cross-references extracted text against industry standard skill requirements for roles like *Machine Learning*, *Data Science*, and *Software Engineering*, scoring the user's proficiency while flagging explicitly missing key skills.

### 💡 Intelligent Recommendations
A deterministic rules engine that combines the user's placement probability and resume score to generate dynamic, concrete advice indicating exactly what the user must prioritize next (e.g. taking foundational algorithms classes or expanding their GitHub portfolio).

### 💬 Career RAG Chatbot
An entirely local generative AI assistant powered by `google/flan-t5-small` and `faiss-cpu`. Using Retrieval-Augmented Generation (RAG), the bot strictly references a provided custom knowledge base (`career_knowledge.txt`) to answer context-specific questions about career transitions and interview preparation. 

---

## 🛠️ Technology Stack
- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Visualizations**: [Plotly Express](https://plotly.com/)
- **Machine Learning Module**: `scikit-learn`, `pandas`, `numpy`
- **Text Processing & NLP**: `PyPDF2`
- **Generative AI / RAG**: `Transformers` (HuggingFace), `Sentence-Transformers`, `FAISS`, `PyTorch`

---

## 🚀 Getting Started Locally

### 1. Requirements
Ensure you have **Python 3.9+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/career-intelligence-system.git
cd career-intelligence-system
```

### 3. Install Dependencies
Install all required packages from the streamlined requirements file:
```bash
pip install -r requirements.txt
```
*(Optionally, do this inside an isolated virtual environment (`venv`))*

### 4. Run the Application
Boot up the Streamlit server directly:
```bash
streamlit run app.py
```
*(The server will start on `http://localhost:8501/` by default)*

---

## ☁️ Deployment (Streamlit Community Cloud)

This framework is highly optimized as a Single Page Application (SPA), meaning it requires no secondary API servers. You can deploy it seamlessly for free directly on **Streamlit Community Cloud**:
1. Push this directory to your GitHub Repository.
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New App**, tie it to your repository branch, and select `app.py` as your Main file.
4. Hit Deploy! Streamlit will automatically fetch the dependencies from `requirements.txt`.
