# NeuroFold 🔬

AI-powered platform to predict potential drug candidates for target proteins, powered by PyTorch, FastAPI, and Gemini.

---

## ✨ Overview
This project enables users to:
- Predict drug-protein binding affinity using ML
- Fetch real-time data from PubChem & UniProt
- Visualize molecular structures and interaction heatmaps
- Get natural-language insights from Gemini AI
- Discover new candidate drugs for any protein target

Deployed via FastAPI with a frontend powered by Next.js.

---

## 🌐 Features

### Core Functionality
- `/predict` - Predict if a given drug binds a protein
- `/discover` - Screen multiple drugs for a given target protein
- Real-time data fetch from public APIs
- Gemini-powered explanations for predictions

### Visualizations
- RDKit-based molecule rendering
- Attention heatmaps of protein-drug interactions
- Top similar compounds via Tanimoto similarity

### Frontend Integration
- Drug cards with images & properties
- Markdown-style Gemini reports
- Async prediction job status for batch jobs

---

## ⚖️ Business Model

### Freemium Approach
- **Free**: Researchers & students, limited usage/month
- **Pro**: Custom model hosting, private datasets, team dashboards

### Monetization Ideas
- API access with pay-per-call pricing
- Premium insights & Gemini reports
- Reports export as PDF/Notebooks

---

## ⚡ Scalability Plan

### Infrastructure
- Dockerized, deployed to cloud (GCP/AWS/Render)
- Async inference for expensive jobs
- Model optimization (quantization, batching)
- Redis caching for PubChem/UniProt calls

### ML & Data
- Expand training data via BindingDB, DrugBank
- Fine-tune on PDBBind or ESM protein embeddings
- Serve multiple model versions

---

## 📽️ Presentation

Check out the project overview deck here: [View Presentation (Canva)](https://www.canva.com/design/DAGjw62AW1E/LHytmVFddrpbi2eiM-nofQ/edit?utm_content=DAGjw62AW1E&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)


---

## 🌍 Contact
- Email: prabhsurat.tech@gmail.com
- Linkedin: [Prabhsurat Singh](https://www.linkedin.com/in/prabhsurat-singh-1868052ab)



<p align="center">Made with ❤️ by Prabhsurat Singh</p>
