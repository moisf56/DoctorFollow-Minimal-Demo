# DoctorFollow Medical Search Demo

**Turkish Medical Document Search and Dose Calculation System**

A demonstration project showcasing an intelligent medical document search system with RAG (Retrieval-Augmented Generation) and pediatric dose calculation for Turkish healthcare professionals.

## Features

- **PDF Upload & Indexing** - Upload Turkish medical PDFs for instant searchability
- **Hybrid Search** - BM25 (lexical) + Semantic search with RRF fusion
- **Conversational Memory** - Context-aware follow-up questions
- **Source Citations** - Vancouver-style medical citations [1], [2], [3]
- **Dose Calculator** - Pediatric drug dosage calculator with safety checks
- **Turkish Optimized** - Special handling for Turkish medical terminology

## Technology Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| **LLM** | AWS Bedrock - Llama 3.1 8B Instruct | Free Tier |
| **Embeddings** | intfloat/e5-small-v2 (local) | Free |
| **Search** | BM25 + Semantic + RRF Fusion | Free |
| **Interface** | Gradio | Free |
| **Deployment** | Hugging Face Spaces | Free |

**Total Cost: $0** 🎉

## 📋 Prerequisites

- Python 3.9+
- AWS Account with Bedrock access
- 4GB+ RAM (for embedding model)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd DoctorFollow-Minimal-Demo

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

```bash
# Copy the example env file
copy .env.example .env  # Windows
# OR
cp .env.example .env    # Mac/Linux

# Edit .env and add your AWS credentials
# AWS_ACCESS_KEY_ID=your_key
# AWS_SECRET_ACCESS_KEY=your_secret
# AWS_DEFAULT_REGION=us-east-1
```

**Getting AWS Credentials:**
1. Go to [AWS Console](https://console.aws.amazon.com/)
2. Navigate to IAM → Users → Your User → Security Credentials
3. Create Access Key
4. Enable Bedrock access in your region

### 3. Download Sample PDF

Download a Turkish medical guideline from:
- [T.C. Sağlık Bakanlığı](https://hsgm.saglik.gov.tr/)
- Place PDF in `sample_data/` folder

### 4. Run the Application

```bash
python app.py
```

Open your browser to: **http://localhost:7860**

## 📖 Usage Guide

### Uploading a PDF

1. Click on **"📄 Doküman Arama"** tab
2. Click **"PDF Dosyası Yükle"**
3. Select your Turkish medical PDF
4. Click **"📤 Yükle ve İndeksle"**
5. Wait for indexing to complete (~1-2 min for 50-page PDF)

### Asking Questions

**Example Questions (Turkish):**
```
Çocuklarda parasetamol dozajı nedir?
Amoksisilin ne zaman kullanılır?
Bu ilacın yan etkileri nelerdir?
```

**Follow-up Questions:**
```
Bu dozajı kaç saatte bir vermem gerekir?
Maksimum günlük doz nedir?
```

The system will respond with **cited answers** like:
> Çocuklarda parasetamol dozu 10-15 mg/kg/doz şeklinde uygulanır [1]. Her 4-6 saatte bir tekrarlanabilir [2].

### Using the Dose Calculator

1. Go to **"💊 Doz Hesaplama"** tab
2. Select drug: Amoksisilin, Parasetamol, or İbuprofen
3. Enter patient weight (kg) and age (years)
4. Click **"🧮 Dozu Hesapla"**

Example:
- Drug: Amoksisilin
- Weight: 25 kg
- Age: 7 years
- **Result:** 312.5 mg, twice daily

## 📁 Project Structure

```
doctorfollow-demo/
├── app.py                 # Gradio UI interface
├── rag.py                 # RAG system core
├── utils.py               # Helper functions (chunking, citations)
├── dose_calculator.py     # Drug dosage calculator
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore rules
├── README.md             # This file
└── sample_data/          # PDF storage
    └── README.md         # Data folder guide
```

## 🧪 Testing

Test the dose calculator:

```bash
python dose_calculator.py
```

Test the RAG system (requires AWS credentials):

```bash
python rag.py
```

## 🚢 Deploying to Hugging Face Spaces

### 1. Create Space

1. Go to [Hugging Face](https://huggingface.co/)
2. Create new Space
   - Name: `doctorfollow-demo`
   - SDK: **Gradio**
   - Hardware: **CPU Basic (free)**

### 2. Upload Files

Upload these files to your Space:
- `app.py`
- `rag.py`
- `utils.py`
- `dose_calculator.py`
- `requirements.txt`
- `README.md`

### 3. Add Secrets

In Space Settings → Repository Secrets, add:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

### 4. Launch

Wait 5-10 minutes for build. Your app will be live!

## 🎯 Key Features Explained

### 1. Hybrid Search (BM25 + Semantic)

- **BM25:** Lexical matching for exact term matching
- **Semantic:** e5-small-v2 embeddings for meaning-based search
- **RRF Fusion:** Combines both rankings for optimal results

### 2. Citation System

Based on RAG best practices:
- **Vancouver Style:** Numeric citations [1], [2], [3]
- **Validation:** Checks citations reference valid sources
- **Grounding:** Verifies claims exist in source documents

### 3. Turkish Language Optimization

- Turkish character handling (ğüşıöçĞÜŞİÖÇ)
- Turkish stop words filtering
- Turkish medical term recognition
- Adjusted similarity thresholds for Turkish agglutination

### 4. Safety Features

- Dose range validation
- Age/weight restrictions
- Maximum dose warnings
- Clinical disclaimer on all calculations

## ⚠️ Important Disclaimers

**Medical Disclaimer:**
This system is for **demonstration and educational purposes only**.

- NOT intended for actual clinical use
- NOT a substitute for professional medical advice
- All dose calculations must be verified by a physician
- Always consult healthcare professionals for patient care

**Data Privacy:**
- Only use publicly available medical guidelines
- Do not upload patient records or confidential data

## 🐛 Troubleshooting

### AWS Bedrock Error

```
⚠️ AWS Bedrock bağlantısı mevcut değil
```

**Solution:**
1. Check `.env` file has correct AWS credentials
2. Verify Bedrock is enabled in your AWS region
3. Ensure your IAM user has Bedrock permissions

### Model Download Slow

```
📦 Loading embedding model (e5-small-v2)...
```

**Solution:**
First run downloads ~400MB model. Subsequent runs are fast.

### PDF Extraction Failed

```
❌ Hata: PDF extraction failed
```

**Solution:**
- Ensure PDF is not password-protected
- Use text-based PDFs (not scanned images)
- Try a smaller PDF first

## 📊 Performance

- **PDF Indexing:** ~1-2 minutes for 50-page document
- **Query Response:** ~2-4 seconds with Bedrock
- **Memory Usage:** ~2GB RAM (embedding model)
- **Concurrent Users:** Supports multiple users on free tier

## 🔄 Development Timeline

This project was built in **2 hours** following the plan in [To-Dos.txt](To-Dos.txt):

- ⏱️ **Setup (15 min):** Environment, dependencies
- ⏱️ **Core Development (60 min):** RAG system, dose calculator
- ⏱️ **UI Development (30 min):** Gradio interface
- ⏱️ **Testing (15 min):** Local testing, bug fixes

## 🤝 Contributing

This is a demo project, but contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - See LICENSE file for details

## 👥 Authors

**DoctorFollow Team**
- Built as a 2-hour demo project
- Showcasing RAG capabilities for medical search

## 🔗 Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Gradio Documentation](https://gradio.app/docs/)
- [T.C. Sağlık Bakanlığı](https://hsgm.saglik.gov.tr/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Version:** 1.0.0
**Last Updated:** 2025
**Status:** Demo Project ✅
