# 📰 AI Powered Fake News Detection using Fine-Tuned BERT Transformer

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-Educational-green?style=for-the-badge)

</p>

---

## 📌 Project Overview

The **AI Powered Fake News Detection System** is an intelligent web application that detects whether a news article is **Fake** or **Real** using a **Fine-Tuned BERT (Bidirectional Encoder Representations from Transformers)** model.

The application provides an easy-to-use interface where users can paste any news article and instantly receive:

- ✅ Prediction (Fake / Real)
- 📊 Confidence Score
- ⚡ Fast AI-powered inference

The project demonstrates the practical application of **Natural Language Processing (NLP)** and **Deep Learning** for misinformation detection.

---

# ✨ Features

- 📰 Detects Fake and Real News
- 🤖 Fine-Tuned BERT Transformer Model
- 📈 Confidence Score Prediction
- 🎨 Modern Streamlit User Interface
- ⚡ Fast Real-Time Prediction
- 📱 Responsive Layout
- 💻 Easy to Use

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Hugging Face Transformers | BERT Model |
| Streamlit | Web Application |
| Scikit-learn | Dataset Splitting & Evaluation |
| Pandas | Data Processing |
| NumPy | Numerical Computation |

---

# 🧠 Machine Learning Model

**Model Used**

- BERT Base Uncased
- Fine-Tuned for Binary Text Classification

**Classification**

- Fake News
- Real News

**Framework**

- Hugging Face Transformers
- PyTorch

---

# 📂 Project Structure

```
AI_Powered_Fake_News_Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Train_BERT.ipynb
│   ├── 05_Evaluation.ipynb
│   └── 06_prediction.ipynb
│
├── reports/
├── outputs/
└── src/
```

---

# 📊 Dataset

Dataset used:

**Fake and Real News Dataset**

Source:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset contains two classes:

- Fake News
- True News

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Ankita0789/AI_Powered_Fake_News_Detection.git
```

Go inside the project

```bash
cd AI_Powered_Fake_News_Detection
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate Virtual Environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit App

```bash
streamlit run app.py
```

---

# 🎯 How It Works

1. User pastes a news article.
2. The text is tokenized using the BERT tokenizer.
3. The fine-tuned BERT model processes the text.
4. The model predicts whether the article is Fake or Real.
5. The confidence score is displayed along with the prediction.

---

# 📸 Application Preview

> Add screenshots of your application here after deployment.

Example:

```
Home Page Screenshot

Prediction Result Screenshot

Confidence Score Screenshot
```

---

# 📈 Future Enhancements

- 🌐 Live News URL Prediction
- 🌍 Multi-language Support
- 📊 Explainable AI (XAI)
- ☁ Cloud Deployment
- 📱 Mobile Friendly Interface
- 🔍 News Source Verification

---

# 👩‍💻 Author

**Ankita Joshi**

Master of Computer Applications (MCA)

Major Project

---

# ⭐ Acknowledgements

- Hugging Face
- PyTorch
- Streamlit
- Kaggle
- BERT Research Paper
