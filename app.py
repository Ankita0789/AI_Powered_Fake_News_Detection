import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

st.set_page_config(
    page_title="AI Powered Fake News Detection",
    page_icon="📰",
    layout="wide"
)
st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
background:linear-gradient(135deg,#eef5ff,#ffffff);
}

/* Heading */

.main-title{
text-align:center;
font-size:36px;
font-weight:bold;
color:#0F172A;
margin-top:-20px;
}

.subtitle{
text-align:center;
font-size:16px;
color:#475569;
margin-bottom:25px;
}

/* Text Area */

textarea{

border:3px solid black !important;

border-radius:15px !important;

font-size:17px !important;

background:white !important;

}

/* Button */

div.stButton{

display:flex;

justify-content:center;

}

div.stButton>button{

background:linear-gradient(90deg,#2563EB,#1D4ED8);

color:white;

width:240px;

height:55px;

border-radius:30px;

font-size:20px;

font-weight:bold;

border:none;

box-shadow:0px 8px 18px rgba(0,0,0,0.35);

transition:0.3s;

}

div.stButton>button:hover{

transform:scale(1.05);

background:linear-gradient(90deg,#1E40AF,#1D4ED8);

color:white;

}

</style>
""",unsafe_allow_html=True)
st.markdown(
"""
<div class="main-title">
📰 AI Powered Fake News Detection
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="subtitle">
Powered by Fine-Tuned BERT Transformer
</div>
""",
unsafe_allow_html=True
)

# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    tokenizer = BertTokenizer.from_pretrained(
        "models/fake_news_bert"
    )

    model = BertForSequenceClassification.from_pretrained(
        "models/fake_news_bert"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()
# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_news(news_text):

    encoding = tokenizer(
        news_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probabilities = torch.softmax(outputs.logits, dim=1)

    fake_probability = probabilities[0][0].item() * 100
    real_probability = probabilities[0][1].item() * 100

    confidence, prediction = torch.max(probabilities, dim=1)

    if prediction.item() == 0:
        label = "Fake News"
    else:
        label = "Real News"

    return (
        label,
        confidence.item() * 100,
        fake_probability,
        real_probability
    )
# --------------------------------------------------
# TEXT AREA
# --------------------------------------------------

news_text = st.text_area(
    label="News Article",
    label_visibility="collapsed",
    height=140,
    placeholder="📰 Paste your complete news article here..."
)
# --------------------------------------------------
# PREDICT BUTTON

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    predict = st.button(
        "🔍 Predict News",
        use_container_width=True
    )
    # --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict:

    if news_text.strip() == "":

        st.warning("⚠ Please enter a news article before clicking Predict.")

    else:

        with st.spinner("🤖 AI is analysing the news article..."):

            prediction, confidence, fake_prob, real_prob = predict_news(news_text)

        st.markdown(
            """
            <h3 style="text-align:center;margin-bottom:15px;">
            Prediction Result
            </h3>
            """,
            unsafe_allow_html=True
        )

        left, right = st.columns(2)

        with left:

            if prediction == "Real News":

                st.success("✅ REAL NEWS")

            else:

                st.error("🚨 FAKE NEWS")

        with right:

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )