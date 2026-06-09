import streamlit as st
import pandas as pd
import joblib
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ======================================
# DOWNLOAD NLTK
# ======================================

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# ======================================
# LOAD MODEL
# ======================================

model = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ======================================
# LOAD NORMALIZATION DICTIONARY
# ======================================

norm_df = pd.read_csv(
    "colloquial-indonesian-lexicon.csv"
)

normalization_dict = dict(
    zip(
        norm_df['slang'],
        norm_df['formal']
    )
)

# ======================================
# STEMMER
# ======================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ======================================
# STOPWORD
# ======================================

stop_words = set(
    stopwords.words('indonesian')
)

negation_words = {
    'tidak',
    'bukan',
    'ga',
    'gak',
    'tak',
    'kurang',
    'belum'
}

stop_words = stop_words - negation_words

# ======================================
# PREPROCESSING
# ======================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r'http\S+|www\S+',
        '',
        text
    )

    text = re.sub(
        r'@\w+',
        '',
        text
    )

    text = re.sub(
        r'#\w+',
        '',
        text
    )

    text = re.sub(
        r'\d+',
        '',
        text
    )

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


def normalize_text(tokens):

    result = []

    for word in tokens:

        if word in normalization_dict:
            result.append(
                normalization_dict[word]
            )
        else:
            result.append(word)

    return result


def preprocess_text(text):

    text = clean_text(text)

    tokens = word_tokenize(text)

    tokens = normalize_text(tokens)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    return " ".join(tokens)

# ======================================
# MAPPING LABEL
# ======================================

label_mapping = {

    "C1": {
        "aspek": "Daya Tarik",
        "sentimen": "Positif"
    },

    "C6": {
        "aspek": "Daya Tarik",
        "sentimen": "Negatif"
    },

    "C2": {
        "aspek": "Akses + Daya Tarik",
        "sentimen": "Negatif"
    }
}

# ======================================
# UI
# ======================================

st.set_page_config(
    page_title="ABSA Borobudur",
    page_icon="🏛️"
)

st.title(
    "🏛️ ABSA Borobudur"
)

st.write(
    "Analisis Aspek dan Sentimen Ulasan Wisata Borobudur menggunakan TF-IDF dan SVM Linear"
)

review = st.text_area(
    "Masukkan Ulasan Wisata"
)

# ======================================
# PREDIKSI
# ======================================

if st.button("Prediksi"):

    if review.strip() == "":

        st.warning(
            "Masukkan ulasan terlebih dahulu."
        )

    else:

        clean_review = preprocess_text(review)

        vector = tfidf.transform(
            [clean_review]
        )

        pred = model.predict(vector)[0]

        hasil = label_mapping[pred]

        st.success(
            f"Kelas: {pred}"
        )

        st.write(
            f"**Aspek:** {hasil['aspek']}"
        )

        st.write(
            f"**Sentimen:** {hasil['sentimen']}"
        )

        with st.expander(
            "Lihat hasil preprocessing"
        ):
            st.write(clean_review)