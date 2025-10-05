import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Dashboard Analisis Sentimen Sepeda Listrik",
    page_icon="🚲",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-positive {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #155724;
    }
    .prediction-negative {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #721c24;
    }
    .prediction-neutral {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

def remove_username(text):
    regex = r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)"
    result = re.sub(regex, "", text)
    return result

def text_cleaner(text):
    result = text.lower()
    regex = r"([\d]+|[\n\s]|(\\n)|(rt)+|(user)+|[^\w]+|(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/=]*)))+"
    result = re.sub(regex, ' ', result)
    return result

def text_stopword(text, stopwords):
    clean_text = []
    words = text.split()
    for w in words:
        if w not in stopwords:
            clean_text.append(w.lower())
    return ' '.join(clean_text)

def text_slangword(text, slang_in, slang_to):
    clean_text = []
    words = text.split()
    for w in words:
        if w in slang_in:
            index = slang_in.index(w)
            clean_text.append(slang_to[index].lower())
        else:
            clean_text.append(w.lower())
    return ' '.join(clean_text)

def sanitize_text(text, stopwords, slang_in, slang_to):
    result = remove_username(text)
    result = text_cleaner(result)
    result = text_stopword(result, stopwords)
    result = text_slangword(result, slang_in, slang_to)
    return result

def sentiment_analysis_lexicon_indonesia(text, lexicon_positive, lexicon_negative):
    score = 0
    for word in text:
        if word in lexicon_positive:
            score = score + lexicon_positive[word]
        if word in lexicon_negative:
            score = score + lexicon_negative[word]
    
    if score > 0:
        polarity = 'positif'
    elif score < 0:
        polarity = 'negatif'
    else:
        polarity = None   # neutral dihapus (None)
    return score, polarity

@st.cache_data
def load_data_and_train_model():
    """Load data dan train model sekali saja"""
    try:
        data_paths = {
            'dataset': ['filter_data_sentimen.csv', 'data/filter_data_sentimen.csv', 'hasil/filter_data_sentimen.csv'],
            'stopwords': ['stopwordsID.txt', 'data/stopwordsID.txt', 'hasil/stopwordsID.txt'],
            'slangwords': ['slangword.txt', 'data/slangword.txt', 'hasil/slangword.txt'],
            'lexicon_positive': ['lexicon_positive_ver1.csv', 'data/lexicon_positive_ver1.csv', 'hasil/lexicon_positive_ver1.csv'],
            'lexicon_negative': ['lexicon_negative_ver1.csv', 'data/lexicon_negative_ver1.csv', 'hasil/lexicon_negative_ver1.csv']
        }
        
        df = None
        for path in data_paths['dataset']:
            if os.path.exists(path):
                df = pd.read_csv(path, encoding='latin-1')
                break
        
        if df is None:
            return None, "Dataset tidak ditemukan"
        
        stopwords = []
        for path in data_paths['stopwords']:
            if os.path.exists(path):
                with open(path, 'r', encoding='latin-1') as f:
                    stopwords = [line.strip() for line in f if line.strip()]
                break
        
        slang_in, slang_to = [], []
        for path in data_paths['slangwords']:
            if os.path.exists(path):
                with open(path, 'r', encoding='latin-1') as f:
                    for line in f:
                        if '\t' in line:
                            parts = line.strip().split('\t')
                            slang_in.append(parts[0])
                            slang_to.append(parts[-1])
                break
        
        lexicon_positive = {}
        for path in data_paths['lexicon_positive']:
            if os.path.exists(path):
                with open(path, 'r', encoding='latin-1') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            try:
                                lexicon_positive[row[0]] = int(row[1])
                            except ValueError:
                                continue
                break
        
        lexicon_negative = {}
        for path in data_paths['lexicon_negative']:
            if os.path.exists(path):
                with open(path, 'r', encoding='latin-1') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            try:
                                lexicon_negative[row[0]] = int(row[1])
                            except ValueError:
                                continue
                break
        
        text_column = None
        for col in df.columns:
            if col.lower() in ['text', 'tweet', 'full_text', 'content', 'message']:
                text_column = col
                break
        
        if text_column is None:
            text_column = df.select_dtypes(include=['object']).columns[0]
        
        columns_to_drop = ['conversation_id_str', 'created_at', 'favorite_count',
                         'id_str', 'image_url', 'lang', 'location', 'quote_count', 
                         'reply_count', 'retweet_count', 'tweet_url', 'user_id_str']
        
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        df['clean_text'] = df[text_column].apply(
            lambda x: sanitize_text(str(x), stopwords, slang_in, slang_to) if pd.notna(x) else ""
        )
        
        df['tokens'] = df['clean_text'].str.split()
        results = df['tokens'].apply(
            lambda x: sentiment_analysis_lexicon_indonesia(x, lexicon_positive, lexicon_negative) if isinstance(x, list) else (0, None)
        )
        results = list(zip(*results))
        df['polarity_score'] = results[0]
        df['polarity'] = results[1]

        # 🔥 hapus data yang None (neutral)
        df = df.dropna(subset=['polarity'])

        
        filter_data = df[df['polarity_score'] != 0].copy()
        filter_data = filter_data.replace('', np.nan, regex=True)
        filter_data = filter_data.dropna()
        
        if len(filter_data) == 0:
            return df, "Data tidak cukup untuk training"
        
        polarity_mapping = {"negatif": -1, "positif": 1}
        filter_data["polarity_encoded"] = filter_data["polarity"].map(polarity_mapping)
        
        vectorizer = TfidfVectorizer(max_features=3000)
        X = vectorizer.fit_transform(filter_data['clean_text']).toarray()
        y = filter_data["polarity_encoded"]
        
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.3, random_state=42
        )
        
        model = SVC(kernel='linear',probability=True, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        
        return {
            'df': df,
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'stopwords': stopwords,
            'slang_in': slang_in,
            'slang_to': slang_to,
            'lexicon_positive': lexicon_positive,
            'lexicon_negative': lexicon_negative
        }, "Success"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

if 'model_data' not in st.session_state:
    with st.spinner("🔄 Loading data dan training model..."):
        model_data, message = load_data_and_train_model()
        
        if model_data is None:
            st.error(f"❌ {message}")
            st.info("""
            📁 **Pastikan file berada di folder yang benar:**
            - filter_data_sentimen.csv
            - stopwordsID.txt  
            - slangword.txt
            - lexicon_positive_ver1.csv
            - lexicon_negative_ver1.csv
            """)
            st.stop()
        else:
            st.session_state.model_data = model_data
            st.success("✅ Data dan model berhasil dimuat!")

st.markdown('<h1 class="main-header">🚲 Dashboard Analisis Sentimen Sepeda Listrik</h1>', unsafe_allow_html=True)

st.sidebar.title("🧭 Menu")
menu = st.sidebar.radio("Pilih fitur:", ["📊 Dashboard", "🔮 Prediksi Teks"])

if menu == "📊 Dashboard":
    model_data = st.session_state.model_data
    df = model_data['df']
    
    st.header("📊 Dashboard Overview")
    
    sentiment_counts = df['polarity'].value_counts()
    total_data = len(df)
    
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Data</h3>
            <h2>{total_data:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>😊 Positif</h3>
            <h2>{sentiment_counts.get('positif', 0):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>😔 Negatif</h3>
            <h2>{sentiment_counts.get('negatif', 0):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='text-align:'>📊 Distribusi Sentimen</h3>", unsafe_allow_html=True)
        colors = {'positif': '#28a745', 'negatif': '#dc3545', }
        fig_pie = px.pie(
            values=sentiment_counts.values, 
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map=colors
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("<h3 style='text-align: center;'>📈 Perbandingan Kinerja SVM dengan dan tanpa SMOTE</h3>", unsafe_allow_html=True)
        metrik = ["Akurasi", "Precision", "Recall", "F1-Score"]
        svm_tanpa = [0.884, 0.898, 0.884, 0.839]
        svm_dengan = [0.991, 0.992, 0.991, 0.991]
        plt.figure(figsize=(8,5))
        plt.plot(metrik, svm_tanpa, marker='o', label="SVM Tanpa SMOTE", color='red')
        plt.plot(metrik, svm_dengan, marker='o', label="SVM Dengan SMOTE", color='blue')
        for i, val in enumerate(svm_tanpa):
            plt.text(i, val-0.03, f"{val:.3f}", color='red', ha='center')
        for i, val in enumerate(svm_dengan):
            plt.text(i, val+0.03, f"{val:.3f}", color='blue', ha='center')
        plt.xlabel("Metrik")
        plt.ylabel("Nilai")
        plt.ylim(0.7, 1.0)
        plt.legend()
        plt.grid(True)

        st.pyplot(plt)
        
    
    st.subheader("📋 Statistik Detail")
    
    stats_df = pd.DataFrame({
        'Kategori': sentiment_counts.index,
        'Jumlah': sentiment_counts.values,
        'Persentase': [(count/total_data)*100 for count in sentiment_counts.values]
    })
    stats_df = stats_df.round(2)
    st.dataframe(stats_df, use_container_width=True)

elif menu == "🔮 Prediksi Teks":
    st.header("🔮 Prediksi Sentimen Teks")
    
    model_data = st.session_state.model_data
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "✍️ Masukkan teks tentang sepeda listrik:",
            placeholder="Contoh: Sepeda listrik ini sangat bagus, mudah digunakan dan hemat energi!",
            height=150
        )
    
    with col2:
        st.markdown("""
        ### 💡 Tips:
        - Gunakan bahasa Indonesia
        - Tuliskan opini tentang sepeda listrik  
        - Semakin detail, semakin akurat
        """)
    
    if st.button("🎯 Analisis Sentimen", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("🔍 Menganalisis sentimen..."):

                clean_input = sanitize_text(
                    user_input, 
                    model_data['stopwords'],
                    model_data['slang_in'], 
                    model_data['slang_to']
                )
                
                tokens = clean_input.split()
                lexicon_score, lexicon_polarity = sentiment_analysis_lexicon_indonesia(
                    tokens,
                    model_data['lexicon_positive'],
                    model_data['lexicon_negative']
                )
                
                if clean_input.strip():
                    input_vector = model_data['vectorizer'].transform([clean_input])
                    input_vector_dense = input_vector.toarray()

                    prediction = model_data['model'].predict(input_vector_dense)[0]

                    proba = model_data['model'].predict_proba(input_vector_dense)[0]
                    classes = list(model_data['model'].classes_)  # biasanya [-1, 1]
                    neg_idx = classes.index(-1) if -1 in classes else 0
                    pos_idx = classes.index(1) if 1 in classes else 1
                    pos_prob = float(proba[pos_idx]) * 100.0
                    neg_prob = float(proba[neg_idx]) * 100.0

                    if prediction == 1:
                        final_sentiment = "Positif"
                        sentiment_class = "positive"
                        emoji = "😊"
                        confidence_pct = pos_prob
                    else:
                        final_sentiment = "Negatif"
                        sentiment_class = "negative"
                        emoji = "😔"
                        confidence_pct = neg_prob
                
                st.markdown("### 🎯 Hasil Prediksi")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if sentiment_class == "positive":
                        st.markdown(f"""
                        <div class="prediction-positive">
                            <h2>{emoji} {final_sentiment.upper()}</h2>
                            <p>Teks mengandung sentimen positif</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif sentiment_class == "negative":
                        st.markdown(f"""
                        <div class="prediction-negative">
                            <h2>{emoji} {final_sentiment.upper()}</h2>
                            <p>Teks mengandung sentimen negatif</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-neutral">
                            <h2>{emoji} {final_sentiment.upper()}</h2>
                            <p>Teks bersifat netral</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                # Tampilkan Akurasi Prediksi (probabilitas)
                    st.metric(label="Akurasi Prediksi", value=f"{confidence_pct:.1f}%")

                # Notifikasi keyakinan berdasar probabilitas
                if confidence_pct >= 80:
                    st.success("🎯 Prediksi sangat yakin")
                elif confidence_pct >= 60:
                    st.info("🎯 Prediksi cukup yakin")
                else:
                    st.warning("🎯 Prediksi kurang yakin")

                # # (opsional) tampilkan akurasi model dari data uji
                # st.caption(f"Akurasi model (uji): {model_data['accuracy']*100:.1f}%")