import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Load data
info_tourism = pd.read_csv("https://raw.githubusercontent.com/khikisb/SistemRekomendasiWisata/main/tourism_with_id.csv")

# Preprocessing functions
def clean_punct(text):
    clean_tag = re.compile('@\S+')
    clean_url = re.compile('https?:\/\/.*[\r\n]*')
    clean_hastag = re.compile('#\S+')
    clean_symbol = re.compile('[^a-zA-Z]')
    text = clean_tag.sub('', str(text))
    text = clean_url.sub('', text)
    text = clean_hastag.sub(' ', text)
    text = clean_symbol.sub(' ', text)
    return text

def tokenize_text(text_data):
    return word_tokenize(text_data)

def remove_stopwords(tokenized_data):
    stop_words = set(stopwords.words('indonesian'))
    return [word for word in tokenized_data if word not in stop_words]

def preprocess_text(text):
    text = clean_punct(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    return ' '.join(tokens)

# Tab pertama: Filter Tempat Wisata
def filter_places():
    st.sidebar.title('Filter Tempat Wisata')
    min_price = info_tourism['Price'].min()
    max_price = info_tourism['Price'].max()
    categories = st.sidebar.selectbox('Category wisata?', info_tourism['Category'].unique())
    cities = st.sidebar.selectbox('Lokasi?', info_tourism['City'].unique())
    selected_price_range = st.sidebar.slider('Range Harga?', min_value=min_price, max_value=max_price, value=(min_price, max_price))

    min_price, max_price = selected_price_range

    # Filter data berdasarkan input pengguna
    filtered_data = info_tourism[(info_tourism['Category'] == categories) &
                                 (info_tourism['City'] == cities) &
                                 (info_tourism['Price'] >= min_price) &
                                 (info_tourism['Price'] <= max_price)]

    # Tampilkan hasil filter
    st.header('Tempat Wisata yang Sesuai dengan Preferensi Kamu')
    if len(filtered_data) == 0:
        st.write('Maaf, tidak ada tempat wisata yang sesuai dengan preferensi Kamu.')
    else:
        st.write(filtered_data[['Place_Name', 'Description', 'Category', 'City', 'Price', 'Rating']])

# Tab kedua: Rekomendasi berdasarkan Deskripsi
def recommend_by_description(info_tourism, tfidf_model, tfidf_matrix):
    st.title('Sistem Rekomendasi Tempat Wisata')
    user_input = st.text_area(
        'Ceritakan kamu mau pergi kemana? dengan siapa? dan ingin melakukan apa?', 
        placeholder='Deskripsi:'
    )
    
    if st.button('Dapatkan Rekomendasi'):
        if user_input:
            preprocessed_input = preprocess_text(user_input)
            user_tfidf = tfidf_model.transform([preprocessed_input])

            # Hitung cosine similarity antara input pengguna dan deskripsi tempat wisata
            similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)

            # Dapatkan indeks tempat wisata yang direkomendasikan berdasarkan similarity scores
            similarity_scores = similarity_scores.flatten()
            recommended_indices = similarity_scores.argsort()[::-1]

            # Filter recommendations with similarity score >= 0.18
            filtered_indices = [i for i in recommended_indices if similarity_scores[i] >= 0.18]

            if filtered_indices:
                # Buat DataFrame untuk menampilkan tempat wisata yang direkomendasikan dengan skor cosine similarity
                recommended_places = info_tourism.iloc[filtered_indices][['Place_Name', 'Description', 'Category', 'City', 'Price', 'Rating']]
                recommended_places['Similarity_Score'] = similarity_scores[filtered_indices]

                st.write("Tempat wisata yang direkomendasikan berdasarkan deskripsi Kamu:")
                st.dataframe(recommended_places)

                st.write("Top 3 rekomendasi tempat wisata:")
                st.dataframe(recommended_places.head(3))

                st.write("Top 5 rekomendasi tempat wisata:")
                st.dataframe(recommended_places.head(5))

                st.write("Top 7 rekomendasi tempat wisata:")
                st.dataframe(recommended_places.head(7))
            else:
                st.write("Tidak ada rekomendasi yang sesuai dengan preferensi Kamu.")
        else:
            st.write("Hindari menggunakan nama kota, Karena kami akan merekomendasikan tempat yang paling cocok dengan Kamu di Seluruh Indonesia")

# Load necessary data
hasilproses = pd.read_csv("hasilproses.csv")
tfidf_model = TfidfVectorizer().fit(hasilproses['Description'])  # Fit TF-IDF on tourism descriptions
tfidf_matrix = tfidf_model.transform(hasilproses['Description'])  # Transform tourism descriptions

# Main App
st.title("Sistem Rekomendasi Tempat Wisata")

# Pilihan tab
tabs = ["Filter Tempat Wisata", "Rekomendasi berdasarkan Deskripsi"]
choice = st.sidebar.radio("Navigasi", tabs)

# Tampilkan tab yang dipilih
if choice == "Filter Tempat Wisata":
    filter_places()
elif choice == "Rekomendasi berdasarkan Deskripsi":
    recommend_by_description(info_tourism, tfidf_model, tfidf_matrix)
