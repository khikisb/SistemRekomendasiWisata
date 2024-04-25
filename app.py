import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
info_tourism = pd.read_csv("https://raw.githubusercontent.com/khikisb/SistemRekomendasiWisata/main/tourism_with_id.csv")

# Harga minimum dan maksimum yang diperbarui
min_price = 0
max_price = 900000

# Sidebar untuk input pengguna
st.sidebar.title('Filter Tempat Wisata')
categories = st.sidebar.selectbox('Category wisata?', info_tourism['Category'].unique())
cities = st.sidebar.selectbox('Lokasi?', info_tourism['City'].unique())
selected_price_range = st.sidebar.slider('Range Harga?', min_value=min_price, max_value=max_price, value=(min_price, max_price))

min_price, max_price = selected_price_range

# Filter data berdasarkan input pengguna
filtered_data = info_tourism[(info_tourism['Category'] == categories) &
                             (info_tourism['City'] == cities) &
                             (info_tourism['Price'] >= min_price) &
                             (info_tourism['Price'] <= max_price)]

# Tab untuk rekomendasi berdasarkan deskripsi
st.title('Rekomendasi Tempat Wisata')

# Membuat tab kedua
if st.sidebar.button('Rekomendasi berdasarkan deskripsi'):
    st.markdown('## Rekomendasi Tempat Wisata berdasarkan Deskripsi')
    user_input = st.text_area("Ceritakan kamu mau pergi kemana?")

    if user_input:
        # Pra-pemrosesan teks pada input pengguna
        tfidf = TfidfVectorizer(stop_words='english')
        info_tourism['Description'] = info_tourism['Description'].fillna('')
        tfidf_matrix = tfidf.fit_transform(info_tourism['Description'])

        # Pra-pemrosesan teks pada input pengguna
        user_input_tfidf = tfidf.transform([user_input])

        # Hitung cosine similarity antara input pengguna dan deskripsi tempat wisata
        similarity_scores = cosine_similarity(user_input_tfidf, tfidf_matrix)

        # Dapatkan indeks tempat wisata yang direkomendasikan berdasarkan similarity scores
        recommended_indices = similarity_scores.argsort()[0][::-1][:5]

        # Tampilkan tempat wisata yang direkomendasikan
        recommended_places = info_tourism.iloc[recommended_indices]['Place_Name'].tolist()

        st.write("Tempat wisata yang direkomendasikan berdasarkan deskripsi Anda:")
        st.write(recommended_places)
    else:
        st.write("Silakan ceritakan tentang tempat yang ingin Anda kunjungi untuk menerima rekomendasi yang lebih baik.")
