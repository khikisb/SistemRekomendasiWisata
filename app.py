import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
info_tourism = pd.read_csv("https://raw.githubusercontent.com/khikisb/SistemRekomendasiWisata/main/tourism_with_id.csv")

# Sidebar untuk input pengguna
st.sidebar.header('Pilih Preferensi Anda')

# List pilihan untuk input
categories = info_tourism['Category'].unique()
cities = info_tourism['City'].unique()

# Harga minimum dan maksimum yang diperbarui
min_price = 0
max_price = 900000

# Input dari user
selected_category = st.sidebar.selectbox('Category wisata?', categories)
selected_city = st.sidebar.selectbox('Lokasi?', cities)
selected_price_range = st.sidebar.slider('Range Harga?', min_value=min_price, max_value=max_price, value=(min_price, max_price))

min_price, max_price = selected_price_range

# Filter data berdasarkan input pengguna
filtered_data = info_tourism[(info_tourism['Category'] == selected_category) &
                              (info_tourism['City'] == selected_city) &
                              (info_tourism['Price'] >= min_price) &
                              (info_tourism['Price'] <= max_price)]

# Tampilkan hasil rekomendasi dengan model
st.header('Rekomendasi Tempat Wisata (dengan Model)')

if len(filtered_data) == 0:
    st.write('Maaf, tidak ada tempat wisata yang sesuai dengan preferensi Anda.')
else:
    # Contoh model sederhana menggunakan TF-IDF dan cosine similarity
    tfidf = TfidfVectorizer(stop_words='english')
    info_tourism['Description'] = info_tourism['Description'].fillna('')
    tfidf_matrix = tfidf.fit_transform(info_tourism['Description'])

    # Hitung similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Dapatkan indeks tempat wisata yang sesuai dengan filter
    filtered_indices = filtered_data.index

    # Buat daftar tempat wisata yang direkomendasikan
    recommended_places = []

    for idx in filtered_indices:
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Ambil 5 tempat teratas (tidak termasuk dirinya sendiri)

        place_indices = [i[0] for i in sim_scores]
        recommended_places.extend(info_tourism.iloc[place_indices]['Place_Name'].tolist())

    # Hapus duplikat tempat wisata yang direkomendasikan
    recommended_places = list(set(recommended_places))

    st.write(recommended_places)
