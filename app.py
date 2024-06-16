import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import string

# Load data
info_tourism = pd.read_csv("https://raw.githubusercontent.com/khikisb/SistemRekomendasiWisata/main/tourism_with_id.csv")

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

# Tab kedua: Rekomendasi berdasarkan deskripsi
def recommend_by_description():
    user_input = st.text_area("Ceritakan kamu mau pergi kemana? dengan siapa?dan ingin melakukan apa?")
    st.write('Contoh : saya ingin pergi dengan keluarga dan ingin melihat lukisan lukisan yang indah')
    st.write('Contoh : saya ingin pergi ke pantai yang masih jarang orang tahu')
    if user_input:
        # Pilihan untuk menampilkan rekomendasi teratas
        ntop = st.selectbox('Berapa rekomendasi yang ingin ditampilkan?', options=[3, 5, 7], index=0)

        # Pra-pemrosesan teks pada input pengguna
        stop_factory = StopWordRemoverFactory()
        stop_words = stop_factory.get_stop_words()
        stop_words.extend(string.punctuation)

        tfidf = TfidfVectorizer(stop_words=stop_words)
        info_tourism['Description'] = info_tourism['Description'].fillna('')
        tfidf_matrix = tfidf.fit_transform(info_tourism['Description'])

        user_tfidf = tfidf.transform([user_input])

        # Hitung cosine similarity antara input pengguna dan deskripsi tempat wisata
        similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)

        # Dapatkan indeks tempat wisata yang direkomendasikan berdasarkan similarity scores
        similarity_scores = similarity_scores.flatten()
        recommended_indices = similarity_scores.argsort()[::-1]

        # Batasi hasil rekomendasi ke jumlah yang dipilih oleh pengguna (ntop)
        top_indices = recommended_indices[:ntop]

        # Buat DataFrame untuk menampilkan tempat wisata yang direkomendasikan dengan skor cosine similarity
        recommended_places = info_tourism.iloc[recommended_indices][['Place_Name', 'Description', 'Category', 'City', 'Price', 'Rating']]
        recommended_places['Similarity_Score'] = similarity_scores[recommended_indices]

        st.write("Tempat wisata yang direkomendasikan berdasarkan deskripsi Kamu:")
        for index, row in recommended_places.iterrows():
            if row['Similarity_Score'] < 0.2:
                st.markdown(f"**{row['Place_Name']}**\n{row['Description']}\n**Category**: {row['Category']}, **City**: {row['City']}, **Price**: {row['Price']}, **Rating**: {row['Rating']}, **Similarity Score**: {row['Similarity_Score']:.2f} (Score below 0.2)")
            else:
                st.markdown(f"**{row['Place_Name']}**\n{row['Description']}\n**Category**: {row['Category']}, **City**: {row['City']}, **Price**: {row['Price']}, **Rating**: {row['Rating']}, **Similarity Score**: {row['Similarity_Score']:.2f}")

        # Filter recommendations to the top ntop regardless of score
        recommended_top_places = recommended_places.head(ntop)
        if recommended_top_places.empty:
            st.write("Tidak ada rekomendasi yang sesuai dengan preferensi Kamu.")
        else:
            st.write("Berikut rekomendasi teratas untuk Kamu:")
            st.write(recommended_top_places)

    else:
        st.write("Hindari menggunakan nama kota, Karena kami akan merekomendasikan tempat yang paling cocok dengan Kamu di Seluruh Indonesia")

# Main App
st.title("Sistem Rekomendasi Tempat Wisata")

# Pilihan tab
tabs = ["Filter Tempat Wisata", "Rekomendasi berdasarkan Deskripsi"]
choice = st.sidebar.radio("Navigasi", tabs)

# Tampilkan tab yang dipilih
if choice == "Filter Tempat Wisata":
    filter_places()
elif choice == "Rekomendasi berdasarkan Deskripsi":
    recommend_by_description()
