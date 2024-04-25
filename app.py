import streamlit as st
import pandas as pd

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

# Tampilkan hasil rekomendasi
st.header('Rekomendasi Tempat Wisata')
if len(filtered_data) == 0:
    st.write('Maaf, tidak ada tempat wisata yang sesuai dengan preferensi Anda.')
else:
    st.write(filtered_data[['Place_Name', 'Description', 'Category', 'City', 'Price', 'Rating']])
