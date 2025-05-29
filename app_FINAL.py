import streamlit as st
import cv2
import os
import base64
import numpy as np

from detection_FINAL import detect_number_plates, recognize_number_plates, extract_tax_info, validate_tax
from ultralytics import YOLO
from easyocr import Reader

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Auto NPR",  # Judul halaman
    page_icon=":car:",      # Ikon halaman
    layout="wide"           # Layout lebar
)
st.title('Deteksi Plat Nomor & Validasi Pajak :car:')
st.markdown("---")  # Garis pemisah

def image_to_base64(image_array):
    """
    Konversi numpy array gambar ke string base64.
    Args:
        image_array: Gambar dalam bentuk numpy array
    Returns:
        String base64 yang merepresentasikan gambar
    """
    # Encode gambar ke format JPEG dalam memori
    is_success, buffer = cv2.imencode(".jpg", image_array)
    if is_success:
        # Konversi buffer ke base64 string
        return base64.b64encode(buffer).decode("utf-8")
    return ""  # Return string kosong jika gagal

# CSS untuk mengatur tampilan gambar
st.markdown(
    """
    <style>
    /* Kelas untuk gambar dengan tinggi maksimum 400px */
    .fixed-image {
        max-height: 400px;       /* Batas tinggi */
        width: auto;             /* Lebar menyesuaikan proporsi */
        object-fit: contain;     /* Gambar tidak terdistorsi */
        margin: 0 auto;          /* Posisi tengah */
        display: block;          /* Tampilan blok */
    }
    /* Container untuk hasil deteksi */
    .result-container {
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True  # Izinkan HTML tidak aman
)

# Uploader file untuk gambar plat nomor
uploaded_file = st.file_uploader("Unggah Gambar Plat Nomor", type=["jpg", "jpeg", "png"])
upload_path = "uploads"  # Folder untuk menyimpan file yang diunggah

# Jika ada file yang diunggah
if uploaded_file is not None:
    # Simpan gambar yang diunggah ke folder uploads
    image_path = os.path.join(upload_path, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())  # Tulis buffer file ke disk
    
    # Tampilkan spinner selama pemrosesan
    with st.spinner("Memproses gambar..."):
        # Load model YOLO dari direktori lokal
        model = YOLO("dataset/runs/detect/car_plate_detection4/weights/best.pt")
        # Inisialisasi EasyOCR reader untuk bahasa Inggris dengan GPU
        reader = Reader(['en'], gpu=True)
        
        # Baca gambar menggunakan OpenCV
        image = cv2.imread(image_path)
        # Konversi dari BGR ke RGB (karena OpenCV menggunakan BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Konversi gambar ke base64 untuk ditampilkan dengan CSS
        # Catatan: OpenCV menggunakan format BGR, jadi konversi RGB->BGR untuk encoding
        image_base64 = image_to_base64(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        
        # Layout dua kolom: kiri untuk gambar asli, kanan untuk hasil deteksi
        col1, col2 = st.columns([2, 3])
        
        # Kolom kiri: tampilkan gambar yang diunggah
        with col1:
            st.subheader("Gambar yang Diunggah")
            if image_base64:
                # Tampilkan gambar dengan tag HTML dan kelas CSS fixed-image
                st.markdown(
                    f'<img src="data:image/jpeg;base64,{image_base64}" class="fixed-image">',
                    unsafe_allow_html=True
                )
            else:
                # Jika konversi base64 gagal, tampilkan dengan st.image
                st.image(image_rgb, use_column_width=True)
        
        # Kolom kanan: hasil deteksi dan validasi pajak
        with col2:
            st.subheader("Hasil Deteksi")
            
            # Deteksi plat nomor pada gambar
            number_plate_list = detect_number_plates(image, model)
            
            # Jika ada plat terdeteksi
            if number_plate_list:
                # Kenali teks pada plat nomor yang terdeteksi
                number_plate_list = recognize_number_plates(image_path, reader, number_plate_list)
                
                # Tampilkan setiap hasil deteksi dalam container
                with st.container():
                    for i, (box, text) in enumerate(number_plate_list):
                        st.divider()  # Garis pemisah
                        st.subheader(f"Deteksi #{i+1}")  # Judul deteksi
                        
                        # Tampilkan teks plat yang dikenali dalam format code
                        st.code(f"Teks Plat: {text}", language="text")
                        
                        # Ekstrak info pajak dari teks plat
                        tax_month, tax_year = extract_tax_info(text)
                        
                        # Jika berhasil ekstrak info pajak
                        if tax_month and tax_year:
                            # Validasi status pajak
                            status, masa_berlaku = validate_tax(tax_month, tax_year)
                            
                            # Tampilkan bulan dan tahun pajak dalam dua kolom
                            col_a, col_b = st.columns(2)
                            col_a.metric("Bulan Pajak", tax_month)
                            col_b.metric("Tahun Pajak", tax_year)
                            
                            # Tampilkan masa berlaku
                            st.metric("Masa Berlaku", masa_berlaku)
                            
                            # Tampilkan status dengan warna sesuai
                            if status == "AKTIF":
                                st.success(f"✅ Status Pajak: {status}")
                            elif status == "KADALUARSA":
                                st.error(f"❌ Status Pajak: {status}")
                            else:
                                st.warning(f"⚠️ {status}")
                        else:
                            # Jika gagal ekstrak info pajak
                            st.warning("Informasi pajak tidak terdeteksi pada plat nomor")
                        
                        # Tampilkan gambar area plat yang terdeteksi
                        cropped_plate = image_rgb[box[1]:box[3], box[0]:box[2]]
                        st.image(cropped_plate, caption="Area Plat Terdeteksi", width=300)
            else:
                # Jika tidak ada plat terdeteksi
                st.error("Tidak terdeteksi plat nomor pada gambar")
else:
    # Pesan jika belum ada gambar yang diunggah
    st.info("Silakan unggah gambar plat nomor kendaraan untuk memulai")