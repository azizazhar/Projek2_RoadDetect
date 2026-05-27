import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 1. KONFIGURASI HALAMAN UTAMA (WIDE)
st.set_page_config(
    page_title="RoadCheck - AI Road Inspection Dashboard",
    page_icon="🛣️",
    layout="wide"
)

# 2. FUNGSI MEMUAT MODEL AI (Kunci Threshold di 0.10)
@st.cache_resource
def muat_model():
    return YOLO('best.pt')

try:
    model = muat_model()
    CONF_THRESHOLD = 0.10 
except Exception as e:
    st.error("File 'best.pt' tidak ditemukan di folder proyek ini. Pastikan file sudah dipindahkan!")
    st.stop()

# ==================== BACKEND MEMORI (SESSION STATE FIX) ====================
if "menu_aktif" not in st.session_state:
    st.session_state.menu_aktif = "Main Program"

# Mengunci file gambar asli agar tidak hilang saat pindah menu
if "foto_saved" not in st.session_state:
    st.session_state.foto_saved = None

# Mengunci hasil kalkulasi AI
if "sudah_deteksi" not in st.session_state:
    st.session_state.sudah_deteksi = False
if "gambar_hasil" not in st.session_state:
    st.session_state.gambar_hasil = None
if "status_level" not in st.session_state:
    st.session_state.status_level = None
if "penjelasan_singkat" not in st.session_state:
    st.session_state.penjelasan_singkat = None
if "kerusakan_unik" not in st.session_state:
    st.session_state.kerusakan_unik = []
if "list_kerusakan" not in st.session_state:
    st.session_state.list_kerusakan = []

# ==================== TAMPILAN SIDEBAR NAVIGASI PANEL UTAMA ====================
with st.sidebar:
    # Logo berada pas di tengah dengan ukuran kotak kecil yang cantik (width=110)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("road_logo.png", width=110)
    
    # Judul teks "Panel Utama" dan caption menjadi rata tengah (Center)
    st.markdown("""
    <div style="text-align: center;">
        <h2 style="margin-bottom: 0; font-family: sans-serif;">Panel Utama</h2>
        <p style="color: gray; font-size: 0.85rem; margin-top: 2px; margin-bottom: 0;">RoadCheck AI v1.0 • 2026</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    # Tombol Menu 1: Main Program
    if st.button("Main Program", use_container_width=True, type="primary" if st.session_state.menu_aktif == "Main Program" else "secondary"):
        st.session_state.menu_aktif = "Main Program"
        st.rerun()
        
    st.write("") 
    
    # Tombol Menu 2: Explore
    if st.button("Explore", use_container_width=True, type="primary" if st.session_state.menu_aktif == "Explore" else "secondary"):
        st.session_state.menu_aktif = "Explore"
        st.rerun()
        
    st.write("") 
    
    # Tombol Menu 3: About
    if st.button("About", use_container_width=True, type="primary" if st.session_state.menu_aktif == "About" else "secondary"):
        st.session_state.menu_aktif = "About"
        st.rerun()
        
    st.write("---")


# ==================== LOGIKA RENDERING HALAMAN UTAMA ====================

# --- Halaman 1: Main Program ---
if st.session_state.menu_aktif == "Main Program":
    # Judul RoadCheck ber-background merah tengah
    st.markdown("""
    <div style="background-color: #FF4B4B; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
        <h1 style="text-align: center; color: white; margin: 0; font-family: sans-serif;">RoadCheck</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center;'>Sistem Pendeteksi Objek Pada Kerusakan Jalan</h3>", unsafe_allow_html=True)
    st.write("---")
    
    st.markdown("### Masukan Data Lapangan")
    file_terunggah = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])
    
    # Jika user mengunggah foto baru, simpan ke memori permanen dan reset status deteksi lama
    if file_terunggah is not None:
        st.session_state.foto_saved = Image.open(file_terunggah)
        st.session_state.sudah_deteksi = False  # Reset hasil deteksi jika ganti foto baru

    # Jika di dalam memori ada foto yang tersimpan, tampilkan ke layar
    if st.session_state.foto_saved is not None:
        st.image(st.session_state.foto_saved, caption="Pratinjau Foto Masuk", use_container_width=True)
        
        # Tombol Deteksi menggunakan background merah (type="primary")
        tombol_deteksi = st.button("Deteksi !", type="primary")
        st.write("---")
        
        if tombol_deteksi:
            with st.spinner("AI sedang menganalisis objek kerusakan..."):
                img_array = np.array(st.session_state.foto_saved)
                results = model.predict(img_array, conf=CONF_THRESHOLD)
                
                # Simpan hasil visualisasi plot YOLO ke memori session_state
                st.session_state.gambar_hasil = results[0].plot()
                
                list_temp = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    list_temp.append(model.names[cls_id])
                
                st.session_state.list_kerusakan = list_temp
                st.session_state.kerusakan_unik = list(set(list_temp))
                
                if len(st.session_state.list_kerusakan) > 0:
                    if "lubang" in st.session_state.kerusakan_unik:
                        st.session_state.status_level = "High"
                        st.session_state.penjelasan_singkat = "Terdeteksi kerusakan fungsional/struktural tingkat tinggi yang membahayakan keselamatan pengguna jalan secara fatal."
                    elif "retak_kulit_buaya" in st.session_state.kerusakan_unik or "pengelupasan_lapisan_permukaan" in st.session_state.kerusakan_unik:
                        st.session_state.status_level = "Medium"
                        st.session_state.penjelasan_singkat = "Terdeteksi kelelahan lapis permukaan aspal tingkat sedang yang jika dibiarkan akan berkembang menjadi lubang struktural."
                    else:
                        st.session_state.status_level = "Low"
                        st.session_state.penjelasan_singkat = "Terdeteksi kerusakan fungsional atau retak garis ringan. Fondasi jalan terpantau masih sangat kokoh."
                else:
                    st.session_state.status_level = "Aman"
                    st.session_state.penjelasan_singkat = "Sistem AI tidak mendeteksi adanya indikasi kerusakan struktural maupun fungsional pada permukaan sampel foto jalan ini."
                
                st.session_state.sudah_deteksi = True

        # BLOK UTAMA ANTI-RESET: Ditampilkan di luar scope file_uploader agar tetap muncul walau uploader kosong
        if st.session_state.sudah_deteksi:
            st.markdown("### Hasil Pendeteksian Objek")
            st.image(st.session_state.gambar_hasil, channels="RGB", caption="Visualisasi Bounding Box YOLOv8", use_container_width=True)
            
            st.write("") 
            st.markdown("### Klasifikasi Kerusakan")
            
            if st.session_state.status_level != "Aman":
                if st.session_state.status_level == "High":
                    warna_box = st.error
                elif st.session_state.status_level == "Medium":
                    warna_box = st.warning
                else:
                    warna_box = st.info
                    
                warna_box(f"**Tingkat Kerusakan: {st.session_state.status_level}**\n\n*Keterangan:* {st.session_state.penjelasan_singkat}")
                
                st.write("**Rincian Objek Terdeteksi:**")
                for item in st.session_state.kerusakan_unik:
                    jumlah = st.session_state.list_kerusakan.count(item)
                    st.write(f"- {item.replace('_', ' ').title()}: **{jumlah} titik**")
                
                st.write("---")
                
                if st.session_state.status_level == "Low":
                    st.info("""
                    **Low ➔ Saran: Crack Sealing (Penutupan Retak)**
                    
                    *Aksi Lapangan:* Bersihkan celah-celah retakan garis menggunakan udara bertekanan kompresor, kemudian injeksikan bahan pengisi berupa aspal cair panas (*sealant*) agar celah tertutup rapat dan kedap dari rembesan air hujan.
                    """)
                elif st.session_state.status_level == "Medium":
                    st.warning("""
                    **Medium ➔ Saran: Patching (Penambalan)**
                    
                    *Aksi Lapangan:* Lakukan metode pengerukan atau penggalian lokal (*patching*) pada area aspal yang mengelupas atau mengalami retak kulit buaya, bersihkan sisa material lama, lalu isi dan padatkan kembali menggunakan material aspal hotmix baru.
                    """)
                elif st.session_state.status_level == "High":
                    st.error("""
                    **High ➔ Saran: Reconstruction (Pengaspalan Ulang / Rekonstruksi)**
                    
                    *Aksi Lapangan:* Adanya kerusakan kategori berat berupa lubang struktural masif mewajibkan dilakukannya tindakan reconstruction struktur perkerasan jalan secara menyeluruh atau pembongkaran fondasi lokal agar daya dukung beban jalan kembali normal.
                    """)
            else:
                st.success("✅ **Klasifikasi: Jalan Normal / Aman**")
                st.write(st.session_state.penjelasan_singkat)

# --- Halaman 2: Explore ---
elif st.session_state.menu_aktif == "Explore":
    st.markdown("""
    <div style="background-color: #FF4B4B; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
        <h2 style="text-align: center; color: white; margin: 0; font-family: sans-serif;">Analisis Penjelasan Kerusakan</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("Berikut adalah jabaran detail mengenai 6 jenis kerusakan aspal yang mampu diidentifikasi oleh sistem:")
    st.write("---")
    
    k1, k2 = st.columns(2)
    with k1:
        with st.container(border=True):
            st.markdown("### 1. Lubang (Pothole)")
            st.write("Depresi lokal pada permukaan jalan yang membentuk mangkok akibat hilangnya material aspal secara struktural. Sangat berbahaya bagi kendaraan kecil.")
        with st.container(border=True):
            st.markdown("### 2. Retak Kulit Buaya (Alligator Cracking)")
            st.write("Retakan yang saling sambung-menyambung membentuk pola kotak kecil mirip sisik buaya. Dipicu oleh kelelahan beban lalu lintas berat secara terus-menerus.")
        with st.container(border=True):
            st.markdown("### 3. Pengelupasan Lapisan (Pavement Peeling)")
            st.write("Lepasnya ikatan antara film aspal dengan batuan agregat, menyebabkan permukaan atas aus dan menjadi kasar.")
    with k2:
        with st.container(border=True):
            st.markdown("### 4. Retak Memanjang (Longitudinal Cracking)")
            st.write("Retakan tunggal yang arahnya sejajar dengan jalur lalu lintas jalan. Umumnya terjadi akibat sambungan perkerasan yang kurang sempurna saat pengaspalan.")
        with st.container(border=True):
            st.markdown("### 5. Retak Blok (Block Cracking)")
            st.write("Retakan berbentuk blok/kotak berukuran besar. Disebabkan oleh perubahan suhu ekstrem (termal) yang membuat lapisan aspal menyusut.")
        with st.container(border=True):
            st.markdown("### 6. Retak Pinggir (Edge Cracking)")
            st.write("Retakan yang terjadi di area sepanjang tepian aspal jalan. Biasanya dipicu oleh lemahnya sokongan tanah lateral atau sistem drainase pinggir yang buruk.")

# --- Halaman 3: About ---
elif st.session_state.menu_aktif == "About":
    st.markdown("""
    <div style="background-color: #FF4B4B; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
        <h2 style="text-align: center; color: white; margin: 0; font-family: sans-serif;">Tentang Model Kecerdasan Buatan</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("Informasi spesifikasi teknis model kecerdasan buatan yang tertanam di dalam sistem aplikasi:")
    st.write("---")
    
    with st.container(border=True):
        st.markdown("### Spesifikasi Arsitektur Model AI")
        st.write("1. **Arsitektur Utama:** YOLOv8 (You Only Look Once) Small Model (`yolov8s.pt`)")
        st.write("2. **Akurasi Model (mAP50):** Berada di kisaran **80.2%** berdasarkan hasil kompilasi data validasi pelatihan akhir")
        st.write("3. **Jumlah Kelas Deteksi:** 6 Kelas Kerusakan Permukaan Jalan Yaitu:")
        st.write("") 
        st.write("a. Lubang (Pothole)")
        st.write("b. Pengelupasan Lapisan Permukaan")
        st.write("c. Retak Blok (Block Cracking)")
        st.write("d. Retak Kulit Buaya (Alligator Cracking)")
        st.write("e. Retak Memanjang (Longitudinal Cracking)")
        st.write("f. Retak Pinggir (Edge Cracking)")
