import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import pandas as pd

# --------------------------
# Definisi Model CNN Sederhana
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --------------------------
# Fungsi-Fungsi Utilitas dengan Caching
# --------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_path: str = "cnn-tomato_disease-20250205.pth"):
    """
    Memuat model CNN yang telah dilatih.
    """
    model = SimpleCNN(num_classes=10)
    try:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except Exception as e:
        st.error(f"Error memuat model: {e}")
    return model

@st.cache_data(show_spinner=False)
def load_disease_info(json_path: str = "tomato_diseases_description.json"):
    """
    Memuat informasi penyakit dari file JSON.
    """
    try:
        with open(json_path, "r") as f:
            info = json.load(f)
    except Exception as e:
        st.error(f"Error memuat informasi penyakit: {e}")
        info = {}
    return info

@st.cache_data(show_spinner=False)
def get_test_transform(image_size=(224, 224)):
    """
    Mendefinisikan pipeline transformasi untuk preprocessing gambar.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def classify_image(image: Image.Image, model, transform, class_names, threshold=0.6):
    """
    Melakukan klasifikasi terhadap gambar yang diberikan.
    Jika confidence < threshold, kita kembalikan label "unknown" (bukan daun tomat).
    """
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        confidence_val = confidence.item()
        predicted_label = class_names[predicted_idx.item()]

    # Jika nilai confidence di bawah threshold, tandai sebagai "unknown"
    if confidence_val < threshold:
        return "unknown", confidence_val
    else:
        return predicted_label, confidence_val

# --------------------------
# Inisialisasi Variabel Global
# --------------------------
class_names = [
    'bacterial_spot', 'early_blight', 'late_blight', 'leaf_mold',
    'septoria_leaf_spot', 'spider_mites', 'target_spot',
    'yellow_leaf_curl_virus', 'mosaic_virus', 'healthy'
]
dir_images = "informasi_penyakit_image"

# Muat model, informasi penyakit, dan transformasi gambar (menggunakan caching)
model = load_model()
disease_info = load_disease_info()
test_transform = get_test_transform()

# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.title("Dashboard")
page = st.sidebar.radio("Pilih Menu", ["Welcome", "Klasifikasi", "Informasi Penyakit", "About"])

# --------------------------
# Halaman Welcome
# --------------------------
if page == "Welcome":
    st.title("Scanomato 🔍🍅")
    st.header("👋 Selamat Datang di Aplikasi Pendeteksi Penyakit Tanaman Tomat!")
    st.write("Aplikasi ini bertujuan untuk melakukan klasifikasi jenis penyakit pada tanaman tomat berdasarkan gambar daunnya.")
    st.write("Contoh tampilan:")
    # Pastikan file gambar tersedia di folder 'display_images'
    if os.path.exists("display_images/display_perbandingan.png"):
        st.image("display_images/display_perbandingan.png", width=300)
    else:
        st.image("https://via.placeholder.com/300", caption="Contoh Gambar")
    
    st.header("Model yang Digunakan")
    st.write("Aplikasi ini menggunakan model CNN sederhana dengan arsitektur sebagai berikut:")
    if os.path.exists("display_images/cnn_model.png"):
        st.image("display_images/cnn_model.png")
    else:
        st.image("https://via.placeholder.com/300", caption="Diagram Model")
    st.write("Silakan pilih menu di samping untuk melakukan klasifikasi, melihat informasi penyakit, dan informasi tentang aplikasi ini.")

# --------------------------
# Halaman Klasifikasi
# --------------------------
elif page == "Klasifikasi":
    st.title("🔍 Klasifikasi Penyakit!")

    classification_type = st.selectbox("Pilih Jenis Pendeteksi", ["Pilih Jenis Klasifikasi", "Satuan", "Banyak"])

    if classification_type == "Pilih Jenis Klasifikasi":
        st.write("Silakan pilih jenis klasifikasi terlebih dahulu.")
    
    elif classification_type == "Satuan":
        image = None  # Inisialisasi variabel gambar
        sumber = st.radio("Pilih Sumber Gambar", ("Upload Gambar", "Ambil Foto"))
        
        if sumber == "Upload Gambar":
            uploaded_file = st.file_uploader("Upload Gambar", type=["JPG", "png", "jpeg"])
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Gambar yang Diunggah", width=300)
                except Exception as e:
                    st.error(f"Error membuka gambar: {e}")
        else:
            foto = st.camera_input("Ambil Foto")
            if foto is not None:
                try:
                    image = Image.open(foto).convert("RGB")
                    st.image(image, caption="Gambar yang Diambil", width=300)
                except Exception as e:
                    st.error(f"Error membuka gambar: {e}")
        
        # Proses klasifikasi jika gambar tersedia
        if image is not None:
            predicted_label, confidence = classify_image(
                image, model, test_transform, class_names
            )
            
            if predicted_label == "unknown":
                st.warning(
                    f"Gambar ini mungkin bukan daun tomat. (Confidence: {confidence:.2f})"
                )
            else:
                st.write(f"**Hasil Diagnosa:** {predicted_label.replace('_', ' ').title()}")
                st.write(f"**Probabilitas Prediksi:** {confidence * 100:.2f}%")
                
                # Dapatkan informasi penyakit terkait
                info = disease_info.get(predicted_label, {
                    "nama_lain": "Tidak ada informasi",
                    "deskripsi": "Informasi tidak tersedia",
                    "penanganan": "Tidak ada rekomendasi"
                })
                st.subheader("Informasi Penyakit:")
                st.write(f"📌 **Nama Lain:** {info.get('nama_lain', 'Tidak ada informasi')}")
                st.write(f"📌 **Deskripsi:** {info.get('deskripsi', 'Informasi tidak tersedia')}")
                st.write(f"💊 **Pengobatan:** {info.get('penanganan', 'Tidak ada rekomendasi')}")

                if confidence < 0.6:
                    st.error("Kepercayaan model kurang dari 60%. Silakan unggah ulang gambar!")
    
    elif classification_type == "Banyak":
        uploaded_files = st.file_uploader(
            "Upload Gambar (Batch)",
            type=["JPG", "png", "jpeg"],
            accept_multiple_files=True
        )
        if uploaded_files:
            results = []
            for uploaded_file in uploaded_files:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                except Exception as e:
                    st.error(f"Error membuka {uploaded_file.name}: {e}")
                    continue

                predicted_label, confidence = classify_image(
                    image, model, test_transform, class_names
                )
                
                # Jika unknown, tambahkan catatan
                if predicted_label == "unknown":
                    results.append([
                        uploaded_file.name, 
                        "Unknown (Not a Tomato Leaf)", 
                        confidence
                    ])
                else:
                    results.append([
                        uploaded_file.name, 
                        predicted_label, 
                        confidence
                    ])
            
            if results:
                df = pd.DataFrame(results, columns=["Nama File", "Label", "Confidence"])
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "classification_results.csv", "text/csv", key="download-csv")

# --------------------------
# Halaman Informasi Penyakit
# --------------------------
elif page == "Informasi Penyakit":
    st.title("🩺 Informasi Penyakit")
    st.write("Berikut adalah penjelasan dari tiap hasil klasifikasi yang mungkin dideteksi oleh aplikasi:")

    for key, value in disease_info.items():
        # Tentukan path gambar untuk setiap penyakit
        image_path = os.path.join(dir_images, f"{key}.JPG")
        if not os.path.exists(image_path):
            image_path = "https://via.placeholder.com/150"

        st.header(f"{key.replace('_', ' ').title()}")
        st.image(image_path, caption=key.replace('_', ' ').title(), width=250)
        st.markdown(f"""
        **{key.replace('_', ' ').title()}**  
        - **Nama Lain:** {value.get("nama_lain", "Tidak tersedia")}  
        - **Deskripsi:** {value.get("deskripsi", "Tidak tersedia")}  
        - **Penanganan:** {value.get("penanganan", "Tidak tersedia")}  
        """)

# --------------------------
# Halaman About
# --------------------------
elif page == "About":
    st.title("📌 About the App")
    st.markdown("""
    Aplikasi ini dibuat untuk memenuhi tugas besar **Lab AI: Study Group** di **Telkom University**.
    """)
    
    st.write("Scanomato merupakan hasil dari tugas besar kelompok 2, yang diinisiasi dengan tujuan mendukung Sustainable Development Goals:")
    st.markdown("""
    - **SDG 2**: Zero Hunger  
    - **SDG 8**: Decent Work and Economic Growth  
    - **SDG 12**: Responsible Consumption and Production  
    """)
 
    st.header("👥 Kelompok 2 - Developers")
    developers = [
        "M. Rifqi Dzaky Azhad",
        "Hauzan Rafi Attallah",
        "Fransiskus Harris Berliandu",
        "Kemas M Aryadary Rasyad",
        "Yoanes Bosko Galih Suripramesthi"
    ]
    for name in developers:
        st.markdown(f"- {name}")

    st.markdown("---")
    st.write("🔗 [GitHub](https://github.com/Liamours/Tugas-Besar-Lab-AI-SG)")
