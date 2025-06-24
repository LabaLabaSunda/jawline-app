import streamlit as st
import cv2
import mediapipe as mp
import math
import numpy as np

st.title("📏 Jawline Scanner")
st.write("Upload foto wajah kamu dan kami akan hitung proporsi rahangnya.")
st.markdown("## 📸 Unggah wajah kamu dan lihat hasil analisis rahangnya!")


uploaded_file = st.file_uploader("Unggah foto wajah (jpg/png)", type=["jpg", "jpeg", "png"])

def hitung_jarak(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def nilai_jawline(jaw_width, chin_height):
    if chin_height == 0:
        return 0
    ratio = jaw_width / chin_height
    score = min(100, max(0, int(100 / ratio)))
    return score

def labelkan(score):
    if score >= 90:
        return "💪 Sigma"
    elif score >= 60:
        return "👍 Lumayan"
    elif score >= 30:
        return "😅 Fat"
    else:
        return "🥲 Obesitas"

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            jaw_left = face_landmarks.landmark[234]
            jaw_right = face_landmarks.landmark[454]
            chin = face_landmarks.landmark[152]

            p1 = (int(jaw_left.x * w), int(jaw_left.y * h))
            p2 = (int(jaw_right.x * w), int(jaw_right.y * h))
            pc = (int(chin.x * w), int(chin.y * h))
            midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

            jaw_width = hitung_jarak(p1, p2)
            chin_height = hitung_jarak(midpoint, pc)
            score = nilai_jawline(jaw_width, chin_height)
            label = labelkan(score)

            # Gambar hasil
            cv2.line(image, p1, p2, (0, 255, 0), 2)
            cv2.line(image, midpoint, pc, (255, 0, 0), 2)
            cv2.putText(image, f"{label} ({score})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Hasil: {label} ({score})", use_container_width=True)
        st.success(f"Jaw Width: {jaw_width:.2f}px | Chin Height: {chin_height:.2f}px | Score: {score} → {label}")
    else:
        st.warning("❌ Wajah tidak terdeteksi, coba upload foto yang lebih jelas.")
