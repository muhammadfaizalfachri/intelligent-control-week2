import pandas as pd
import numpy as np
import cv2
from sklearn.svm import SVC   # Ganti ke SVM

# =======================
# LANGKAH 1: Load Dataset & Training Model
# =======================
color_data = pd.read_csv("colors banyak.csv")  # format: R,G,B,ColorName

# Fitur (RGB) dan label (nama warna)
X = color_data[['R', 'G', 'B']]
y = color_data['ColorName']

# Buat model SVM
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X, y)

def predict_color(bgr_pixel):
    """Prediksi nama warna menggunakan SVM"""
    b, g, r = bgr_pixel
    # Masukkan ke model sebagai DataFrame dengan nama kolom yang sama
    rgb_pixel = pd.DataFrame([[r, g, b]], columns=['R', 'G', 'B'])
    color_pred = model.predict(rgb_pixel)[0]

    # Ambil nilai RGB asli dari dataset untuk preview warna
    row = color_data[color_data['ColorName'] == color_pred].iloc[0]
    return color_pred, (int(row['B']), int(row['G']), int(row['R']))

# =======================
# LANGKAH 2: OpenCV
# =======================
cap = cv2.VideoCapture(1)

box_size = 40
distance = 200  # jarak dari tengah
tolerance = 20  # untuk hitung akurasi

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    cx, cy = width // 2, height // 2

    # ROI kiri
    x1a, y1a = cx - distance - box_size // 2, cy - box_size // 2
    x2a, y2a = x1a + box_size, y1a + box_size
    roi1 = frame[y1a:y2a, x1a:x2a]

    # ROI kanan
    x1b, y1b = cx + distance - box_size // 2, cy - box_size // 2
    x2b, y2b = x1b + box_size, y1b + box_size
    roi2 = frame[y1b:y2b, x1b:x2b]

    # =======================
    # Fungsi deteksi warna per ROI
    # =======================
    def detect_color(roi):
        avg_color = roi.mean(axis=(0, 1)).astype(int)
        color_pred, bgr_pred = predict_color(avg_color)

        # Hitung akurasi sederhana (berapa % pixel mendekati warna prediksi)
        pixels = roi.reshape(-1, 3)
        diffs = np.linalg.norm(pixels - np.array(bgr_pred), axis=1)
        match_pixels = np.sum(diffs < tolerance)
        total_pixels = pixels.shape[0]
        accuracy = (match_pixels / total_pixels * 100) if total_pixels > 0 else 0

        return color_pred, bgr_pred, accuracy

    # Deteksi 2 warna
    color1, bgr1, acc1 = detect_color(roi1)
    color2, bgr2, acc2 = detect_color(roi2)

    # =======================
    # Tampilkan hasil
    # =======================
    # Kotak sampling
    cv2.rectangle(frame, (x1a, y1a), (x2a, y2a), (0, 0, 0), 2)
    cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (0, 0, 0), 2)

    # Preview warna prediksi
    cv2.rectangle(frame, (50, 50), (100, 100), bgr1, -1)
    cv2.rectangle(frame, (200, 50), (250, 100), bgr2, -1)

    # Text info
    cv2.putText(frame, f'Warna1: {color1} ({acc1:.1f}%)', (50, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, f'Warna2: {color2} ({acc2:.1f}%)', (50, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow('Color Detection - SVM (2 Kotak)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
