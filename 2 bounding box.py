import pandas as pd
import numpy as np
import cv2

# =======================
# LANGKAH 1: Load Dataset
# =======================
color_data = pd.read_csv("colors banyak.csv")  # format: R,G,B,ColorName

# Buat dictionary warna
color_dict = {
    row['ColorName']: (row['R'], row['G'], row['B'])
    for _, row in color_data.iterrows()
}

def nearest_color_name(bgr_pixel):
    """Cari nama warna terdekat dari dataset berdasarkan jarak Euclidean"""
    b, g, r = bgr_pixel
    min_dist = float('inf')
    nearest_name = None
    for name, (R, G, B) in color_dict.items():
        dist = (R-r)**2 + (G-g)**2 + (B-b)**2
        if dist < min_dist:
            min_dist = dist
            nearest_name = name
    return nearest_name

# =======================
# LANGKAH 2: OpenCV
# =======================
cap = cv2.VideoCapture(1)

box_size = 40
tolerance = 20  # makin kecil makin ketat akurasinya
distance = 200  # jarak dari tengah layar (lebih jauh dari sebelumnya)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    cx, cy = width // 2, height // 2

    # ============== BUAT 2 ROI ==============
    # Kotak 1: di kiri tengah
    x1a, y1a = cx - distance - box_size // 2, cy - box_size // 2
    x2a, y2a = x1a + box_size, y1a + box_size
    roi1 = frame[y1a:y2a, x1a:x2a]

    # Kotak 2: di kanan tengah
    x1b, y1b = cx + distance - box_size // 2, cy - box_size // 2
    x2b, y2b = x1b + box_size, y1b + box_size
    roi2 = frame[y1b:y2b, x1b:x2b]

    # =======================
    # Fungsi deteksi warna per ROI
    # =======================
    def detect_color(roi):
        avg_color = roi.mean(axis=(0, 1)).astype(int)
        color_pred = nearest_color_name(avg_color)

        R_pred, G_pred, B_pred = color_dict[color_pred]
        pixels = roi.reshape(-1, 3)
        diffs = np.linalg.norm(pixels - np.array([B_pred, G_pred, R_pred]), axis=1)
        match_pixels = np.sum(diffs < tolerance)
        total_pixels = pixels.shape[0]
        accuracy = (match_pixels / total_pixels * 100) if total_pixels > 0 else 0
        return color_pred, (B_pred, G_pred, R_pred), accuracy

    # Deteksi untuk 2 ROI
    color1, bgr1, acc1 = detect_color(roi1)
    color2, bgr2, acc2 = detect_color(roi2)

    # =======================
    # Tampilkan hasil
    # =======================
    # Kotak sampling
    cv2.rectangle(frame, (x1a, y1a), (x2a, y2a), (0, 0, 0), 2)
    cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (0, 0, 0), 2)

    # Preview warna prediksi (2 kotak di atas layar)
    cv2.rectangle(frame, (50, 50), (100, 100), bgr1, -1)
    cv2.rectangle(frame, (200, 50), (250, 100), bgr2, -1)

    # Text info
    cv2.putText(frame, f'Warna1: {color1} ({acc1:.1f}%)', (50, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, f'Warna2: {color2} ({acc2:.1f}%)', (50, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow('Color Detection - Realtime (2 Kotak)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
