import pandas as pd
import numpy as np
import cv2

# =======================
# LANGKAH 1: Load Dataset
# =======================
color_data = pd.read_csv("warnalumayan.csv")  # format: R,G,B,ColorName

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
cap = cv2.VideoCapture(0)

box_size = 40
tolerance = 20  # makin kecil makin ketat akurasinya

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    cx, cy = width // 2, height // 2

    # Kotak sampling
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2
    roi = frame[y1:y2, x1:x2]

    # Warna rata-rata ROI
    avg_color = roi.mean(axis=(0, 1)).astype(int)
    color_pred = nearest_color_name(avg_color)

    # =======================
    # Hitung akurasi realtime
    # =======================
    R_pred, G_pred, B_pred = color_dict[color_pred]
    pixels = roi.reshape(-1, 3)
    diffs = np.linalg.norm(pixels - np.array([B_pred, G_pred, R_pred]), axis=1)
    match_pixels = np.sum(diffs < tolerance)
    total_pixels = pixels.shape[0]
    accuracy = (match_pixels / total_pixels * 100) if total_pixels > 0 else 0

    # =======================
    # Tampilkan hasil
    # =======================
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

    # Preview warna prediksi
    preview_x, preview_y = 50, 120
    cv2.rectangle(frame, (preview_x, preview_y),
                  (preview_x+50, preview_y+50),
                  (B_pred, G_pred, R_pred), -1)

    # Text info
    cv2.putText(frame, f'Warna: {color_pred}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f'Akurasi: {accuracy:.2f}%', (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('Color Detection - Realtime', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
