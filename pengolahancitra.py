import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# UTILITY & HELPER FUNCTIONS
# ==============================================================================

def create_sample_images():
    """
    Membuat 3 citra sintetis untuk pengujian jika tidak ada file gambar lokal.
    1. Citra Kontras Rendah (Intensitas berkisar 80-140)
    2. Citra Berkadar Salt & Pepper Noise
    3. Citra Malam Hari (Gelap + Gaussian Noise)
    """
    # Base pattern (bentuk geometris sederhana)
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    xx, yy = np.meshgrid(x, y)
    base_pattern = (np.sin(xx * 10) + np.cos(yy * 10) + 2) / 4.0

    # 1. Citra Kontras Rendah (Rentang intensitas 80 - 140)
    img_low_contrast = (80 + base_pattern * 60).astype(np.uint8)

    # 2. Citra Normal (0 - 255)
    img_normal = (base_pattern * 255).astype(np.uint8)

    # 3. Citra Malam Hari / Gelap (Intensitas 10 - 70)
    img_dark = (10 + base_pattern * 60).astype(np.uint8)

    return img_low_contrast, img_normal, img_dark

def add_salt_and_pepper_noise(img, prob=0.05):
    """Menambahkan Salt & Pepper Noise ke dalam citra."""
    noisy_img = img.copy()
    num_salt = np.ceil(prob * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[tuple(coords)] = 255

    num_pepper = np.ceil(prob * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[tuple(coords)] = 0
    return noisy_img

def add_gaussian_noise(img, mean=0, sigma=25):
    """Menambahkan Gaussian Noise ke dalam citra."""
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return noisy

def apply_stretching(img):
    """Menerapkan Histogram Stretching (Min-Max Normalization)."""
    min_val = float(np.min(img))
    max_val = float(np.max(img))
    if max_val == min_val:
        return img.copy()
    stretched = ((img.astype(np.float32) - min_val) / (max_val - min_val) * 255.0)
    return stretched.astype(np.uint8)

# ==============================================================================
# MODUL UTAMA UNTUK MINI PROJECT (SLIDE 26)
# ==============================================================================

def run_mini_project(img_input=None):
    """
    Menjalankan alur evaluasi Mini Project:
    - Analisis Histogram
    - Evaluasi Contrast Enhancement (Equalization, Stretching, CLAHE)
    - Simulasi Noise (Salt & Pepper, Gaussian)
    - Evaluasi Spatial Filtering (Mean, Median, Gaussian Filter)
    """
    if img_input is None:
        img_low, _, _ = create_sample_images()
        img = img_low
    else:
        img = img_input

    # Ensure grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 1. CONTRAST ENHANCEMENT
    eq_result = cv2.equalizeHist(gray)
    stretch_result = apply_stretching(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(gray)

    # Visualisasi Contrast Enhancement & Histogram
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Evaluasi Contrast Enhancement & Histogram', fontsize=16)

    titles = ['Citra Asli', 'Equalization', 'Stretching', 'CLAHE']
    images = [gray, eq_result, stretch_result, clahe_result]

    for i in range(4):
        axes[0, i].imshow(images[i], cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(titles[i])
        axes[0, i].axis('off')

        axes[1, i].hist(images[i].ravel(), 256, [0, 256], color='black', alpha=0.7)
        axes[1, i].set_xlim([0, 256])
        axes[1, i].set_xlabel('Intensitas')
        if i == 0:
            axes[1, i].set_ylabel('Jumlah Piksel')

    plt.tight_layout()
    plt.show()

    # 2. NOISE SIMULATION & FILTERING COMPARISON
    sp_noisy = add_salt_and_pepper_noise(gray, prob=0.05)
    gauss_noisy = add_gaussian_noise(gray, mean=0, sigma=25)

    # Filter untuk Salt & Pepper Noise
    sp_mean = cv2.blur(sp_noisy, (5, 5))
    sp_median = cv2.medianBlur(sp_noisy, 5)
    sp_gauss = cv2.GaussianBlur(sp_noisy, (5, 5), 0)

    # Filter untuk Gaussian Noise
    gn_mean = cv2.blur(gauss_noisy, (5, 5))
    gn_median = cv2.medianBlur(gauss_noisy, 5)
    gn_gauss = cv2.GaussianBlur(gauss_noisy, (5, 5), 0)

    # Plot Filter vs Salt & Pepper Noise
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Perbandingan Filter pada Salt & Pepper Noise', fontsize=14)
    axes[0].imshow(sp_noisy, cmap='gray'); axes[0].set_title('S&P Noise'); axes[0].axis('off')
    axes[1].imshow(sp_mean, cmap='gray'); axes[1].set_title('Mean Filter'); axes[1].axis('off')
    axes[2].imshow(sp_median, cmap='gray'); axes[2].set_title('Median Filter'); axes[2].axis('off')
    axes[3].imshow(sp_gauss, cmap='gray'); axes[3].set_title('Gaussian Filter'); axes[3].axis('off')
    plt.tight_layout()
    plt.show()

    # Plot Filter vs Gaussian Noise
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Perbandingan Filter pada Gaussian Noise', fontsize=14)
    axes[0].imshow(gauss_noisy, cmap='gray'); axes[0].set_title('Gaussian Noise'); axes[0].axis('off')
    axes[1].imshow(gn_mean, cmap='gray'); axes[1].set_title('Mean Filter'); axes[1].axis('off')
    axes[2].imshow(gn_median, cmap='gray'); axes[2].set_title('Median Filter'); axes[2].axis('off')
    axes[3].imshow(gn_gauss, cmap='gray'); axes[3].set_title('Gaussian Filter'); axes[3].axis('off')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# IMPLEMENTASI STUDI KASUS (SLIDE 25)
# ==============================================================================

def solve_case_studies():
    """Menjalankan dan menampilkan hasil eksekusi Kasus 1, 2, dan 3."""
    print("=" * 60)
    print("EVALUASI DAN JAWABAN LATIHAN KASUS (SLIDE 25)")
    print("=" * 60)

    # Citra sampel
    img_low, _, img_dark = create_sample_images()

    # KASUS 1: Kontras Rendah (80-140), Tanpa Noise
    print("\n[Kasus 1] Kontras rendah (80-140), tanpa noise")
    print("Pilihan Terbaik: Histogram Stretching")
    print("Alasan: Karena rentang piksel sempit di daerah tengah (80-140) tanpa noise,")
    print("Stretching secara presisi menarik nilai min-max ke 0-255 tanpa mengubah")
    print("proporsi distribusi atau memicu over-enhancement seperti Equalization.")

    # KASUS 2: Salt & Pepper Noise
    print("\n[Kasus 2] Banyak titik hitam-putih (Salt & Pepper Noise)")
    print("Pilihan Terbaik: Median Filter")
    print("Alasan: Salt & Pepper berupa nilai ekstrem (0 & 255). Median Filter mengambil")
    print("nilai tengah pada jendela tetangga, sehingga mampu menghilangkan titik noise")
    print("secara total tanpa mengaburkan (blurring) tepi objek.")

    # KASUS 3: Foto Malam Hari (Gaussian Noise + Kontras Rendah)
    print("\n[Kasus 3] Foto malam hari (Gaussian Noise + Kontras Rendah)")
    print("Urutan Terbaik: Step 1 Filtering -> Step 2 Contrast Enhancement (CLAHE)")
    print("Alasan Diskusi: Filtering WAJIB dilakukan TERLEBIH DAHULU.")
    print("Jika Contrast Enhancement dilakukan lebih dulu, ampltitudo noise akan ikut")
    print("diperkuat (diamplifikasi) oleh penyesuaian kontras, membuat noise semakin sulit")
    print("dibersihkan pada tahap filtering.")

    # Demonstrasi Pipeline Kasus 3 (Malam Hari)
    sp_noisy_dark = add_gaussian_noise(img_dark, mean=0, sigma=20)

    # Skenario A: Filtering dulu baru CLAHE (BENAR)
    filtered_first = cv2.GaussianBlur(sp_noisy_dark, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    correct_pipeline = clahe.apply(filtered_first)

    # Skenario B: CLAHE dulu baru Filtering (SALAH)
    clahe_first = clahe.apply(sp_noisy_dark)
    wrong_pipeline = cv2.GaussianBlur(clahe_first, (5, 5), 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Diskusi Kasus 3: Urutan Filtering vs Contrast Enhancement', fontsize=14)
    axes[0].imshow(sp_noisy_dark, cmap='gray'); axes[0].set_title('Citra Input (Gelap + Noise)'); axes[0].axis('off')
    axes[1].imshow(correct_pipeline, cmap='gray'); axes[1].set_title('BENAR: Filter -> CLAHE'); axes[1].axis('off')
    axes[2].imshow(wrong_pipeline, cmap='gray'); axes[2].set_title('SALAH: CLAHE -> Filter'); axes[2].axis('off')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# LIVE CAMERA / REAL-TIME PROCESSING
# ==============================================================================

def process_live_frame(frame, mode='clahe'):
    """Memproses satu frame kamera sesuai mode yang dipilih."""
    if frame is None:
        return None

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if mode == 'original':
        processed = gray
    elif mode == 'equalized':
        processed = cv2.equalizeHist(gray)
    elif mode == 'stretching':
        processed = apply_stretching(gray)
    elif mode == 'clahe':
        processed = clahe.apply(gray)
    elif mode == 'median':
        processed = cv2.medianBlur(gray, 5)
    elif mode == 'gaussian':
        processed = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        processed = gray

    original_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([original_bgr, processed_bgr])

    cv2.putText(combined, 'Original', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, mode.upper(), (original_bgr.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return combined


def run_camera(mode='clahe', camera_index=0):
    """Menjalankan preview kamera real-time untuk pengolahan citra."""
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Kamera dengan index {camera_index} tidak dapat dibuka.")
        return False

    print("=== Kamera Live Aktif ===")
    print("Tekan tombol keyboard berikut:")
    print("  1 = Original | 2 = Equalized | 3 = Stretching | 4 = CLAHE")
    print("  5 = Median | 6 = Gaussian | s = Simpan Foto | q = Keluar")

    key_map = {
        ord('1'): 'original',
        ord('2'): 'equalized',
        ord('3'): 'stretching',
        ord('4'): 'clahe',
        ord('5'): 'median',
        ord('6'): 'gaussian',
    }

    current_mode = mode
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        processed = process_live_frame(frame, mode=current_mode)
        if processed is not None:
            cv2.imshow('Pengolahan Citra Real-Time', processed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            folder_name = 'captured_images'
            import os
            os.makedirs(folder_name, exist_ok=True)
            saved_count += 1
            filename = os.path.join(folder_name, f'capture_{saved_count}_{current_mode}.png')
            cv2.imwrite(filename, cv2.cvtColor(process_live_frame(frame, mode=current_mode), cv2.COLOR_BGR2RGB))
            print(f"Foto berhasil disimpan: {filename}")
        elif key in key_map:
            current_mode = key_map[key]
            print(f"Mode aktif: {current_mode}")

    cap.release()
    cv2.destroyAllWindows()
    return True

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    print("=== PILIHAN MODUL PENGOLAHAN CITRA ===")
    print("1. Demo gambar statis")
    print("2. Demo kasus studi")
    print("3. Kamera live")
    print("4. Keluar")

    pilihan = input("Masukkan pilihan [1/2/3/4]: ").strip()

    if pilihan == '1':
        run_mini_project()
    elif pilihan == '2':
        solve_case_studies()
    elif pilihan == '3':
        run_camera(mode='clahe')
    else:
        print("Program selesai.")