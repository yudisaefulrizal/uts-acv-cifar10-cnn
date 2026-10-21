# Target Pengerjaan: UTS Advanced Computer Vision (Projek)

> **Mata kuliah:** Advanced Computer Vision · **Kelas:** 3 / 03MKMM002 (S-2 Teknik Informatika)
> **Dosen:** Dr. Kahfi H Suradiradja, M.Kom · **Jenis ujian:** Projek
> **Sumber soal:** `UTS-Advenced Computer Vision V322 - Projek.pdf`

## 0. Ringkasan

Membangun **CNN dari awal** untuk klasifikasi **10 kelas objek** pada dataset publik **CIFAR-10**. Alur dibuat **sederhana dan terfokus: satu langkah, satu metode**.

**Deliverable:**

| No | File | Isi |
|----|------|-----|
| 1 | `uts_acv_cnn.ipynb` | Kode lengkap, bersih, berkomentar |
| 2 | `Laporan_UTS_ACV_<NIM>_<Nama>.pdf` | Laporan 5 bab sesuai struktur soal |
| 3 | `hasil/` (`loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`, dll.) | Gambar dan angka untuk laporan |

---

## 1. Dataset: CIFAR-10

- **Sumber:** Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto.
- **Lokasi:** `data/cifar-10-images/train|test/<kelas>/*.png` (gambar per kelas, lihat `data/README.md`).
- **Citra:** 32×32 RGB.
- **10 kelas:** `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

**Keputusan (dikunci):**

| Bagian | Pilihan | Keterangan |
|--------|---------|------------|
| Data train | 50.000 citra (5.000 per kelas) | Seluruh data train bawaan |
| Data validasi | **20% dari data train** (10.000, 1.000 per kelas) | Sesuai soal (10–20%) |
| Data test | 10.000 citra (1.000 per kelas) | Seluruh data test bawaan |
| Seed | `42` | Hasil reproducible |

Semua kelas seimbang, jadi tidak perlu membahas ketidakseimbangan kelas.

---

## 2. Alur Kerja (satu langkah = satu metode)

| No | Langkah (soal) | Metode yang dipakai |
|----|----------------|---------------------|
| 1 | Muat data | Baca gambar PNG dari folder per kelas (`PIL`) |
| 2 | Pra-pemrosesan | **Normalisasi /255.0** + label **one-hot** |
| 3 | Arsitektur CNN | **3 blok Conv2D + MaxPooling2D → Flatten → Dense → Dense softmax** |
| 4 | Kompilasi | **Adam** (lr 0,001) + **CategoricalCrossentropy** + **accuracy** |
| 5 | Pelatihan | `train_test_split` 80:20 (stratified) lalu `model.fit`, 20 epoch, batch 64 |
| 6 | Evaluasi | **`classification_report`** + **`confusion_matrix`** (scikit-learn) |

### Langkah 1: Muat Data
- [x] Baca gambar dari `data/cifar-10-images/train` dan `test` dengan fungsi `muat_folder`, lalu ubah ke bentuk `(N, 32, 32, 3)`.
- [x] Nama kelas diambil dari nama folder.
- [x] Cetak `shape` data train dan test.
- [x] Tampilkan 1 contoh citra per kelas (grid 2×5) untuk laporan.

### Langkah 2: Pra-Pemrosesan
- [x] Normalisasi: `x = x.astype("float32") / 255.0`.
- [x] Cetak `x.min()` dan `x.max()` sebagai bukti rentang 0–1.
- [x] Ubah label ke **one-hot**: `to_categorical(y, 10)`.

### Langkah 3: Arsitektur CNN (dari awal)

```
Input (32, 32, 3)
Conv2D(32, 3×3, ReLU, padding='same')   → MaxPooling2D(2×2)
Conv2D(64, 3×3, ReLU, padding='same')   → MaxPooling2D(2×2)
Conv2D(128, 3×3, ReLU, padding='same')  → MaxPooling2D(2×2)
Flatten
Dense(128, ReLU)
Dense(10, Softmax)
```

- [x] Bangun dengan `keras.Sequential`.
- [x] Jalankan `model.summary()` dan simpan hasilnya untuk tabel laporan.
- [x] **Tanpa** Dropout, BatchNorm, augmentasi, atau model pretrained (baseline murni; perbaikan cukup jadi saran di Kesimpulan).

### Langkah 4: Kompilasi
- [x] `optimizer=Adam(learning_rate=0.001)`, `loss="categorical_crossentropy"`, `metrics=["accuracy"]`.

### Langkah 5: Pelatihan
- [x] Bagi 20% data train untuk validasi (`train_test_split`, stratified, `seed=42`), lalu `model.fit(..., validation_data=..., epochs=20, batch_size=64)`.
- [x] Simpan history ke `history.csv`.

### Langkah 6: Evaluasi
- [x] `model.evaluate(x_test, y_test)` → test loss & accuracy.
- [x] Prediksi dengan `argmax`, lalu `classification_report` → Accuracy, Precision, Recall, F1-Score.
- [x] `confusion_matrix` → heatmap (seaborn) dengan nama kelas.
- [x] Plot kurva **Loss** dan **Accuracy** (train vs val) per-epoch.
- [x] Catat pasangan kelas yang paling sering tertukar (nilai di luar diagonal terbesar).

---

## 3. Laporan PDF (struktur wajib soal)

### Bab 1: Pendahuluan
- [x] **Latar Belakang:** klasifikasi citra dan peran CNN (singkat).
- [x] **Tujuan:** membangun CNN dari awal untuk mengklasifikasikan 10 kelas objek CIFAR-10.
- [x] **Dataset:** CIFAR-10 (sumber/sitasi), 10 kelas, jumlah train/val/test, ukuran 32×32 RGB, grid contoh per kelas.

### Bab 2: Metodologi
- [x] **Pra-pemrosesan:** normalisasi /255.0 (alasannya: skala input seragam 0–1 sehingga pelatihan lebih stabil) dan one-hot encoding label.
- [x] **Arsitektur:** tabel dari `model.summary()` + penjelasan aktivasi (ReLU di hidden layer, Softmax di output).
- [x] **Parameter pelatihan (tabel):** Adam, CategoricalCrossentropy, lr 0,001, batch 64, 20 epoch, validasi 20%.

### Bab 3: Hasil dan Analisis
- [x] Grafik Loss (train vs val) dan grafik Accuracy (train vs val).
- [x] Tabel Accuracy, Precision, Recall, F1-Score (per kelas + rata-rata).
- [x] Gambar confusion matrix.
- [x] **Analisis overfitting:** pakai angka dan epoch konkret dari kurva (mis. "accuracy train naik terus sampai X%, accuracy val berhenti di Y% sejak epoch Z, val loss naik setelah epoch Z").
- [x] **Analisis kesalahan:** kelas yang paling sering tertukar + alasannya. Kandidat yang umum muncul di CIFAR-10:
  - `cat` vs `dog` (hewan berbulu berkaki empat, bentuk mirip di 32×32),
  - `automobile` vs `truck` (kendaraan darat dengan bentuk dan latar serupa),
  - `deer` vs `horse`, `airplane` vs `ship` (latar langit/laut biru yang mirip).

### Bab 4: Kesimpulan
- [x] Ringkas performa model dan ada/tidaknya overfitting.
- [x] Dua saran: **Dropout** dan **Data Augmentation**.

### Bab 5: Lampiran
- [x] Source code notebook (bersih, berkomentar, sudah *Restart & Run All*). Kode juga ada di Lampiran 5.2 Laporan_UTS_ACV.md.

---

## 4. Checklist Kepatuhan Soal

| Persyaratan | Status |
|-------------|--------|
| Data train & test dimuat | [x] |
| Normalisasi piksel /255.0 | [x] |
| CNN dari awal (Conv2D, MaxPooling2D, Flatten, Dense, softmax) | [x] |
| Optimizer, CategoricalCrossentropy, accuracy | [x] |
| Validasi 20% dari train, history disimpan | [x] |
| Evaluasi di data test: Accuracy, Precision, Recall, F1 | [x] |
| Confusion matrix | [x] |
| Grafik Loss & Accuracy (train vs val) | [x] |
| Analisis overfitting & analisis kesalahan | [x] |
| Kesimpulan + 2 saran | [x] |
| Kode berkomentar sebagai lampiran, laporan PDF | [x] |

---

## 5. Catatan

- Angka di laporan harus sama dengan output notebook (pakai `seed=42`).
- Karena CIFAR-10 banyak dipakai, nilai pembeda ada di **kualitas analisis** Bab 3: tulis dengan angka dan epoch spesifik dari hasil sendiri, bukan kalimat umum.
