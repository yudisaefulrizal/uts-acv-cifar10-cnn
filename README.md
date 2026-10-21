# Klasifikasi Citra CIFAR-10 dengan CNN

Projek UTS mata kuliah **Advanced Computer Vision** (Program Studi Teknik Informatika S-2, Universitas Pamulang).

Model CNN sederhana dibangun dengan Keras untuk mengklasifikasikan citra CIFAR-10 ke dalam 10 kelas: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*.

## Hasil

| Metrik (data test, 10.000 citra) | Nilai |
|---|---|
| Accuracy | 69,60% |
| Macro precision / recall / F1 | 0,6944 / 0,6960 / 0,6931 |
| Test loss | 1,7789 |

Model mengalami overfitting: accuracy training mencapai 94,81% di epoch 20, sedangkan validasi tertahan di kisaran 67–72% dan val loss terendah ada di epoch 4. Analisis lengkap ada di laporan.

## Isi repo

| File / folder | Keterangan |
|---|---|
| `uts_acv_cnn.ipynb` | Notebook: muat data, pra-pemrosesan, model, pelatihan, evaluasi |
| `Laporan_UTS_ACV.pdf` / `.md` | Laporan projek |
| `hasil/` | Kurva loss dan accuracy, confusion matrix, `history.csv`, `classification_report.txt`, model `.keras` |
| `siapkan_data.py` | Mengunduh CIFAR-10 dan mengubahnya menjadi gambar PNG per kelas |
| `requirements.txt` | Daftar paket Python |
| `TARGET.md` | Daftar target pengerjaan |

## Cara menjalankan

Dibutuhkan Python 3.10–3.13 (TensorFlow belum mendukung Python 3.14).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python siapkan_data.py        # unduh CIFAR-10 (sekitar 170 MB) ke data/
jupyter notebook uts_acv_cnn.ipynb   # lalu Run All
```

Pelatihan 20 epoch memakan waktu sekitar 10 menit di CPU. Seed dikunci (`42`), sehingga hasilnya dapat direproduksi.

## Dataset

CIFAR-10 diunduh dari https://www.cs.toronto.edu/~kriz/cifar.html

Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.
