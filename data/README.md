# Dataset CIFAR-10

Folder ini tidak ikut disimpan di repo (ukuran sekitar 400 MB). Buat dengan menjalankan `python siapkan_data.py` dari folder utama.

**Sumber:** https://www.cs.toronto.edu/~kriz/cifar.html
**Sitasi:** Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.

## Struktur hasil

```
data/
  raw/cifar-10-python.tar.gz   arsip asli (MD5 c58f30108f718f92721af3b95e74349a, diverifikasi oleh skrip)
  cifar-10-images/             gambar PNG per kelas, dibaca oleh notebook
    train/<kelas>/*.png            5.000 gambar per kelas (50.000)
    test/<kelas>/*.png             1.000 gambar per kelas (10.000)
```

Gambar berukuran 32×32 piksel RGB. Nama folder adalah label kelas.

**Kelas:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
