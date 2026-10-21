"""Mengunduh CIFAR-10 dan menyimpannya sebagai gambar PNG per kelas.

Hasil (dibaca oleh uts_acv_cnn.ipynb):
    data/cifar-10-images/train/<kelas>/<kelas>_NNNN.png   (5.000 per kelas)
    data/cifar-10-images/test/<kelas>/<kelas>_NNNN.png    (1.000 per kelas)

Penggunaan:  python siapkan_data.py [folder_data]     (bawaan: data)
"""
import hashlib
import pickle
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"  # checksum resmi dari situs CIFAR-10


def unduh(tujuan: Path):
    """Mengunduh arsip bila belum ada, lalu memverifikasi checksum MD5-nya."""
    if not tujuan.exists():
        tujuan.parent.mkdir(parents=True, exist_ok=True)
        print(f"Mengunduh {URL} (sekitar 170 MB, bisa lama) ...")
        urllib.request.urlretrieve(URL, tujuan)
    md5 = hashlib.md5(tujuan.read_bytes()).hexdigest()
    if md5 != MD5:
        sys.exit(f"Checksum tidak cocok ({md5}); hapus {tujuan} lalu jalankan ulang.")
    print("Arsip terverifikasi (MD5 cocok).")


def baca_batch(path: Path):
    """Membaca satu file batch CIFAR-10: citra (N, 32, 32, 3) dan label (N,)."""
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    x = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return x, batch[b"labels"]


def main():
    data = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    arsip = data / "raw" / "cifar-10-python.tar.gz"
    keluaran = data / "cifar-10-images"
    unduh(arsip)

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(arsip) as tar:
            tar.extractall(tmp, filter="data")
        src = Path(tmp) / "cifar-10-batches-py"
        kelas = pickle.load(open(src / "batches.meta", "rb"))["label_names"]
        for split, files in [("train", [f"data_batch_{i}" for i in range(1, 6)]),
                             ("test", ["test_batch"])]:
            for k in kelas:
                (keluaran / split / k).mkdir(parents=True, exist_ok=True)
            hitung = dict.fromkeys(kelas, 0)
            for nama in files:
                x, y = baca_batch(src / nama)
                for citra, label in zip(x, y):
                    k = kelas[label]
                    Image.fromarray(citra).save(keluaran / split / k / f"{k}_{hitung[k]:04d}.png")
                    hitung[k] += 1
            print(f"{split}: {sum(hitung.values())} gambar")
    print("Selesai:", keluaran)


if __name__ == "__main__":
    main()
