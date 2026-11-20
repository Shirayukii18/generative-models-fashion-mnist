# Generative Models pada Fashion-MNIST

Eksperimen ini membandingkan Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM. MLP Classifier digunakan untuk menguji informasi visual yang tersimpan pada fitur klasifikasi.

## Menjalankan eksperimen

Gunakan Python 3.12. Konfigurasi PyTorch menggunakan CUDA 12.8. Jalankan perintah berikut dari folder repository.

```bash
python -m pip install -r requirements.txt
python train.py
python evaluate.py
```

Untuk menjalankan kode per cell, buka [notebook](Generative_Models.ipynb). Dataset diunduh saat pertama kali diperlukan. Checkpoint, metrik, dan figure disimpan di folder `outputs/`.

## Riwayat kode

[Riwayat eksperimen](../../commits/experiment-history/) menampilkan tiga revisi kode dari pengerjaan lokal, termasuk implementasi awal dan perbaikan BatchNorm DCGAN. Branch ini hanya memuat file kode publik dari commit asal. Identitas commit disamarkan.
