# Brain Tumor Classification using Convolutional Neural Network (CNN)

## Overview

Brain Tumor Classification using Convolutional Neural Network (CNN) merupakan proyek pembelajaran machine learning yang bertujuan untuk mengklasifikasikan citra Magnetic Resonance Imaging (MRI) otak ke dalam empat kategori tumor, yaitu **Glioma**, **Meningioma**, **Pituitary**, dan **No Tumor**. Model dikembangkan menggunakan framework **PyTorch** dengan pendekatan Deep Learning berbasis CNN yang meliputi tahapan eksplorasi dataset, preprocessing, perancangan arsitektur model, proses training, hingga evaluasi performa model.

---

## Dataset

Dataset yang digunakan merupakan **Brain Tumor Classification (MRI)** yang tersedia secara publik di Kaggle dan disusun oleh **Sartaj Bhuvaji**. Dataset terdiri atas citra MRI otak yang telah dipisahkan ke dalam data **Training** dan **Testing**, masing-masing dengan empat kelas target.

Dataset dapat diakses melalui:

https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

### Kelas Dataset

* Glioma
* Meningioma
* No Tumor
* Pituitary

---

## Arsitektur Model

Model CNN yang dikembangkan terdiri atas tiga blok **Convolutional Neural Network** dengan fungsi aktivasi **ReLU** dan **Max Pooling** untuk proses ekstraksi fitur. Hasil ekstraksi fitur kemudian diteruskan ke lapisan **Fully Connected** yang dilengkapi **Dropout** untuk melakukan klasifikasi ke dalam empat kelas tumor.

---

## Konfigurasi Pelatihan

| Parameter     |            Nilai |
| ------------- | ---------------: |
| Image Size    |        224 × 224 |
| Batch Size    |               64 |
| Optimizer     |             Adam |
| Learning Rate |            0.001 |
| Loss Function | CrossEntropyLoss |
| Epoch         |               20 |

---

## Hasil

Model terbaik yang diperoleh pada penelitian ini menghasilkan:

* **Best Validation Accuracy : 74.37%**
* **Test Accuracy : 74.37%**

---

## Struktur Proyek

```text
brain-tumor-classification-cnn/
│
├── Dataset/
├── Models/
│   ├── best_model.pth
│   └── cnn.py
├── Notebooks/
│   ├── 01_dataset.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model.ipynb
│   ├── 04_training.ipynb
│   └── 05_evaluation.ipynb
├── Utils/
│   ├── data_loader.py
│   ├── train.py
│   └── evaluate.py
├── settings/
│   └── config.py
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

1. Clone repository.
2. Install seluruh dependency menggunakan `requirements.txt`.
3. Jalankan notebook secara berurutan mulai dari:

   * `01_dataset.ipynb`
   * `02_preprocessing.ipynb`
   * `03_model.ipynb`
   * `04_training.ipynb`
   * `05_evaluation.ipynb`

---

## Teknologi

* Python
* PyTorch
* TorchVision
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## Lisensi Dataset

Dataset yang digunakan merupakan dataset publik milik **Sartaj Bhuvaji** yang tersedia di Kaggle dengan lisensi **MIT License**.

---

## Author

Project ini dikembangkan sebagai tugas akhir mata kuliah Pembelajaran Mesin oleh:

1. Geugeut Nyarikawanti Surahmat (103132400002)
2. Wahyuni Salsabila (103132400010)

Program Studi S1 Sains Data
Fakultas Informatika
Telkom University Purwokerto