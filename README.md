# Brain Tumor Classification using Convolutional Neural Network (CNN)

## Anggota Kelompok

1. **Geugeut Nyarikawanti Surahmat** (103132400002)
2. **Wahyuni Salsabila** (103132400010)

Program Studi S1 Sains Data
Fakultas Informatika
Telkom University

---

## Deskripsi Permasalahan

Tumor otak merupakan salah satu penyakit yang memerlukan diagnosis secara cepat dan akurat. Identifikasi jenis tumor melalui citra Magnetic Resonance Imaging (MRI) umumnya dilakukan oleh tenaga medis dan membutuhkan keahlian khusus. Oleh karena itu, diperlukan suatu model klasifikasi berbasis Deep Learning yang mampu membantu proses identifikasi jenis tumor secara otomatis.

Pada proyek ini dikembangkan model **Convolutional Neural Network (CNN)** menggunakan framework PyTorch untuk mengklasifikasikan citra MRI otak ke dalam empat kategori, yaitu **Glioma**, **Meningioma**, **Pituitary**, dan **No Tumor**.

---

## Dataset

Dataset yang digunakan adalah **Brain Tumor Classification (MRI)** yang dipublikasikan melalui Kaggle.

* **Nama Dataset:** Brain Tumor Classification (MRI)
* **Sumber:** Kaggle
* **Owner:** Sartaj Bhuvaji
* **Lisensi:** MIT License

https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

Dataset terdiri atas dua folder utama:

* Training
* Testing

Dengan empat kelas:

* Glioma
* Meningioma
* No Tumor
* Pituitary

Karena ukuran dataset cukup besar, dataset tidak disertakan pada repository ini. Dataset dapat diunduh melalui tautan di atas dan ditempatkan pada folder:

```text
Dataset/
├── Training/
└── Testing/
```

---

## Tahapan Preprocessing

Tahapan preprocessing yang diterapkan pada penelitian ini meliputi:

1. Resize citra menjadi 224 × 224 piksel.
2. Data augmentation menggunakan:

   * Random Horizontal Flip
   * Random Rotation (10°)
3. Konversi citra menjadi Tensor.
4. Normalisasi menggunakan nilai mean dan standard deviation dataset.

---

## Metode yang Digunakan

Metode utama yang digunakan adalah **Convolutional Neural Network (CNN)** dengan framework PyTorch.

Arsitektur model terdiri atas:

* Convolution Layer (32 filter)
* ReLU Activation
* Max Pooling
* Convolution Layer (64 filter)
* ReLU Activation
* Max Pooling
* Convolution Layer (128 filter)
* ReLU Activation
* Max Pooling
* Fully Connected Layer (256 neuron)
* Dropout (0.5)
* Output Layer (4 kelas)

Konfigurasi pelatihan:

| Parameter     |            Nilai |
| ------------- | ---------------: |
| Optimizer     |             Adam |
| Learning Rate |            0.001 |
| Batch Size    |               64 |
| Epoch         |               20 |
| Loss Function | CrossEntropyLoss |

---

## Cara Menjalankan Program

1. Clone repository.

2. Install dependency.

```bash
pip install -r requirements.txt
```

3. Unduh dataset dari Kaggle kemudian letakkan pada folder:

```text
Dataset/
├── Training/
└── Testing/
```

4. Jalankan proses training.

```bash
python train.py
```

5. Evaluasi model.

```bash
python evaluate.py
```

6. Prediksi citra baru.

```bash
python predict.py path/to/image.jpg
```

---

## Hasil Eksperimen dan Evaluasi

Model CNN berhasil melakukan klasifikasi citra MRI otak dengan hasil terbaik sebagai berikut:

* Best Validation Accuracy : **74.37%**
* Test Accuracy : **74.37%**

Evaluasi model dilakukan menggunakan:

* Accuracy
* Confusion Matrix
* Classification Report

---

## Kesimpulan

Model Convolutional Neural Network (CNN) yang dikembangkan mampu mengklasifikasikan citra MRI otak ke dalam empat kategori tumor dengan akurasi pengujian sebesar **74.37%**. Hasil tersebut menunjukkan bahwa CNN mampu mempelajari karakteristik visual dari citra MRI dengan cukup baik sehingga dapat digunakan sebagai dasar dalam pengembangan sistem klasifikasi tumor otak berbasis Deep Learning.

Pengembangan selanjutnya dapat dilakukan dengan menerapkan arsitektur yang lebih kompleks, seperti ResNet atau EfficientNet, serta melakukan optimasi hyperparameter dan teknik augmentasi data untuk meningkatkan performa model.