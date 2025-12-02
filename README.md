# Named Entity Recognition using IndoELECTRA

Tugas Besar Natural Language Processing - Named Entity Recognition untuk Bahasa Indonesia menggunakan model IndoELECTRA dan IndoBERT

## Anggota Kelompok

| Nama                       | NIM       |
| -------------------------- | --------- |
| Rayhan Fatih Gunawan       | 122140134 |
| Elsa Elisa Yohana Sianturi | 122140135 |
| NASHWA PUTRI LAISYA        | 122140180 |
| ANISA FITRIYANI            | 122450019 |
| SITI NUR AARIFAH           | 122450006 |
| MUHAMMAD NELWAN FAKHRI     | 122140173 |
| RADITYA ERZA FARANDI       | 122140209 |

## Deskripsi Project

Project ini mengimplementasikan Named Entity Recognition (NER) untuk Bahasa Indonesia menggunakan model **IndoELECTRA** (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) dan IndoBERT IndoELECTRA dan IndoBERT adalah model pre-trained language model yang dioptimalkan untuk bahasa Indonesia, menggunakan arsitektur ELECTRA yang lebih efisien dibandingkan BERT dalam hal computational cost.

### Model: IndoELECTRA

IndoELECTRA menggunakan pendekatan discriminative learning yang melatih model untuk membedakan token asli dari token yang diganti, bukan sekadar memprediksi token yang di-mask seperti pada BERT. Hal ini membuat ELECTRA lebih efisien dan mampu belajar dari semua token input, bukan hanya token yang di-mask.

**Keunggulan IndoELECTRA:**

- Efisiensi komputasi yang lebih baik dibandingkan BERT
- Pre-trained khusus untuk bahasa Indonesia
- Performa yang kompetitif pada berbagai task NLP bahasa Indonesia
- Cocok untuk task sequence labeling seperti NER

### Model: IndoBERT

[penjelasan]

**Keunggulan IndoELECTRA:**
[penjelasan]

### Dataset

Project ini menggunakan dataset **SINGGALANG** yang berisi teks bahasa Indonesia dengan anotasi entitas untuk task Named Entity Recognition.

### Task: Named Entity Recognition (NER)

NER adalah task untuk mengidentifikasi dan mengklasifikasikan entitas bernama dalam teks, seperti:

- **Person (PER)**: Nama orang
- **Organization (ORG)**: Nama organisasi/perusahaan
- **Location (LOC)**: Nama tempat/lokasi
- **Dan entitas lainnya**

## Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama
- **Transformers (Hugging Face)**: Library untuk menggunakan model IndoELECTRA
- **PyTorch/TensorFlow**: Framework deep learning
- **Jupyter Notebook**: Environment untuk development dan eksperimen
