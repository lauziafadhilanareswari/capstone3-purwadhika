# 🛒 E-Commerce Customer Churn Risk Identification & Mitigation Strategy

> *"Using Machine Learning Models to Predict and Reduce Churn Rates in Online Retail Companies"*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-189025)](https://xgboost.readthedocs.io)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-8E44AD)](https://imbalanced-learn.org)

</div>

---

| | |
|---|---|
| **Author** | Lauzia Fadhila Nareswari |
| **Domain** | E-Commerce |
| **Task Type** | Binary Classification |
| **Final Model** | Random Forest (Tuned) — threshold 0.4 |

---

## Project Repository Structure

```
capstone3-purwadhika/
│
├── README.md                                   # Dokumentasi proyek (file ini)
├── Ecommerce_Churn-Final.ipynb                 # End-to-end Jupyter
├── data_ecommerce_customer_churn.csv          # Dataset mentah
└── Model                                       # Dataset bersih hasil data cleaning
      └── preprocessor.pkl                      # Saved final model via Pickle
      └── best_rf_model.pkl                     # Saved final model via Pickle
```

---

## Project Introduction

E-commerce adalah model bisnis yang memungkinkan perusahaan dan konsumen melakukan transaksi jual beli produk atau layanan secara elektronik melalui internet. Salah satu tantangan terbesar yang dihadapi oleh perusahaan e-commerce adalah **retensi pelanggan**.

Project ini bertujuan untuk membangun model machine learning berbasis **klasifikasi biner** guna memprediksi risiko churn pelanggan secara proaktif, sehingga perusahaan dapat melakukan intervensi yang tepat sasaran sebelum pelanggan benar-benar berhenti bertransaksi.

---

## Business Problem & Data Understanding

### 1. Business Problem

Perusahaan e-commerce menghadapi masalah serius dalam mempertahankan pelanggan:

- 📉 **Churn rate mencapai 16.3%** dari total pelanggan
- 💸 Biaya akuisisi pelanggan baru **5–25× lebih mahal** dibanding mempertahankan pelanggan lama
- ❓ **Business Question:** *Pelanggan mana yang paling mungkin churn dalam waktu dekat?*

**Mengapa masalah ini penting?**
Setiap pelanggan yang churn berarti hilangnya seluruh Customer Lifetime Value (CLV) dan terbuangnya biaya akuisisi awal. Dengan mendeteksi churner lebih awal, perusahaan dapat mengalokasikan anggaran retensi secara efisien dan terukur.

### 2. Goals & Target

| | Deskripsi |
|---|---|
| **Goal** | Membangun **model klasifikasi** untuk memprediksi risiko churn pelanggan |
| **Technical Target** | Mencapai **Recall ≥ 0.80** (mendeteksi minimal 80% calon churner) |
| **Business Target** | Menurunkan churn rate **20–30%** melalui kampanye retensi yang lebih terarah |

### 3. Analytical Approach

**Task Type:** Binary Classification
```
Target Variable — Churn:  0 = No Churn  |  1 = Churn
```

**Mengapa fokus pada Recall (Sensitivity)?**

| Tipe Kesalahan | Dampak Bisnis |
|---|---|
| **False Positive** *(pelanggan loyal diprediksi churn)* | 🟡 **Rendah** — sedikit pemborosan biaya promosi |
| **False Negative** *(churner tidak terdeteksi)* | 🔴 **Kritis** — kehilangan seluruh CLV + biaya akuisisi awal terbuang |

> Meminimalkan **False Negative** adalah prioritas utama → optimalkan **Recall**.

### 4. Dataset Overview

| Properti | Nilai |
|---|---|
| **File** | `data_ecommerce_customer_churn.csv` |
| **Ukuran Awal** | 3,941 baris × 11 kolom |
| **Setelah Cleaning** | 3,270 baris |
| **Churn Rate** | 16.3% (534 churner dari 3,270 pelanggan) |

**Deskripsi Fitur:**

**👤 Customer Profile**
| Kolom | Tipe | Deskripsi |
|---|---|---|
| `PreferedOrderCat` | Kategorikal | Kategori pesanan yang paling sering dibeli bulan lalu |
| `MaritalStatus` | Kategorikal | Status pernikahan pelanggan |

**🛒 Customer Behavior** *(Top Feature Importance)*
| Kolom | Tipe | Deskripsi |
|---|---|---|
| `Tenure` | Numerik | Lama pelanggan bergabung di platform (bulan) |
| `WarehouseToHome` | Numerik | Jarak antara gudang dan rumah pelanggan (km) |
| `NumberOfDeviceRegistered` | Numerik | Jumlah perangkat terdaftar pada akun pelanggan |
| `NumberOfAddress` | Numerik | Jumlah alamat yang ditambahkan oleh pelanggan |
| `DaySinceLastOrder` | Numerik | Jumlah hari sejak pesanan terakhir |
| `CashbackAmount` | Numerik | Rata-rata cashback yang diterima bulan lalu |

**⭐ Customer Satisfaction**
| Kolom | Tipe | Deskripsi |
|---|---|---|
| `Complain` | Biner (0/1) | Apakah ada komplain yang diajukan bulan lalu |
| `SatisfactionScore` | Numerik (1–5) | Skor kepuasan layanan pelanggan |

**🎯 Target**
| Kolom | Tipe | Deskripsi |
|---|---|---|
| `Churn` | Biner (0/1) | `0` = tidak churn, `1` = churn |

---

## Data Cleaning, Feature Selection & Feature Engineering

### 1. Data Cleaning

| Langkah | Detail | Alasan |
|---|---|---|
| **Remove Duplicates** | 3,941 → 3,270 baris (671 duplikat dihapus) | Duplikat menyebabkan model belajar pola yang sama berulang → overfitting |
| **Label Consistency** | `'Mobile'` → `'Mobile Phone'` | Inkonsistensi label membuat model memperlakukan kategori sama sebagai dua hal berbeda |
| **Missing Values — Numerik** | Imputasi dengan **Median** | Robust terhadap outlier dibanding Mean |
| **Missing Values — Kategorikal** | Imputasi dengan **Most Frequent** | Mempertahankan distribusi kategori yang dominan |

### 2. Feature Selection

Korelasi fitur numerik terhadap target `Churn`:

| Rank | Fitur | Korelasi | Interpretasi |
|---|---|---|---|
| 1 | `Tenure` | **-0.336** | Semakin lama bergabung → lebih loyal → churn risk lebih rendah |
| 2 | `Complain` | **+0.262** | Pelanggan yang pernah komplain 2× lebih mungkin churn |
| 3 | `CashbackAmount` | **-0.151** | Cashback rendah → insentif kurang → lebih mudah churn |
| 4 | `DaysSinceLastOrder` | **-0.149** | Absen lama dari transaksi = sinyal churn dini |
| 5 | `NumberOfDeviceRegistered` | **+0.110** | Korelasi positif moderat dengan churn |

### 3. Feature Engineering

Dua fitur baru dibuat berdasarkan domain knowledge dan hipotesis bisnis:

| Fitur Baru | Definisi | Hipotesis |
|---|---|---|
| `NewCustomer` | Tenure ≤ 3 bulan | Pelanggan baru belum memiliki loyalitas → risiko churn lebih tinggi |
| `LowCashback` | Cashback di bawah Q1 | Insentif rendah = lebih mudah berpindah platform |

**⚠️ Feature Engineering Paradox**

Kedua fitur engineered justru **menurunkan performa model** karena memperkenalkan multikolinearitas — keduanya diturunkan langsung dari `Tenure` dan `CashbackAmount` yang sudah ada.

| Model | Recall | ROC-AUC |
|---|---|---|
| Random Forest **Sebelum** FE | 0.822 | 0.957 |
| Random Forest **Setelah** FE | 0.757 | 0.956 |

**Keputusan:** Fitur engineered **dihapus**. Baseline RF sudah mampu menangkap pola tersebut secara alami.

### 4. Data Transformation

**Scaling — RobustScaler (untuk fitur numerik)**
> Bekerja menggunakan **Median + Interquartile Range (IQR)**, sehingga nilai ekstrem atau outlier tidak mengganggu perhitungan skala data — lebih robust dibanding StandardScaler yang sensitif terhadap outlier.

**Encoding — OneHotEncoder (untuk fitur kategorikal)**
> Memecah setiap kategori menjadi kolom biner (0/1) agar algoritma dapat mempelajari pola tanpa mengasumsikan adanya urutan atau hierarki antar kategori.

### 5. Imbalanced Data Handling — SMOTE

Dataset sangat tidak seimbang: hanya **16.3% churner**. Tanpa penanganan, model akan bias dan selalu memprediksi "tidak churn".

| Kelas | Sebelum SMOTE | Setelah SMOTE |
|---|---|---|
| 0 (No Churn) | 2,189 | 2,189 |
| 1 (Churn) | 427 | 2,189 |

**SMOTE** (Synthetic Minority Over-sampling Technique) dipilih karena:
- Menciptakan sampel sintetis dengan variasi yang realistis (bukan sekadar duplikasi)
- Diterapkan **hanya pada training set** untuk menghindari data leakage

### 6. Full Preprocessing Pipeline

```
Raw Data (3,941 rows)
        │
        ▼
  Data Cleaning
  ├── Remove duplicates       → 3,270 rows
  └── Label consistency fix
        │
        ▼
  Train-Test Split (Stratified 80/20)
  ├── Train set: 80% (2,616 rows)
  └── Test set:  20%   (654 rows)
        │
        ▼
  Feature Engineering (Train only)         [dropped after FE paradox]
        │
        ▼
  Data Transformation (fit on Train, transform both)
  ├── Numerical  → Median Imputation + RobustScaler
  └── Categorical→ Mode Imputation  + OneHotEncoder
        │
        ▼
  SMOTE (Train set only)
  └── Minority class: 427 → 2,189 synthetic samples
        │
        ▼
  Balanced Train Set ready for Modeling
```

---

## Analytics — Algorithm & Evaluation Metrics

### 1. Evaluation Metrics

**Primary Metric: Recall (Sensitivity)**

```
Recall = TP / (TP + FN)
```

Recall dipilih karena dalam konteks churn prediction, **biaya False Negative jauh lebih besar** daripada False Positive. Gagal mendeteksi churner berarti kehilangan seluruh CLV pelanggan tersebut.

**Secondary Metrics:**

| Metric | Formula | Kegunaan dalam Konteks Bisnis |
|---|---|---|
| **Precision** | TP / (TP + FP) | Efisiensi biaya promosi — seberapa tepat sasaran kampanye retensi |
| **F1-Score** | 2 × (P × R) / (P + R) | Keseimbangan antara mendeteksi churner & efisiensi biaya |
| **ROC-AUC** | Area under ROC curve | Kemampuan model membedakan churner vs non-churner secara keseluruhan |

### 2. Cara Kerja Algoritma

<details>
<summary><b>📘 Logistic Regression</b></summary>

Logistic Regression adalah algoritma klasifikasi linear yang menghitung probabilitas suatu data termasuk kelas tertentu menggunakan **fungsi sigmoid**:

```
P(y=1|X) = 1 / (1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ))
```

- Output berupa probabilitas (0–1), lalu dikonversi ke kelas berdasarkan threshold
- Bekerja baik untuk data yang linear separable
- Mudah diinterpretasi karena koefisien menunjukkan kontribusi tiap fitur
- Tuning menggunakan regularisasi L1/L2 dan parameter C untuk mencegah overfitting

</details>

<details>
<summary><b>🌲 Random Forest — Final Model</b></summary>

Random Forest adalah algoritma **Ensemble Bagging** yang membangun banyak Decision Tree secara paralel, lalu menggabungkan prediksi dari semua pohon melalui **majority voting**.

**Cara kerja:**
1. Buat N bootstrap sample dari training data (sampling dengan pengembalian)
2. Untuk setiap sample, bangun 1 Decision Tree secara independen
3. Pada setiap node split, hanya subset fitur acak yang dipertimbangkan (*feature randomness*)
4. Prediksi akhir = voting mayoritas dari semua N pohon

**Mengapa RF dipilih sebagai final model?**
- Mengurangi variance dibanding single Decision Tree
- Robust terhadap overfitting
- Memberikan feature importance yang interpretatif
- Performa stabil dan konsisten

**Hyperparameter Tuning (RandomizedSearchCV):**
| Parameter | Fungsi |
|---|---|
| `max_depth` | Kedalaman maksimal tiap pohon — mengontrol kompleksitas model |
| `min_samples_split` | Jumlah minimum sampel untuk melakukan split pada node |
| `n_estimators` | Jumlah pohon yang dibangun dalam ensemble |

</details>

<details>
<summary><b>⚡ XGBoost</b></summary>

XGBoost adalah algoritma **Gradient Boosting** yang membangun pohon secara **sekuensial** — setiap pohon baru dibangun untuk memperbaiki kesalahan pohon sebelumnya.

**Cara kerja:**
1. Mulai dengan prediksi awal (biasanya rata-rata target)
2. Hitung residual error dari prediksi saat ini
3. Bangun pohon baru untuk memprediksi residual tersebut
4. Perbarui prediksi dengan menambahkan pohon baru (× learning rate)
5. Ulangi hingga N iterasi

**Perbedaan dengan Random Forest:**
- RF: pohon dibangun **paralel + independen**
- XGBoost: pohon dibangun **sekuensial + saling memperbaiki**

</details>

<details>
<summary><b>👥 K-Nearest Neighbors (KNN)</b></summary>

KNN adalah algoritma **instance-based learning** yang mengklasifikasikan data baru berdasarkan label mayoritas dari K tetangga terdekatnya dalam feature space.

**Cara kerja:**
1. Hitung jarak antara data baru dengan semua data training
2. Pilih K tetangga terdekat
3. Prediksi = kelas mayoritas dari K tetangga tersebut

KNN tidak membangun model eksplisit (*lazy learning*) — seluruh komputasi terjadi saat prediksi. Rentan terhadap data berdimensi tinggi dan skala fitur yang berbeda.

</details>

<details>
<summary><b>🌳 Decision Tree</b></summary>

Decision Tree membangun struktur pohon rekursif dengan membagi data berdasarkan fitur yang memberikan **Gini impurity** atau **information gain** terbaik di setiap node.

Kelemahan utama: sangat rentan terhadap overfitting — terbukti dari Train Recall = 1.0 tetapi Test Recall hanya 0.741 pada project ini.

</details>

### 3. Model Comparison

| Model | Karakteristik | Train Recall | Test Recall | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (base) | Linear, sigmoid function | 0.824 | 0.907 | 0.918 |
| Random Forest (base) | Ensemble Bagging | 1.000 | 0.822 | 0.957 |
| K-Neighbors | Instance-based, distance metrics | 0.994 | 0.804 | 0.868 |
| Decision Tree | Single tree, prone to overfitting | 1.000 | 0.741 | 0.811 |
| Logistic Regression (tuned) | L1/L2 regularization | 0.829 | 0.916 | 0.917 |
| **Random Forest (tuned) ⭐** | **Tuned: max_depth, min_samples_split** | **1.000** | **0.813** | **0.959** |
| XGBoost | Gradient Boosting, sequential trees | 0.982 | 0.795 | 0.938 |

### 4. Final Model: RF Tuned — Threshold 0.4

**Perbandingan 3 model terbaik pada threshold optimal:**

| Metric | LR Tuned (t=0.5) | **RF Tuned (t=0.4) ⭐** | XGBoost (t=0.5) |
|---|---|---|---|
| **Recall** | 0.916 | **0.916** | 0.794 |
| **Precision** | 0.384 | **0.681** | 0.739 |
| **F1-Score** | 0.541 | **0.781** | 0.766 |
| **ROC-AUC** | 0.917 | **0.959** | 0.955 |

**Alasan pemilihan RF Tuned (t=0.4):**
- ✅ **Recall setara** dengan LR Tuned (0.916) → 92 dari 100 churner terdeteksi
- ✅ **Precision jauh lebih baik** (0.681 vs 0.384) → promosi lebih tepat sasaran dan hemat biaya
- ✅ **F1-Score tertinggi** (0.781) → keseimbangan Precision-Recall terbaik
- ✅ **ROC-AUC tertinggi** (0.959) → kemampuan diskriminasi terbaik di antara semua model

**Dampak bisnis — perbandingan dengan dan tanpa model (test set: 654 pelanggan, 107 churner):**

| Skenario | Churner Terdeteksi (TP) | Churner Terlewat (FN) | Promo Salah Sasaran (FP) | Recall |
|---|---|---|---|---|
| Tanpa Model (semua = no churn) | 0 | 107 | 0 | 0.000 |
| Prediksi Acak | 43 | 64 | 277 | 0.402 |
| **RF Tuned (t=0.4)** | **98** | **9** | **46** | **0.916** |

> 💡 Pada skala penuh (534 churner), model mampu menjangkau **489 dari 534 churner** dengan promosi yang lebih terarah dibanding model alternatif.

**Top 10 Feature Importances (RF Tuned):**

| Rank | Fitur | Korelasi dengan Churn |
|---|---|---|
| 1 | `Tenure` | -0.336 |
| 2 | `Complain` | +0.262 |
| 3 | `CashbackAmount` | -0.151 |
| 4 | `DaysSinceLastOrder` | -0.149 |
| 5 | `NumberOfDeviceRegistered` | +0.110 |
| 6 | `WarehouseToHome` | — |
| 7 | `NumberOfAddress` | — |
| 8 | `SatisfactionScore` | — |
| 9 | `PreferedOrderCat_Mobile Phone` | — |
| 10 | `MaritalStatus_Single` | — |

---

## Conclusion & Recommendation

### 1. Kesimpulan

**Key Findings:**
1. **Churn rate aktual: 16.3%** (534 dari 3,270 pelanggan)
2. **Pelanggan baru** (Tenure ≤ 3 bulan) memiliki **risiko churn tertinggi** — korelasi -0.35
3. **Pelanggan yang pernah komplain 2× lebih mungkin churn**
4. **Cashback rendah + lama tidak order** = sinyal churn dini
5. **Kombinasi jarak gudang jauh + riwayat komplain** = segmen risiko tertinggi

**Final Model: Tuned Random Forest (threshold 0.4)**
- Recall **0.916** → 92 dari 100 churner terdeteksi
- F1 **0.781** | ROC-AUC **0.959** → terbaik dari 7 model yang diuji
- Estimasi dampak bisnis: churn rate turun **~27–36%** jika 30–40% churner yang terdeteksi berhasil dipertahankan

### 2. Kapan Model Dapat Dipercaya?

| Kondisi | Kepercayaan Model |
|---|---|
| Pelanggan dengan data lengkap dan pola transaksi normal | ✅ **Dapat dipercaya** |
| Pelanggan baru (< 1 bulan), data banyak missing value | ⚠️ **Kurang dapat dipercaya** |
| Perubahan tren pasar akibat kompetitor baru | ⚠️ **Perlu revalidasi** |
| Periode promosi besar (Harbolnas, flash sale) | ⚠️ **Anomali musiman — perhatikan konteks** |

### 3. Keterbatasan Project

| Kategori | Keterbatasan | Rekomendasi ke Depan |
|---|---|---|
| **Data historis** | Tidak menangkap perubahan perilaku akibat kompetitor baru | Tambahkan data kompetitor & tren pasar |
| **Fitur terbatas** | Belum ada frekuensi transaksi dan nilai pesanan | Tambahkan `order_frequency`, `avg_order_value`, `total_spend` |
| **SMOTE** | Data sintetis — performa perlu revalidasi berkala | Uji teknik lain: ADASYN, `class_weight='balanced'` |
| **XGBoost** | Tuning belum optimal — performa default mendekati RF Tuned | Lakukan hyperparameter tuning penuh untuk perbandingan yang adil |
| **Data eksternal** | Belum digunakan | Integrasikan: CPI (BPS), kalender promosi, sentimen review produk |

### 4. Actionable Recommendations

| # | Insight | Aksi yang Direkomendasikan |
|---|---|---|
| 1 | Tenure rendah = risiko churn tinggi | Buat **program onboarding & loyalty reward** khusus 3 bulan pertama (extra cashback, welcome voucher) |
| 2 | Komplain → churn meningkat | Bangun **sistem respons komplain cepat**: SLA resolusi < 24 jam + follow-up proaktif pasca penyelesaian |
| 3 | Cashback rendah = lebih mudah churn | Tawarkan **cashback yang dipersonalisasi** untuk pelanggan dengan riwayat cashback rendah |
| 4 | Lama tidak order | Kirim **notifikasi re-engagement** (push/email) setelah X hari inaktivitas + diskon/promo |
| 5 | Jarak jauh + riwayat komplain | Buat **segmen prioritas khusus**: `WarehouseToHome > median` AND `Complain = 1` → intervensi proaktif sebelum churn |
| 6 | Model tersedia | **Integrasikan model ke CRM**: scoring churn pelanggan secara mingguan/bulanan → trigger kampanye retensi otomatis untuk pengguna high-risk |

---

### Load Saved Model
```python
import pickle

# Load final model
with open('model/best_model_rf_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict with optimal threshold (0.4)
y_pred_proba = model.predict_proba(X_new)[:, 1]
y_pred = (y_pred_proba >= 0.4).astype(int)
```

---

## 🔗 Link Deliverables

| Deliverables | Link |
|---|---|
| 📓 Jupyter Notebook | *https://github.com/lauziafadhilanareswari/capstone3-purwadhika.git* |
| 🎞️ Slide Presentasi | *https://drive.google.com/file/d/1KKxeZy9rd-BRHAkro6ve48C1zcq07T8O/view?usp=sharing* |
| 🎥 Video Penjelasan | *YouTube: https://youtu.be/vw33FPv63F0 / Google Drive: https://drive.google.com/file/d/15IydmfIhRSz0km7z0MKvWqxFp2f9d3gt/view?usp=sharing* |