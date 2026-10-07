# Eksperimen SML — Telco Customer Churn

Eksperimen ini menggunakan dataset **Telco Customer Churn** untuk masalah
binary classification: memprediksi apakah pelanggan akan churn berdasarkan
karakteristik layanan dan akun pelanggan. Target yang digunakan adalah kolom
`Churn`.

## Sumber data

- Sumber: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Nama file sumber: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Salinan raw yang digunakan project:
  `dataset/raw/telco_churn_raw.csv`

Dataset raw dipertahankan apa adanya. Perubahan data hanya dilakukan pada
notebook eksperimen dan pipeline preprocessing pada task berikutnya.

## Struktur project

```text
Eksperimen_SML_kuzanf3b/
├── dataset/
│   ├── raw/
│   │   └── telco_churn_raw.csv
│   └── preprocessed/
│       └── .gitkeep
├── preprocessing/
│   ├── Template_Eksperimen_MSML.ipynb
│   ├── Eksperimen_kuzanf3b.ipynb
│   └── artifacts/
│       └── .gitkeep
├── requirements.txt
└── README.md
```

`Eksperimen_kuzanf3b.ipynb` adalah salinan kerja dari template resmi. Urutan
tahap template dipertahankan: perkenalan dataset, import library, memuat
dataset, EDA, dan data preprocessing. Isi eksperimen manual akan dilengkapi
pada notebook tersebut, termasuk pemeriksaan kualitas data, distribusi target,
dan visualisasi distribusi fitur.

## Menjalankan eksperimen

Gunakan Python 3.12.7 dan buat environment terisolasi:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab
```

Buka `preprocessing/Eksperimen_kuzanf3b.ipynb` dari root repository. Notebook
menggunakan dataset lokal melalui path relatif agar dapat dijalankan ulang tanpa
bergantung pada URL eksternal. Jika notebook dibuka dari direktori berbeda,
jalankan kernel dengan working directory root repository.

## Konvensi output

Hasil dataset siap latih akan ditulis ke `dataset/preprocessed/`. Artefak
pendukung preprocessing seperti metadata dan preprocessor akan ditulis ke
`preprocessing/artifacts/`.

## Hasil eksperimen manual

Notebook menghasilkan dua file untuk pemeriksaan awal:

- `dataset/preprocessed/telco_churn_train.csv`
- `dataset/preprocessed/telco_churn_test.csv`

Preprocessing yang diterapkan adalah konversi `TotalCharges` ke numerik,
imputasi median untuk fitur numerik, imputasi modus untuk fitur kategorikal,
one-hot encoding kategori, dan standardisasi fitur numerik. `customerID`
dihapus karena merupakan identifier. Split dilakukan dengan `stratify=y` dan
`random_state=42`; seluruh transformer di-fit hanya pada data train.

Pada dataset ini, output berisi 5.634 baris train, 1.409 baris test, 45 fitur
hasil transformasi, dan kolom target `Churn` bernilai `0` atau `1`. Output
tidak memiliki missing value setelah preprocessing. File hasil ini adalah
artefak eksperimen manual dan akan direproduksi oleh `automate_kuzanf3b.py`
pada task berikutnya.

## Preprocessing otomatis (Task 3)

Script otomatis tersedia pada `preprocessing/automate_kuzanf3b.py`.

Jalankan dari root repository:

```bash
python preprocessing/automate_kuzanf3b.py
```

Opsional (override path):

```bash
python preprocessing/automate_kuzanf3b.py \
  --input-path dataset/raw/telco_churn_raw.csv \
  --output-dir dataset/preprocessed \
  --artifacts-dir preprocessing/artifacts
```

Output otomatis:

- `dataset/preprocessed/telco_churn_train.csv`
- `dataset/preprocessed/telco_churn_test.csv`
- `preprocessing/artifacts/preprocessor.joblib`
- `preprocessing/artifacts/feature_names.json`
- `preprocessing/artifacts/preprocessing_metadata.json`

## Workflow preprocessing (GitHub Actions)

Workflow ada di `.github/workflows/preprocess.yml` dan akan berjalan saat:

- `workflow_dispatch` (manual trigger),
- push/pull request yang menyentuh dataset raw, script preprocessing, requirements, atau file workflow.

Workflow melakukan install dependency, menjalankan `automate_kuzanf3b.py`, lalu
mengunggah hasil preprocessing dan artefak sebagai GitHub Actions artifact
`telco-churn-preprocessed`.