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
pada task berikutnya.

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
`preprocessing/artifacts/`. Nama file dan skema output final akan ditetapkan
bersamaan dengan implementasi `automate_kuzanf3b.py`.