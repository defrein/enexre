# ENEXRE BC5CDR Experiment

Kerangka eksperimen untuk ekstraksi entitas dan relasi Chemical-Disease pada BC5CDR.

## Setup

Jalankan dari terminal VS Code:

```bash
bash setup_venv.bash
```

Setelah selesai, buka `Protocol.ipynb`, lalu pilih kernel:

```text
Python (enexre)
```

## Struktur

```text
configs/        konfigurasi NER dan RE
data/           dataset dan manifest lokal
checkpoints/    model terbaik hasil training
predictions/    hasil prediksi test
results/        metrik dan tabel hasil
logs/           log training dan environment
Protocol.ipynb  notebook eksperimen awal
```

## Urutan Penelitian

Rujukan utama ada di `PENELITIAN_STEP.md`. Secara ringkas:

1. Validasi dataset BC5CDR.
2. Bentuk data NER.
3. Latih dan evaluasi NER.
4. Bentuk kandidat RE dengan gold entities.
5. Uji baseline co-occurrence.
6. Latih dan evaluasi RE.
7. Uji pipeline NER-RE.
8. Simpan konfigurasi, log, prediksi, dan hasil.

## Validasi Dataset

Letakkan file BC5CDR PubTator resmi pada struktur berikut:

```text
data/bc5cdr/train.txt
data/bc5cdr/dev.txt
data/bc5cdr/test.txt
```

Lalu jalankan:

```bash
.venv/Scripts/python.exe scripts/validate_bc5cdr.py
```

Hasil validasi disimpan ke:

```text
results/dataset_validation.json
```

Script validasi mengecek jumlah dokumen, duplikasi PMID antar subset, validitas offset anotasi, tipe entitas, MeSH ID, relasi CID, dan duplikasi relasi.

## Catatan Reproducibility

Gunakan random seed yang tercatat di `configs/config_ner.yaml` dan `configs/config_re.yaml`.
Setelah dependency final stabil, simpan versi aktual dengan:

```bash
.venv/Scripts/python.exe -m pip freeze > requirements.lock.txt
```
