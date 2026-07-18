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

## Membentuk Dataset NER

Setelah validasi BC5CDR lulus, bentuk dataset NER dengan label BIO:

```bash
.venv/Scripts/python.exe scripts/build_ner_dataset.py
```

Output utama:

```text
data/processed/ner/train.jsonl
data/processed/ner/dev.jsonl
data/processed/ner/test.jsonl
data/processed/ner/label_map.json
results/ner_preprocessing_report.json
```

Script ini menggunakan tokenizer PubMedBERT dari `configs/config_ner.yaml`, menyelaraskan label dengan `offset_mapping`, dan memakai sliding window untuk dokumen yang melebihi `max_sequence_length`.

## Smoke Test Training NER

Sebelum training penuh, jalankan smoke test kecil:

```bash
.venv/Scripts/python.exe scripts/train_ner.py --smoke-test --cpu
```

Smoke test hanya memakai sedikit batch untuk memastikan dataset, model, loss, evaluasi, dan penyimpanan metrik berjalan.

Output training NER:

```text
checkpoints/ner/
logs/ner/
predictions/ner/
results/ner/
```

Training penuh PubMedBERT sebaiknya dijalankan dengan GPU, misalnya melalui Colab.

## Training NER Full di Google Colab

Notebook siap pakai tersedia di `Colab_NER_Training.ipynb`.

Cara paling praktis adalah menyimpan project ini di GitHub, lalu clone repository dari Colab. Aktifkan GPU terlebih dahulu melalui:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Clone repository dan masuk ke folder project:

```python
!git clone https://github.com/USERNAME/enexre.git
%cd enexre
```

Jika repository private, gunakan GitHub personal access token:

```python
!git clone https://TOKEN@github.com/USERNAME/enexre.git
%cd enexre
```

Install dependency:

```python
!pip install -r requirements.txt
```

Cek GPU:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

Pastikan file dataset NER sudah tersedia:

```python
!ls data/processed/ner
```

Minimal harus ada:

```text
train.jsonl
dev.jsonl
test.jsonl
label_map.json
```

Jika file processed belum ikut di-push ke GitHub, bentuk ulang dataset dari file BC5CDR lokal:

```python
!python scripts/validate_bc5cdr.py
!python scripts/build_ner_dataset.py
```

Jalankan smoke test di Colab:

```python
!python scripts/train_ner.py --smoke-test --cleanup-smoke-checkpoint
```

Jika smoke test berhasil, jalankan training penuh satu konfigurasi default:

```python
!python scripts/train_ner.py
```

Untuk eksperimen manual, misalnya satu seed dan learning rate tertentu:

```python
!python scripts/train_ner.py --seed 13 --learning-rate 3e-5 --batch-size 8 --run-name seed13_lr3e-5_bs8
```

Untuk seleksi model yang lebih hemat waktu, coba beberapa learning rate dengan seed yang sama:

```python
%%bash
python scripts/train_ner.py --seed 13 --learning-rate 1e-5 --batch-size 8 --run-name seed13_lr1e-5_bs8
python scripts/train_ner.py --seed 13 --learning-rate 3e-5 --batch-size 8 --run-name seed13_lr3e-5_bs8
python scripts/train_ner.py --seed 13 --learning-rate 5e-5 --batch-size 8 --run-name seed13_lr5e-5_bs8
```

Pilih konfigurasi berdasarkan nilai `best_dev_f1` pada:

```text
results/ner/*_metrics.json
```

Simpan hasil training ke Google Drive agar checkpoint tidak hilang saat runtime Colab berhenti:

```python
from google.colab import drive

drive.mount("/content/drive")
```

```python
!mkdir -p /content/drive/MyDrive/enexre_outputs
!cp -r results/ner /content/drive/MyDrive/enexre_outputs/results_ner
!cp -r logs/ner /content/drive/MyDrive/enexre_outputs/logs_ner
!cp -r predictions/ner /content/drive/MyDrive/enexre_outputs/predictions_ner
!cp -r checkpoints/ner /content/drive/MyDrive/enexre_outputs/checkpoints_ner
```

Jangan push `checkpoints/ner/` ke GitHub biasa kecuali memakai Git LFS, karena ukuran model dapat besar. Untuk pelaporan eksperimen, artefak yang penting adalah metrik di `results/ner/`, prediksi di `predictions/ner/`, dan checkpoint model terbaik.

## Evaluasi NER Terbaik

Model terbaik dipilih berdasarkan `best_dev_f1` tertinggi pada file:

```text
results/ner/*_metrics.json
```

Setelah training selesai, evaluasi checkpoint terbaik pada test set:

```bash
python scripts/evaluate_ner.py
```

Secara default, script ini mengabaikan smoke test, memilih checkpoint full training terbaik yang masih tersedia, lalu mengevaluasi:

```text
data/processed/ner/test.jsonl
```

Output evaluasi final:

```text
results/ner/best_test_metrics.json
predictions/ner/best_test_predictions.jsonl
```

Jika ingin mengevaluasi checkpoint tertentu:

```bash
python scripts/evaluate_ner.py --checkpoint checkpoints/ner/seed13_lr3e-5_bs8
```

Metrik utama yang dilaporkan adalah entity-level precision, recall, dan F1 dari `seqeval`. Test set sebaiknya hanya dipakai setelah konfigurasi model dipilih dari development set.

## Catatan Reproducibility

Gunakan random seed yang tercatat di `configs/config_ner.yaml` dan `configs/config_re.yaml`.
Setelah dependency final stabil, simpan versi aktual dengan:

```bash
.venv/Scripts/python.exe -m pip freeze > requirements.lock.txt
```
