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

## Status Hasil Saat Ini

Hasil RE final sementara sudah tersedia sebagai evaluasi single-seed pada test set dengan gold entities. Konfigurasi yang dipilih berasal dari development set:

```text
run: re_seed13_lr3e-5_bs8
checkpoint: checkpoints/re/re_seed13_lr3e-5_bs8
learning_rate: 3e-5
batch_size: 8
seed: 13
best_epoch: 4
threshold: 0.70
```

Artefak utama:

```text
results/re/best_test_metrics.json
results/re/final_re_summary.json
predictions/re/best_test_predictions.jsonl
checkpoints/re/re_seed13_lr3e-5_bs8/
```

Ringkasan test set:

| Metode | Precision | Recall | F1 | False Positive |
| --- | ---: | ---: | ---: | ---: |
| Co-occurrence baseline | 0.1972 | 1.0000 | 0.3295 | 4339 |
| PubMedBERT RE | 0.6694 | 0.6914 | 0.6802 | 364 |

Peningkatan terhadap baseline:

```text
Delta F1 = +0.3507
False positive reduction = 91.61%
```

Hasil ini dikunci sebagai hasil RE final sementara untuk melanjutkan pipeline NER-RE dan integrasi Neo4j. Fine-tuning multi-seed dapat diulang setelah pipeline end-to-end stabil.

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

## Membentuk Dataset RE dan Baseline Co-occurrence

Setelah hasil NER final tersedia, bentuk kandidat Relation Extraction menggunakan gold entities BC5CDR:

```bash
.venv/Scripts/python.exe scripts/build_re_dataset.py
```

Output utama:

```text
data/processed/re/train.jsonl
data/processed/re/dev.jsonl
data/processed/re/test.jsonl
results/re_preprocessing_report.json
```

Script ini membentuk seluruh pasangan konsep Chemical-Disease unik per dokumen, memberi label `1` untuk relasi CID dan `0` untuk non-CID, lalu membuat `marked_text` dengan marker:

```text
[CHEM] ... [/CHEM]
[DISEASE] ... [/DISEASE]
```

Baseline co-occurrence juga dihitung dengan menganggap seluruh kandidat sebagai CID. Hasil saat ini:

| Subset | CID | Non-CID | Total pasangan | Baseline F1 |
| --- | ---: | ---: | ---: | ---: |
| train | 1038 | 4394 | 5432 | 0.3209 |
| dev | 1012 | 4249 | 5261 | 0.3227 |
| test | 1066 | 4339 | 5405 | 0.3295 |

Sebagian `marked_text` melebihi 512 token. `scripts/train_re.py` memakai crop/window di sekitar marker target; jika marker masih tidak lengkap karena mention terlalu berjauhan, script memakai fallback dua snippet lokal agar marker Chemical dan Disease tetap masuk ke input model.

## Smoke Test Training RE

Jalankan smoke test kecil untuk memastikan dataset RE, marker, cropping, loss, threshold selection, dan penyimpanan artefak berjalan:

```bash
.venv/Scripts/python.exe scripts/train_re.py --smoke-test --cpu --cleanup-smoke-checkpoint
```

Output training RE:

```text
checkpoints/re/
logs/re/
predictions/re/
results/re/
```

Training penuh PubMedBERT RE sebaiknya dijalankan dengan GPU:

```bash
python scripts/train_re.py --seed 13 --learning-rate 1e-5 --batch-size 8 --run-name seed13_lr1e-5_bs8
```

Notebook Colab untuk training RE penuh tersedia di:

```text
Colab_RE_Training.ipynb
```

Evaluasi checkpoint RE pada test set:

```bash
python scripts/evaluate_re.py
```

Secara default evaluator memilih checkpoint non-smoke terbaik dari `results/re/*_metrics.json`, memakai threshold terbaik dari development set, lalu menulis:

```text
results/re/best_test_metrics.json
predictions/re/best_test_predictions.jsonl
```

Hasil final sementara yang sudah dibuat:

```text
checkpoint=checkpoints/re/re_seed13_lr3e-5_bs8
threshold=0.70
test_precision=0.6694
test_recall=0.6914
test_f1=0.6802
```

Ringkasan ringkas tersimpan di:

```text
results/re/final_re_summary.json
```

## Evaluasi Pipeline NER-RE

Setelah prediksi NER final dan checkpoint RE final tersedia, jalankan evaluasi
pipeline end-to-end pada test set:

```bash
.venv/Scripts/python.exe scripts/evaluate_pipeline.py --cpu
```

Script ini membentuk kandidat dari span hasil prediksi NER. Untuk evaluasi,
span prediksi dipetakan ke MeSH ID gold hanya jika posisi dan tipe entitas cocok
exact match. Relasi gold yang tidak bisa terbentuk karena entitas NER terlewat
dihitung sebagai false negative pipeline.

Output utama:

```text
data/processed/pipeline/test_candidates.jsonl
results/pipeline/best_test_metrics.json
predictions/pipeline/best_test_predictions.jsonl
```

Hasil test set saat ini:

| Pengujian | Precision | Recall | F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RE dengan gold entities | 0.6694 | 0.6914 | 0.6802 | 737 | 364 | 329 |
| Pipeline NER-RE | 0.6875 | 0.6088 | 0.6458 | 649 | 295 | 417 |

Ringkasan kandidat pipeline:

```text
candidate_pairs=4401
gold_cid_relations=1066
predicted_spans=10367
matched_spans=8853
unmatched_spans=1514
documents_without_candidates=9
threshold=0.70
```

## Catatan Reproducibility

Gunakan random seed yang tercatat di `configs/config_ner.yaml` dan `configs/config_re.yaml`.
Setelah dependency final stabil, simpan versi aktual dengan:

```bash
.venv/Scripts/python.exe -m pip freeze > requirements.lock.txt
```
