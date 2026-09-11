# Langkah-Langkah Pengujian Penelitian

## Tahap 1 — Menetapkan Protokol Pengujian

Sebelum pelatihan dilakukan, ditetapkan protokol pengujian agar keputusan penelitian tidak berubah setelah hasil data *test* diketahui.

Protokol yang digunakan adalah:

1. data *training* digunakan untuk melatih model;
2. data *development* digunakan untuk memilih parameter, model terbaik, dan threshold;
3. data *test* hanya digunakan untuk pengujian akhir;
4. pembagian resmi BC5CDR tidak diubah;
5. seluruh hasil eksperimen disimpan berdasarkan konfigurasi dan *random seed*;
6. metrik utama yang digunakan adalah Precision, Recall, dan F1-Score.

Model akhir dijalankan menggunakan minimal tiga *random seed*, misalnya:

```text
13, 42, dan 100
```

Nilai *random seed* yang benar-benar digunakan harus dicantumkan dalam laporan.

---

## Tahap 2 — Memvalidasi Dataset BC5CDR

Dataset BC5CDR terdiri atas:

| Subset      | Jumlah artikel |
| ----------- | -------------: |
| Training    |            500 |
| Development |            500 |
| Test        |            500 |
| Total       |          1.500 |

Pemeriksaan dilakukan menggunakan program untuk memastikan:

1. tidak ada PMID yang muncul pada lebih dari satu subset;
2. setiap artikel memiliki judul atau abstrak;
3. posisi awal dan akhir anotasi sesuai dengan teks;
4. tipe entitas hanya terdiri atas `Chemical` dan `Disease`;
5. setiap anotasi memiliki MeSH ID;
6. setiap relasi CID menunjuk pada Chemical ID dan Disease ID yang tersedia dalam artikel;
7. tidak terdapat relasi yang sama lebih dari satu kali.

Hasil pemeriksaan disimpan dalam tabel berikut.

| Pemeriksaan                 | Training | Development |    Test |
| --------------------------- | -------: | ----------: | ------: |
| Jumlah dokumen              |  [HASIL] |     [HASIL] | [HASIL] |
| Jumlah mention Chemical     |  [HASIL] |     [HASIL] | [HASIL] |
| Jumlah mention Disease      |  [HASIL] |     [HASIL] | [HASIL] |
| Jumlah konsep Chemical unik |  [HASIL] |     [HASIL] | [HASIL] |
| Jumlah konsep Disease unik  |  [HASIL] |     [HASIL] | [HASIL] |
| Jumlah relasi CID           |  [HASIL] |     [HASIL] | [HASIL] |
| Anotasi tidak valid         |  [HASIL] |     [HASIL] | [HASIL] |

Apabila ditemukan anotasi tidak valid, data tidak langsung diperbaiki secara manual. Kasus tersebut dicatat terlebih dahulu dan penanganannya dilaporkan.

---

## Tahap 3 — Membentuk Data NER

Judul dan abstrak digabungkan menjadi satu teks dengan pemisah yang konsisten.

```text
judul + " " + abstrak
```

Anotasi Chemical dan Disease kemudian dikonversi menjadi label BIO:

```text
B-Chemical
I-Chemical
B-Disease
I-Disease
O
```

Tokenisasi dilakukan menggunakan tokenizer dari checkpoint PubMedBERT. Label disejajarkan dengan token menggunakan informasi posisi karakter atau `offset_mapping`.

Token khusus seperti `[CLS]`, `[SEP]`, dan `[PAD]` tidak dihitung dalam fungsi *loss*.

Setelah konversi, dilakukan pemeriksaan otomatis:

1. teks hasil rekonstruksi sama dengan teks asli;
2. token berlabel Chemical sesuai dengan anotasi Chemical;
3. token berlabel Disease sesuai dengan anotasi Disease;
4. tidak ada label `I-Chemical` tanpa label awal yang sesuai;
5. tidak ada label `I-Disease` tanpa label awal yang sesuai.

Sebanyak `[JUMLAH SAMPEL]` dokumen juga diperiksa secara manual untuk memastikan label BIO telah terbentuk dengan benar.

---

## Tahap 4 — Melatih dan Memilih Model NER

Model yang digunakan adalah PubMedBERT dengan lapisan klasifikasi token.

Contoh ruang pencarian parameter:

| Parameter               | Kandidat nilai   |
| ----------------------- | ---------------- |
| Learning rate           | 1e-5, 3e-5, 5e-5 |
| Batch size              | 8, 16            |
| Maximum epoch           | 10               |
| Early stopping patience | 2                |
| Maximum sequence length | 512              |
| Weight decay            | 0,01             |
| Dropout                 | 0,1              |

Langkah pelatihan:

1. model dilatih menggunakan *training set*;
2. performa dihitung pada *development set* setiap epoch;
3. checkpoint dengan F1-Score tertinggi disimpan;
4. pelatihan dihentikan apabila F1-Score tidak meningkat selama `[PATIENCE]` epoch;
5. konfigurasi dengan F1-Score development tertinggi dipilih;
6. konfigurasi tersebut dibekukan sebelum menguji *test set*.

Selama pemilihan model, *test set* tidak boleh dibuka untuk menentukan parameter.

---

## Tahap 5 — Menguji Model NER

Model NER diuji menggunakan *test set* dengan metode *entity-level exact match*.

Sebuah prediksi dinyatakan benar apabila:

1. posisi awal entitas sama;
2. posisi akhir entitas sama; dan
3. tipe entitas sama.

Hasil dilaporkan secara terpisah untuk Chemical dan Disease.

| Entitas       | Precision |  Recall | F1-Score |
| ------------- | --------: | ------: | -------: |
| Chemical      |   [HASIL] | [HASIL] |  [HASIL] |
| Disease       |   [HASIL] | [HASIL] |  [HASIL] |
| Micro average |   [HASIL] | [HASIL] |  [HASIL] |

Selain nilai metrik, dicatat:

* jumlah True Positive;
* jumlah False Positive;
* jumlah False Negative;
* contoh kesalahan batas entitas;
* contoh kesalahan tipe entitas;
* entitas yang gagal dikenali karena tokenisasi.

Model akhir dijalankan menggunakan tiga *random seed*. Hasil dilaporkan dalam bentuk:

```text
F1-Score rata-rata ± standar deviasi
```

---

## Tahap 6 — Membentuk Kandidat Relation Extraction

Pengujian utama RE menggunakan entitas referensi atau *gold entities* dari BC5CDR. Hal ini dilakukan agar kemampuan RE dapat diukur tanpa dipengaruhi kesalahan NER.

Untuk setiap dokumen:

1. ambil seluruh Chemical ID unik;
2. ambil seluruh Disease ID unik;
3. bentuk seluruh kombinasi Chemical–Disease;
4. bandingkan setiap pasangan dengan anotasi CID.

Apabila suatu dokumen memiliki (m) Chemical ID dan (n) Disease ID, jumlah kandidat pasangan adalah:

[
N_{\text{kandidat}}=m \times n
]

Unit data RE adalah:

```text
(PMID, Chemical MeSH ID, Disease MeSH ID)
```

Contoh:

```text
(354896, D008012, D006323)
```

Pelabelan dilakukan sebagai berikut:

```text
1 = pasangan tercatat sebagai CID
0 = pasangan tidak tercatat sebagai CID
```

Jumlah pasangan dicatat dalam tabel berikut.

| Subset      |     CID | Non-CID | Total pasangan |
| ----------- | ------: | ------: | -------------: |
| Training    | [HASIL] | [HASIL] |        [HASIL] |
| Development | [HASIL] | [HASIL] |        [HASIL] |
| Test        | [HASIL] | [HASIL] |        [HASIL] |

Seluruh kandidat pada *development* dan *test set* harus digunakan. Data negatif tidak boleh dibuang pada proses evaluasi.

---

## Tahap 7 — Menguji Baseline Co-occurrence

Baseline digunakan untuk membuktikan bahwa penambahan model RE memberikan peningkatan.

Pada baseline, seluruh pasangan Chemical–Disease yang muncul dalam dokumen yang sama dianggap sebagai relasi CID.

Prosedurnya adalah:

1. bentuk seluruh kandidat Chemical–Disease pada data *test*;
2. beri prediksi CID kepada seluruh kandidat;
3. bandingkan dengan relasi CID referensi;
4. hitung TP, FP, FN, Precision, Recall, dan F1-Score.

Tabel hasil baseline:

| Metrik         |   Nilai |
| -------------- | ------: |
| True Positive  |    1066 |
| False Positive |    4339 |
| False Negative |       0 |
| Precision      |  0.1972 |
| Recall         |  1.0000 |
| F1-Score       |  0.3295 |

Pada baseline dengan *gold entities*, nilai Recall dapat sangat tinggi atau mencapai 100% karena semua pasangan dianggap positif. Kelemahan baseline biasanya terlihat pada jumlah False Positive dan nilai Precision.

---

## Tahap 8 — Membentuk Input Model RE

Untuk setiap pasangan konsep target, seluruh penyebutan yang sesuai diberi penanda khusus.

```text
[CHEM] ... [/CHEM]
[DISEASE] ... [/DISEASE]
```

Contoh:

```text
[CHEM] Lidocaine [/CHEM] was administered to the patient.
The patient subsequently developed
[DISEASE] heart arrest [/DISEASE].
```

Token marker ditambahkan ke tokenizer:

```text
[CHEM]
[/CHEM]
[DISEASE]
[/DISEASE]
```

Setelah token ditambahkan, ukuran embedding model disesuaikan.

Pemeriksaan dilakukan untuk memastikan:

1. marker Chemical berada pada entitas Chemical target;
2. marker Disease berada pada entitas Disease target;
3. pasangan lain dalam dokumen tidak diberi marker target;
4. label CID sesuai dengan anotasi BC5CDR;
5. marker tidak hilang akibat pemotongan teks.

Apabila panjang masukan melebihi 512 token, digunakan satu aturan tetap yang menjamin kedua marker tetap berada dalam masukan. Aturan yang digunakan dicatat sebagai:

```text
[STRATEGI PENANGANAN TEKS PANJANG]
```

Jumlah data yang mengalami pemotongan juga harus dilaporkan.

---

## Tahap 9 — Melatih dan Memilih Model RE

Model RE menggunakan PubMedBERT dengan lapisan klasifikasi biner.

Contoh konfigurasi:

| Parameter               | Kandidat nilai   |
| ----------------------- | ---------------- |
| Learning rate           | 1e-5, 3e-5, 5e-5 |
| Batch size              | 8, 16            |
| Maximum epoch           | 10               |
| Maximum sequence length | 512              |
| Weight decay            | 0,01             |
| Dropout                 | 0,1              |
| Optimizer               | AdamW            |

Apabila data tidak seimbang, bobot kelas dihitung hanya menggunakan *training set*:

[
w_{\text{CID}}=
\frac{N_{\text{non-CID}}}{N_{\text{CID}}}
]

Langkah pengujian:

1. model dilatih menggunakan data RE *training*;
2. model dievaluasi pada *development set*;
3. checkpoint terbaik dipilih berdasarkan F1-Score CID;
4. konfigurasi terbaik dibekukan;
5. threshold ditentukan menggunakan *development set*.

Threshold yang dapat diuji:

[
\tau \in
{0{,}30,\ 0{,}40,\ 0{,}50,\ 0{,}60,\ 0{,}70}
]

Pasangan diprediksi sebagai CID apabila:

[
P(CID)\geq \tau
]

Threshold dengan F1-Score development tertinggi digunakan untuk *test set*.

---

## Tahap 10 — Menguji Model RE dengan Gold Entities

Model RE diuji menggunakan seluruh kandidat pasangan pada *test set*.

Hasil yang dicatat:

| Metrik         |   Nilai |
| -------------- | ------: |
| True Positive  |     737 |
| False Positive |     364 |
| False Negative |     329 |
| Precision      |  0.6694 |
| Recall         |  0.6914 |
| F1-Score       |  0.6802 |
| Threshold      |    0.70 |

Pengujian ini menjadi hasil utama model RE karena pasangan entitas yang digunakan berasal dari anotasi referensi.

Model RE final dipilih dari tiga *random seed* pada konfigurasi learning rate 3e-5 dan batch size 8. Pemilihan dilakukan berdasarkan F1 tertinggi pada development set:

| Run | Seed | Best epoch | Threshold | Dev Precision | Dev Recall | Dev F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| re_seed13_lr3e-5_bs8 | 13 | 4 | 0.70 | 0.7016 | 0.7273 | 0.7142 |
| re_seed42_lr3e-5_bs8 | 42 | 3 | 0.70 | 0.6949 | 0.7292 | 0.7117 |
| re_seed100_lr3e-5_bs8 | 100 | 6 | 0.60 | 0.6592 | 0.7569 | 0.7047 |

Rata-rata F1 development tiga seed adalah 0.7102 dengan standard deviation 0.0049. Checkpoint `seed13_lr3e-5_bs16_ep20` tidak digunakan dalam laporan karena hanya merupakan uji coba eksploratif.

Checkpoint final:

```text
run = re_seed13_lr3e-5_bs8
learning_rate = 3e-5
batch_size = 8
seed = 13
best_epoch = 4
checkpoint = checkpoints/re/re_seed13_lr3e-5_bs8
```

Model dijalankan menggunakan minimal tiga *random seed* dan dilaporkan sebagai:

```text
Mean F1 ± standard deviation
```

---

## Tahap 11 — Menguji Pipeline NER–RE

Pengujian pipeline dilakukan menggunakan entitas hasil prediksi NER.

Langkahnya adalah:

1. jalankan NER pada data *test*;
2. ambil seluruh Chemical dan Disease hasil prediksi;
3. bentuk pasangan Chemical–Disease;
4. tambahkan entity marker;
5. jalankan model RE;
6. ambil pasangan dengan probabilitas di atas threshold;
7. bandingkan hasil dengan anotasi referensi.

Karena NER hanya menghasilkan span dan tipe entitas, sedangkan relasi BC5CDR menggunakan MeSH ID, pemetaan ke MeSH ID dilakukan hanya untuk proses evaluasi:

1. prediksi entitas dicocokkan dengan anotasi gold berdasarkan exact span dan tipe;
2. prediksi yang cocok menggunakan MeSH ID gold untuk proses penilaian;
3. hasil prediksi yang tidak cocok dengan entitas gold dihitung sebagai kesalahan pipeline;
4. MeSH ID gold tidak diberikan kepada model pada saat inferensi.

Hasil pipeline dilaporkan secara terpisah dari hasil RE dengan *gold entities*.

| Pengujian               | Precision |  Recall | F1-Score |
| ----------------------- | --------: | ------: | -------: |
| RE dengan gold entities |   0.6694 |  0.6914 |   0.6802 |
| Pipeline NER–RE         |   0.6875 |  0.6088 |   0.6458 |

Perbedaan kedua hasil menunjukkan dampak kesalahan NER terhadap ekstraksi relasi.

Hasil pipeline saat ini:

```text
candidate_pairs = 4401
true_positive = 649
false_positive = 295
false_negative = 417
threshold = 0.70
```

---

## Tahap 12 — Membandingkan Baseline dan Metode Usulan

Perbandingan dilakukan menggunakan kandidat, subset, dan metrik yang sama.

| Metode        | Precision |  Recall | F1-Score | False Positive |
| ------------- | --------: | ------: | -------: | -------------: |
| Co-occurrence |    0.1972 |  1.0000 |   0.3295 |           4339 |
| PubMedBERT RE |    0.6694 |  0.6914 |   0.6802 |            364 |

Hasil perbandingan:

```text
Delta F1 = 0.3507
Penurunan FP = 91.61%
```

Persentase perubahan F1-Score dihitung menggunakan:

[
\Delta F1 =
F1_{\text{usulan}}-F1_{\text{baseline}}
]

Penurunan False Positive dihitung menggunakan:

[
Penurunan\ FP =
\frac{FP_{\text{baseline}}-FP_{\text{usulan}}}
{FP_{\text{baseline}}}
\times 100%
]

Metode usulan dinyatakan memberikan peningkatan apabila:

1. F1-Score lebih tinggi daripada baseline;
2. jumlah False Positive lebih rendah;
3. peningkatan tidak hanya terjadi pada satu *random seed*;
4. hasil dapat direproduksi menggunakan konfigurasi yang sama.

---

## Tahap 13 — Melakukan Analisis Kesalahan

Kesalahan model NER dikelompokkan menjadi:

1. entitas tidak terdeteksi;
2. batas entitas terlalu pendek;
3. batas entitas terlalu panjang;
4. Chemical diprediksi sebagai Disease;
5. Disease diprediksi sebagai Chemical;
6. kesalahan akibat singkatan atau istilah tidak umum.

Kesalahan model RE dikelompokkan menjadi:

1. pasangan co-occurrence tanpa hubungan CID;
2. negasi;
3. hubungan terapi yang dianggap sebagai CID;
4. hubungan terdapat pada kalimat berbeda;
5. terdapat banyak Chemical atau Disease dalam satu abstrak;
6. konteks hubungan terlalu panjang;
7. model memilih pasangan entitas yang salah.

Analisis kesalahan otomatis disimpan pada:

```text
results/error_analysis/error_analysis.json
results/error_analysis/error_analysis.md
```

Ringkasan kesalahan:

| Evaluasi | Kandidat | False Positive | False Negative | FN kandidat hilang | FN ditolak RE |
| -------- | -------: | -------------: | -------------: | -----------------: | ------------: |
| RE dengan gold entities | 5405 | 364 | 329 | 0 | 329 |
| Pipeline NER–RE | 4401 | 295 | 417 | 117 | 300 |

Pada pipeline NER–RE, sebanyak 117 relasi CID gold tidak terbentuk sebagai kandidat karena entitas hasil NER tidak cocok dengan anotasi gold. Rinciannya adalah:

```text
disease_missing = 69
chemical_missing = 30
chemical_and_disease_missing = 18
```

Sebanyak 10 False Positive dan 10 False Negative diperiksa sebagai contoh awal dan dibahas pada Bab IV.

---

## Tahap 14 — Membangun dan Menguji Knowledge Graph

Hanya relasi yang diprediksi sebagai CID oleh metode usulan yang dimasukkan ke Neo4j.

### Konfigurasi Neo4j dengan Docker

Neo4j dijalankan menggunakan `docker-compose.neo4j.yml` pada root repository.
Neo4j Browser tersedia pada port `7474`, sedangkan koneksi Bolt tersedia pada
port `7687`. Direktori `data/graph/` dipasang ke direktori import container.

```powershell
.venv/Scripts/python.exe scripts/build_graph.py
docker compose -f docker-compose.neo4j.yml up -d
```

Kredensial default:

```text
URI      = bolt://localhost:7687
Username = neo4j
Password = enexre12345
```

Jalankan `data/graph/neo4j_import.cypher` di Neo4j Browser untuk mengimpor CSV,
lalu jalankan `data/graph/neo4j_validation_queries.cypher` untuk validasi.
Password dapat diganti melalui file `.env` dengan variabel `NEO4J_PASSWORD`.

Node menggunakan:

```text
Chemical
Disease
```

Relationship menggunakan:

```text
CID
```

Pada data BC5CDR, node dapat menggunakan MeSH ID sebagai kunci unik:

```text
(:Chemical {mesh_id: "D008012"})
(:Disease {mesh_id: "D006323"})
```

Pengujian struktur graf meliputi:

1. setiap MeSH ID hanya membentuk satu node;
2. tidak terdapat relationship ganda untuk pasangan dan PMID yang sama;
3. setiap relationship mempunyai PMID;
4. setiap relationship mempunyai nilai confidence;
5. jumlah relationship sama dengan jumlah relasi CID yang dikirim ke Neo4j;
6. node Chemical hanya terhubung ke node Disease;
7. query pencarian dapat dijalankan.

Contoh query pemeriksaan duplikasi:

```cypher
MATCH (n)
WITH n.mesh_id AS mesh_id, count(n) AS total
WHERE mesh_id IS NOT NULL AND total > 1
RETURN mesh_id, total
```

Hasil yang diharapkan:

```text
0 baris
```

Contoh pemeriksaan relationship tanpa PMID:

```cypher
MATCH ()-[r:CID]->()
WHERE r.pmid IS NULL
RETURN count(r) AS total
```

Hasil yang diharapkan:

```text
0
```

Artefak graph dibuat dengan:

```bash
.venv/Scripts/python.exe scripts/build_graph.py
```

Output yang dihasilkan:

```text
data/graph/chemical_nodes.csv
data/graph/disease_nodes.csv
data/graph/cid_edges.csv
data/graph/neo4j_import.cypher
data/graph/neo4j_validation_queries.cypher
results/graph/graph_validation.json
```

Hasil validasi struktur graf lokal:

| Pemeriksaan | Hasil |
| ----------- | ----: |
| Chemical nodes | 291 |
| Disease nodes | 319 |
| CID relationships | 944 |
| Duplicate Chemical nodes | 0 |
| Duplicate Disease nodes | 0 |
| Duplicate relationships | 0 |
| Relationships tanpa PMID | 0 |
| Relationships tanpa confidence | 0 |
| Relationships dengan endpoint tidak valid | 0 |

Status validasi:

```text
passed = true
```

---

## Tahap 15 — Menguji Prototipe Sistem

Pengujian prototipe dilakukan menggunakan metode *black-box testing*.

| Skenario              | Masukan                          | Hasil yang diharapkan              |
| --------------------- | -------------------------------- | ---------------------------------- |
| PMID valid            | PMID memiliki abstrak            | Abstrak berhasil diambil           |
| PMID tidak valid      | PMID tidak tersedia              | Pesan kesalahan ditampilkan        |
| Artikel tanpa abstrak | PMID tanpa abstrak               | Sistem tidak menjalankan model     |
| Tidak ada Chemical    | Abstrak tanpa Chemical           | Tidak ada pasangan RE              |
| Tidak ada Disease     | Abstrak tanpa Disease            | Tidak ada pasangan RE              |
| Banyak entitas        | Beberapa Chemical dan Disease    | Seluruh kandidat terbentuk         |
| Relasi CID            | Pasangan dengan skor ≥ threshold | Relationship disimpan              |
| Relasi non-CID        | Pasangan dengan skor < threshold | Relationship tidak disimpan        |
| Data yang sama        | PMID dimasukkan dua kali         | Node dan edge tidak terduplikasi   |
| Neo4j tidak aktif     | Koneksi database gagal           | Sistem menampilkan pesan kegagalan |

Hasil dicatat sebagai:

| Skenario   | Berhasil | Gagal | Keterangan   |
| ---------- | -------: | ----: | ------------ |
| [SKENARIO] |    [✓/–] | [✓/–] | [KETERANGAN] |

---

Prototipe awal menggunakan query graph dari Neo4j:

```bash
.venv/Scripts/python.exe scripts/query_graph.py --pmid 18801087 --limit 5
```

Interface sistem berbasis web lokal dijalankan dengan:

```bash
.venv/Scripts/python.exe scripts/prototype_app.py --host 127.0.0.1 --port 8000
```

Alamat interface:

```text
http://127.0.0.1:8000
```

Hasil validasi runtime Neo4j:

```text
chemical_nodes = 291
disease_nodes = 319
cid_relationships = 944
invalid_relationships_missing_pmid_or_confidence = 0
```

Hasil uji prototipe awal:

| Skenario | Berhasil | Gagal | Keterangan |
| -------- | -------: | ----: | ---------- |
| Query relasi berdasarkan PMID | ✓ | – | PMID 18801087 mengembalikan 3 relasi CID |
| Query graph tampil di Neo4j Browser | ✓ | – | Graph Chemical-Disease dapat divisualisasikan |
| Interface web menampilkan hasil pencarian | ✓ | – | Endpoint health dan pencarian PMID berhasil |
| Validasi jumlah node dan relationship | ✓ | – | Jumlah sesuai artefak graph |
| Validasi relationship tanpa PMID/confidence | ✓ | – | Hasil 0 |

---

## Tahap 16 — Menguji Abstrak PubMed di Luar BC5CDR

Setelah model dan threshold dibekukan, sistem diuji pada 5 abstrak PubMed yang tidak termasuk dalam BC5CDR.

Abstrak diambil berdasarkan:

```text
("drug-induced"[Title/Abstract] OR "adverse effect"[Title/Abstract])
AND (disease[Title/Abstract] OR toxicity[Title/Abstract])
AND 2020:2026[pdat]
```

Kriteria data:

1. berbahasa Inggris;
2. memiliki abstrak;
3. diterbitkan pada rentang `[TAHUN]`;
4. tidak terdapat dalam PMID BC5CDR;
5. sesuai dengan topik Chemical–Disease.

Karena data tersebut tidak memiliki anotasi gold, evaluasi dilakukan secara manual terhadap sampel hasil.

Setiap relasi diperiksa berdasarkan pertanyaan:

1. apakah Chemical terdeteksi dengan benar;
2. apakah Disease terdeteksi dengan benar;
3. apakah abstrak menyatakan hubungan CID;
4. apakah hubungan yang tersimpan sesuai dengan teks;
5. apakah PMID dan evidence dapat ditelusuri.

Valid Relation Rate dihitung menggunakan:

[
Valid\ Relation\ Rate =
\frac{\text{relasi yang dinilai benar}}
{\text{seluruh relasi yang diperiksa}}
\times 100%
]

Jumlah reviewer, latar belakang reviewer, aturan penilaian, dan jumlah sampel harus dicatat. Apabila penilaian hanya dilakukan oleh peneliti, kondisi tersebut dinyatakan sebagai keterbatasan penelitian.

Hasil awal Tahap 16:

```text
PubMed abstracts = 5
excluded_bc5cdr_pmids = 1500
predicted_entities = 77
chemical_mentions = 9
disease_mentions = 68
candidate_pairs = 96
predicted_CID_relations_at_threshold_0.70 = 0
```

Artefak hasil disimpan pada:

```text
data/external_pubmed/articles.jsonl
data/external_pubmed/predicted_entities.jsonl
data/external_pubmed/candidate_pairs.jsonl
data/external_pubmed/scored_candidate_pairs.jsonl
data/external_pubmed/predicted_relations.jsonl
results/external_pubmed/external_pubmed_summary.json
results/external_pubmed/manual_review_summary.md
```

Pada sampel eksternal awal, model tidak menghasilkan relasi CID yang melewati threshold final 0,70. Skor tertinggi masih jauh di bawah threshold. Hasil ini dicatat sebagai temuan generalisasi eksternal awal dan perlu dibahas sebagai keterbatasan, karena data eksternal tidak memiliki anotasi CID gold serta tidak menyediakan MeSH ID seperti BC5CDR.

---

## Tahap 17 — Menyimpan Bukti Reproduksibilitas

Berkas berikut harus disimpan sebagai lampiran atau repositori penelitian:

```text
data_manifest.json
config_ner.yaml
config_re.yaml
train_ner.py
evaluate_ner.py
build_re_dataset.py
train_re.py
evaluate_re.py
build_graph.py
requirements.txt
README.md
```

Selain itu, simpan:

1. versi dataset;
2. checksum dataset;
3. nama checkpoint PubMedBERT;
4. versi Python;
5. versi PyTorch;
6. versi Transformers;
7. versi Neo4j;
8. spesifikasi CPU, GPU, dan RAM;
9. random seed;
10. parameter pelatihan;
11. checkpoint terbaik;
12. hasil prediksi test;
13. confusion matrix;
14. log pelatihan;
15. hasil query pengujian Neo4j.

Dengan berkas tersebut, pengujian dapat dijalankan kembali oleh peneliti lain menggunakan data dan konfigurasi yang sama.

Bukti reproduksibilitas aktual dibuat dengan:

```bash
.venv/Scripts/python.exe -m pip freeze > requirements.lock.txt
.venv/Scripts/python.exe scripts/collect_reproducibility.py
```

Output:

```text
requirements.lock.txt
results/reproducibility/reproducibility_report.json
results/reproducibility/reproducibility_report.md
```

Ringkasan lingkungan:

```text
Python = 3.10.0
PyTorch = 2.12.1
Transformers = 5.12.1
Neo4j Python driver = 6.2.0
Git branch = re-best
Git commit = c1695af79910d26eae34bb04983b068b41506ac0
```

Ringkasan hasil yang dicatat:

```text
NER test F1 mean = 0.8897
RE gold test F1 = 0.6802
Pipeline NER-RE test F1 = 0.6458
Graph = 291 Chemical nodes, 319 Disease nodes, 944 CID relationships
External PubMed = 5 abstracts, 96 candidates, 0 predicted CID relations
```

---

# Urutan Eksekusi Ringkas

Urutan pengerjaan penelitian secara nyata adalah:

```text
1. Unduh dan validasi BC5CDR
2. Bentuk data NER
3. Latih NER pada training set
4. Pilih model NER pada development set
5. Uji NER pada test set
6. Bentuk pasangan konsep untuk RE
7. Uji baseline co-occurrence
8. Bentuk data RE dengan entity marker
9. Latih RE pada training set
10. Pilih model dan threshold pada development set
11. Uji RE dengan gold entities pada test set
12. Uji pipeline NER–RE pada test set
13. Bandingkan baseline dan metode usulan
14. Lakukan analisis kesalahan
15. Masukkan relasi CID ke Neo4j
16. Uji struktur knowledge graph
17. Uji fungsi prototipe
18. Uji pada abstrak PubMed di luar BC5CDR
19. Simpan konfigurasi, log, prediksi, dan source code
20. Sajikan hasil pada Bab IV
```
