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
| True Positive  | [HASIL] |
| False Positive | [HASIL] |
| False Negative | [HASIL] |
| Precision      | [HASIL] |
| Recall         | [HASIL] |
| F1-Score       | [HASIL] |

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
| True Positive  | [HASIL] |
| False Positive | [HASIL] |
| False Negative | [HASIL] |
| Precision      | [HASIL] |
| Recall         | [HASIL] |
| F1-Score       | [HASIL] |
| Threshold      | [HASIL] |

Pengujian ini menjadi hasil utama model RE karena pasangan entitas yang digunakan berasal dari anotasi referensi.

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
| RE dengan gold entities |   [HASIL] | [HASIL] |  [HASIL] |
| Pipeline NER–RE         |   [HASIL] | [HASIL] |  [HASIL] |

Perbedaan kedua hasil menunjukkan dampak kesalahan NER terhadap ekstraksi relasi.

---

## Tahap 12 — Membandingkan Baseline dan Metode Usulan

Perbandingan dilakukan menggunakan kandidat, subset, dan metrik yang sama.

| Metode        | Precision |  Recall | F1-Score | False Positive |
| ------------- | --------: | ------: | -------: | -------------: |
| Co-occurrence |   [HASIL] | [HASIL] |  [HASIL] |        [HASIL] |
| PubMedBERT RE |   [HASIL] | [HASIL] |  [HASIL] |        [HASIL] |

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

Sebanyak `[JUMLAH SAMPEL]` False Positive dan `[JUMLAH SAMPEL]` False Negative diperiksa dan dibahas pada Bab IV.

---

## Tahap 14 — Membangun dan Menguji Knowledge Graph

Hanya relasi yang diprediksi sebagai CID oleh metode usulan yang dimasukkan ke Neo4j.

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

## Tahap 16 — Menguji Abstrak PubMed di Luar BC5CDR

Setelah model dan threshold dibekukan, sistem diuji pada `[JUMLAH]` abstrak PubMed yang tidak termasuk dalam BC5CDR.

Abstrak diambil berdasarkan:

```text
[QUERY PUBMED]
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
