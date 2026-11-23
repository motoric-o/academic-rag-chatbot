# 📋 Panduan Penilaian Manusia (Human Rating Guide)

> **Owner:** Member D (Lead Evaluasi & Frontend; merujuk `docs/project_spec.md` §0.2, §0.5, §11.1).  
> **Target:** Digunakan oleh seluruh 4 anggota tim (A, B, C, D) saat mengevaluasi sampel jawaban model AI.

---

## 1. Tujuan Evaluasi Manusia
Meskipun sistem memiliki evaluasi otomatis (*strict key-fact match* dan *NLI faithfulness*), penilaian manusia (*human rating*) adalah **standar emas (gold standard)** untuk:
1. Memvalidasi apakah jawaban model benar-benar tepat dan dapat dipahami mahasiswa.
2. Mendeteksi kontradiksi halus (misal: syarat 60 SKS tapi AI menjawab 70 SKS).
3. Mengukur akurasi dari *Secondary LLM Judge* dan menghitung kesepakatan antar penilai (*Cohen's Kappa*).

---

## 2. Skala Penilaian (0 / 1 / 2)

Setiap jawaban model dinilai berdasarkan perbandingannya dengan **pertanyaan** dan teks acuan **gold answer** (buku panduan akademik):

| Skor | Kategori | Kriteria Penilaian | Contoh Kasus |
|:---:|---|---|---|
| **2** | **Benar & Lengkap** *(Fully Correct)* | • Semua fakta kunci dari peraturan akademik terjawab dengan tepat.<br>• Angka, durasi, syarat, atau persentase 100% akurat.<br>• Tidak ada kontradiksi dengan buku panduan. | **Tanya:** *Berapa semester maksimal cuti akademik?*<br>**Gold:** *Maksimal 2 semester berturut-turut.*<br>**Model:** *"Mahasiswa dapat mengambil cuti akademik maksimal 2 semester berturut-turut. Sources: [1]"* → **Skor 2** |
| **1** | **Sebagian Benar** *(Partially Correct)* | • Menjawab sebagian fakta, tetapi ada informasi penting yang terlewat.<br>• Jawaban benar tapi kurang spesifik/kurang lengkap.<br>• Tidak ada informasi yang fatal salah atau membahayakan. | **Tanya:** *Berapa semester maksimal cuti akademik dan syaratnya?*<br>**Gold:** *Maksimal 2 semester berturut-turut melalui persetujuan dosen wali.*<br>**Model:** *"Maksimal 2 semester."* (lupa menyebutkan persetujuan dosen wali) → **Skor 1** |
| **0** | **Salah / Halusinasi** *(Incorrect / Hallucinated)* | • Jawaban salah atau bertentangan dengan peraturan.<br>• Angka atau syarat salah (halusinasi fatal).<br>• Model mengarang jawaban pada soal yang tidak ada di panduan.<br>• Model menjawab `NOT_FOUND` padahal jawabannya ada di konteks. | **Tanya:** *Berapa SKS minimal untuk magang?*<br>**Gold:** *Minimal 84 SKS.*<br>**Model:** *"Minimal 70 SKS."* (salah angka) → **Skor 0** |

### Aturan Khusus untuk Soal Tanpa Jawaban (*Unanswerable*):
- Jika pertanyaan memang **tidak ada jawabannya di buku panduan** (misal: biaya kuliah di dokumen akademik):
  - Model menjawab **`NOT_FOUND`** (atau abstention yang valid) → **Skor 2** (Sangat tepat menolak menjawab).
  - Model **mengarang jawaban** (*false answer / halusinasi*) → **Skor 0** (Kesalahan kritis/fatal).

---

## 3. Format Berkas Penilaian (CSV)

Berkas CSV penilaian disimpan di folder `results/` dengan kolom-kolom berikut:

| Kolom | Tipe | Penjelasan |
|---|---|---|
| `qid` | string | ID pertanyaan (contoh: `fixture_01`, `q017`) |
| `model` | string | Model AI yang diuji (contoh: `qwen3-4b`, `openai/gpt-oss-20b`) |
| `precision` | string | Presisi kuantisasi (contoh: `4-bit`, `8-bit`, `fp16`) |
| `answer` | string | Teks jawaban yang dihasilkan oleh model AI |
| `rating` | integer | **Nilai Anda: `0`, `1`, atau `2`** |
| `rater` | string | Inisial Anda: `A`, `B`, `C`, atau `D` |
| `notes` | string | Catatan tambahan alasan penilaian (opsional) |

---

## 4. Cara Menilai (Pilihan Alat)

Anda dapat mengisi penilaian menggunakan salah satu dari dua cara berikut:

### Opsi A: Menggunakan Spreadsheet (Excel / Google Sheets)
1. Buka berkas template di `results/human_rating_template.csv` menggunakan Microsoft Excel atau Google Sheets.
2. Isi kolom **`rating`** dengan angka `0`, `1`, atau `2`.
3. Isi kolom **`rater`** dengan inisial Anda (misal `D`).
4. Simpan kembali sebagai CSV dengan nama `results/ratings_<inisial>.csv` (contoh: `ratings_D.csv`).

### Opsi B: Menggunakan Rating Form Interaktif (Web Lokal)
1. Buka berkas `results/rating_form.html` di browser (cukup klik dua kali berkas tersebut).
2. Tampilan form interaktif akan memuat sampel jawaban satu per satu.
3. Anda cukup menekan tombol/angka keyboard `[0]`, `[1]`, atau `[2]` untuk menilai dengan cepat.
4. Klik tombol **Export CSV** untuk mengunduh hasil penilaian Anda.

---

## 5. Prosedur Uji Konsistensi (Cohen's Kappa)

Sesuai spesifikasi proyek (`project_spec.md` §0.5):
- Setiap anggota tim menilai bagian sampelnya masing-masing (total ≥60 jawaban per sistem utama).
- **20% dari total pertanyaan dinilai ganda** oleh dua orang berbeda untuk mengukur kesepakatan antar-penilai (*inter-rater agreement*).
- Untuk menghitung nilai Cohen's Kappa, jalankan skrip evaluasi yang disediakan Member D:

```powershell
python -m scripts.evaluate agreement results/ratings_A.csv results/ratings_B.csv
```

**Interpretasi Nilai Kappa ($\kappa$):**
- $\kappa \ge 0.80$: Kesepakatan sangat tinggi (*Almost perfect agreement*)
- $0.60 \le \kappa < 0.80$: Kesepakatan baik (*Substantial agreement*)
- $0.40 \le \kappa < 0.60$: Kesepakatan moderat (*Moderate agreement*)
- $\kappa < 0.40$: Perlu diskusikan ulang pemahaman rubrik antar penilai.
