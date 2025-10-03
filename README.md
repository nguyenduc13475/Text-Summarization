# Text Summarization Project (CO3101)

**Nhóm thực hiện:**
- Nguyễn Văn Đức - 2310790

Dự án này là đồ án tổng hợp của học phần **Hướng trí tuệ nhân tạo (CO3101)**, với đề tài **Text Summarization**.  
Nhóm hiện thực và so sánh nhiều mô hình hiện đại trong tóm tắt văn bản, đồng thời tích hợp thêm một số phương pháp truyền thống để đối chiếu kết quả.

---

## 📌 Các mô hình chính

- **Pointer Generator Network with Coverage**
- **Neural Intra Attention Model**  
  (sử dụng mục tiêu hỗn hợp: Reinforcement Learning loss + Negative Log-Likelihood loss)
- **Transformer**

Ngoài ra, nhóm cũng cài đặt **TextRank** để tham khảo.

---

## 📂 Tokenizer sử dụng

- **Pointer Generator Network** và **Neural Intra Attention Model**:  
  Sử dụng **Word-level tokenizer** với từ vựng trong file `word_level_vocab.json` (25,000 token).
- **Transformer**:  
  Sử dụng **ByteLevelBPETokenizer** với file `merges.txt` và `vocab.json` (50,000 token).

---

## 📊 Dataset

Nhóm sử dụng **CNN/Daily Mail Dataset** gồm:
- **Train**: 287,113 samples  
- **Validation**: 13,368 samples  
- **Test**: 11,490 samples  

Mỗi sample có cấu trúc:
```json
{
  "id": "string",
  "article": "string văn bản gốc",
  "highlights": "string tóm tắt"
}
```

---

## 💾 Checkpoints

- Tất cả checkpoint được lưu trong thư mục:
```
<Tên_Model>_checkpoints/
```

---

## 🚀 Hướng dẫn chạy

### 1. Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình
```bash
python training.py
```
> Các tham số huấn luyện có thể chỉnh trực tiếp trong file `training.py`.

### 3. Đánh giá mô hình
```bash
python evaluation.py
```
> Các tham số đánh giá có thể chỉnh trong file `evaluation.py`.

### 4. Suy luận (Inference) với Beam Search
```bash
python inference.py
```
> Các tham số có thể chỉnh trong file `inference.py`.

### 5. Cross Validation cho Hyperparameter Tuning
```bash
python cross_validation.py
```
> Các tham số có thể chỉnh trong file `cross_validation.py`.

### 6. Chạy TextRank
```bash
python text_rank.py
```

---

## ⚙️ Nền tảng chạy

- Có thể chạy trên **local machine** hoặc **Google Colab**.  
- Hỗ trợ cả **CPU** và **GPU** (khuyến nghị GPU để tăng tốc huấn luyện).

---

## 📈 Metrics đánh giá

Các chỉ số được sử dụng để đánh giá mô hình:
- **ROUGE-1**
- **ROUGE-2**
- **ROUGE-L**
- **BLEU-4**
- **METEOR**
- **BERTScore**
- **MoverScore**

---
