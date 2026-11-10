# Truy vấn ảnh dựa trên hình phác thảo (Sketch-Based Image Retrieval - SBIR)

**Đồ án môn học:** CS406 – Xử lý ảnh và Ứng dụng  
**Nhánh thực hiện:** Nhánh Truyền thống & Hạ tầng (Thành viên A) & Nhánh Học sâu & Phân tích (Thành viên B)

---

## 1. Giới thiệu tổng quan
Dự án triển khai và đối sánh hai hướng tiếp cận giải quyết bài toán Category-level và Zero-Shot SBIR:
1. **Baseline Truyền thống (E1):** Tiền xử lý (Bilateral Filter, Auto-Canny, Zhang-Suen, Dilation) + Dense HOG + Bag-of-Visual-Words (MiniBatchKMeans $K=1000$) + Spatial Pyramid (5000-D).
2. **Mốc trung gian (E2):** OpenAI CLIP ViT-B/32 đóng băng hoàn toàn.
3. **Mô hình hiện đại (E3, E4):** CLIP ViT-B/32 + Deep Visual Prompt Learning (Sain et al., CVPR 2023) kết hợp Triplet Loss và Semantic Classification Loss qua CLIP Text Encoder.
4. **Kiểm thử thực tế (E6):** Đánh giá trên sketch tự vẽ (canvas tương tác và vẽ giấy chụp điện thoại qua hình thái học triệt tiêu ánh sáng).

---

## 2. Cấu trúc thư mục dự án
```text
.
├── configs/                 # File cấu hình YAML cho các thí nghiệm
│   ├── baseline_e1.yaml     # Cấu hình baseline truyền thống
│   └── deep_e2_e4.yaml      # Cấu hình CLIP đóng băng và Deep Prompt Learning
├── data/                    # Dữ liệu và phân chia split (git-ignored cho dữ liệu thô)
│   ├── raw/                 # Thư mục chứa dataset thô (Flickr15k, Sketchy)
│   └── splits/              # File CSV phân chia train/val/test zero-shot
├── docs/                    # Tài liệu yêu cầu môn học, đề xuất đồ án, kế hoạch 4 tuần
├── sbir/                    # Package mã nguồn chính
│   ├── traditional/         # Pipeline xử lý ảnh truyền thống
│   │   ├── preprocess.py    # Lọc song phương, Auto-Canny, làm mảnh Zhang-Suen, giãn nét
│   │   ├── features.py      # Trích xuất Dense HOG đa luồng
│   │   └── bovw.py          # Codebook MiniBatchKMeans, TF-IDF, Spatial Pyramid
│   ├── deep/                # Pipeline học sâu và Foundation Model
│   │   ├── frozen.py        # Mốc trung gian CLIP ViT-B/32 đóng băng
│   │   ├── prompt_clip.py   # Kiến trúc chèn Deep Visual Prompts vào ViT
│   │   ├── losses.py        # Triplet loss và Semantic classification loss
│   │   └── train.py         # Huấn luyện và validation
│   ├── datasets.py          # Dataset loaders cho Flickr15k, Sketchy
│   ├── evaluate.py          # Module đánh giá chuẩn (mAP, mAP@200, P@K)
│   └── index.py             # Dựng chỉ mục FAISS IndexFlatIP truy vấn nhanh
├── scripts/                 # Scripts chạy thực nghiệm E1 - E6
│   └── verify_splits.py     # Script tự động kiểm tra data leakage
├── app/                     # Ứng dụng demo tương tác
│   └── demo.py              # Giao diện Gradio tương tác nét vẽ và tải ảnh
├── notebooks/               # Jupyter / Kaggle notebooks chạy thí nghiệm
├── requirements.txt         # Danh sách thư viện phụ thuộc
├── .gitignore               # Cấu hình loại trừ file rác, dữ liệu nặng và checkpoint
└── README.md
```

---

## 3. Cài đặt môi trường (Installation)
Yêu cầu Python 3.10 trở lên.
```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Trên Linux/macOS
# venv\Scripts\activate   # Trên Windows

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 4. Chuẩn bị dữ liệu (Dataset Preparation)
1. Tải bộ dữ liệu **Sketchy Extended** hoặc **Flickr15k** và đặt vào `data/raw/`.
2. Tạo các phân chia zero-shot và kiểm tra chống rò rỉ dữ liệu:
```bash
python scripts/verify_splits.py --split_file data/splits/sketchy_zero_shot_splits.csv
```

---

## 5. Huấn luyện & Thực nghiệm (Training & Evaluation)
* **Chạy Baseline truyền thống (E1):**
```bash
python scripts/run_e1_baseline.py --config configs/baseline_e1.yaml
```
* **Chạy mốc CLIP đóng băng (E2):**
```bash
python scripts/run_e2_frozen.py --config configs/deep_e2_e4.yaml
```
* **Huấn luyện CLIP + Deep Prompt Learning (E3, E4):**
```bash
python scripts/run_e3_e4_prompt.py --config configs/deep_e2_e4.yaml
```

---

## 6. Chạy Ứng dụng Demo (Demo Application)
Khởi chạy giao diện vẽ sketch tương tác Gradio:
```bash
python app/demo.py
```
Ứng dụng cho phép vẽ sketch trực tiếp hoặc tải ảnh chụp, chọn giữa 3 phương pháp (Baseline, Frozen CLIP, Deep Prompt Learning) và hiển thị top-$K$ kết quả cùng thời gian truy vấn.
