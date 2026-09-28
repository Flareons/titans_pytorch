# Code Vulnerability Detection

Project phát hiện lỗ hổng bảo mật trong mã nguồn sử dụng **CodeBERT Embedding** kết hợp với mô hình **TITANS / Neural Memory**.

## 1. Yêu cầu hệ thống

* Python >= 3.10
* Khuyến nghị sử dụng GPU NVIDIA để tăng tốc
* CUDA tương thích với phiên bản PyTorch trong `requirements.txt`

## 2. Clone repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <PROJECT_FOLDER>
```

## 3. Tạo Virtual Environment

### Windows

```bash
python -m venv .venv
```

Kích hoạt môi trường:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Sau khi kích hoạt thành công, terminal sẽ hiển thị:

```text
(.venv)
```

## 4. Cài đặt thư viện

Cập nhật `pip`:

```bash
python -m pip install --upgrade pip
```

Cài đặt toàn bộ dependencies:

```bash
pip install -r requirements.txt
```

Nếu project sử dụng GPU NVIDIA, hãy đảm bảo PyTorch và CUDA đã được cài đặt đúng theo `requirements.txt`.

Có thể kiểm tra GPU bằng:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Nếu kết quả tương tự:

```text
True
NVIDIA ...
```

thì PyTorch đã nhận GPU.

---

# 5. Tạo dữ liệu Embedding

Sau khi cài đặt đầy đủ requirements, chạy:

```bash
python Embedding.py
```

Script này thực hiện quá trình tạo **CodeBERT embeddings** từ dữ liệu đầu vào và lưu kết quả để sử dụng cho bước huấn luyện / đánh giá model.

Quá trình này có thể mất một khoảng thời gian tùy thuộc vào:

* Số lượng dữ liệu
* CPU/GPU
* Dung lượng RAM
* Độ dài mã nguồn

Sau khi chạy xong, kiểm tra các file dữ liệu/output được tạo ra trước khi chuyển sang bước tiếp theo.

---

# 6. Chạy model

Sau khi đã tạo xong dữ liệu embedding, chạy:

```bash
python test_git_model.py
```

Script này sử dụng dữ liệu embedding đã tạo ở bước trước để chạy model **TITANS / Neural Memory** và thực hiện quá trình train/test theo cấu hình trong project.

```text
Embedding.py
      │
      ▼
CodeBERT Embedding
      │
      ▼
Saved Embedding Data
      │
      ▼
test_git_model.py
      │
      ▼
TITANS / Neural Memory
      │
      ▼
Model Results
```

## 7. Thứ tự chạy

Nếu chạy project từ đầu, thứ tự thực hiện là:

```bash
# 1. Tạo môi trường
python -m venv .venv

# 2. Kích hoạt môi trường
.venv\Scripts\activate

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Tạo CodeBERT embeddings
python Embedding.py

# 5. Chạy model
python test_git_model.py
```

## 8. Lưu ý

`test_git_model.py` phụ thuộc vào dữ liệu được tạo bởi `Embedding.py`. Vì vậy, nếu chưa tạo embedding hoặc đường dẫn dữ liệu không đúng, model có thể không chạy được.

Đảm bảo cấu trúc thư mục và các đường dẫn dữ liệu phù hợp với cấu hình trong source code.

Nếu sử dụng GPU có VRAM thấp, có thể cần giảm batch size hoặc điều chỉnh cấu hình model để tránh lỗi `CUDA out of memory`.

## 9. Troubleshooting

### Không tìm thấy module

Nếu gặp lỗi:

```text
ModuleNotFoundError
```

hãy đảm bảo virtual environment đã được kích hoạt và chạy lại:

```bash
pip install -r requirements.txt
```

### PyTorch không nhận GPU

Kiểm tra:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Nếu trả về:

```text
False
```

hãy kiểm tra lại phiên bản PyTorch, CUDA và driver NVIDIA.

### CUDA Out Of Memory

Nếu gặp:

```text
CUDA out of memory
```

hãy giảm batch size trong cấu hình model hoặc sử dụng mixed precision (FP16/BF16) nếu model hỗ trợ.

---

# 10. Project Structure

Cấu trúc project dự kiến:

```text
Project/
│
├── Embedding.py
├── test_git_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── ...
│
├── embeddings/
│   └── ...
│
└── ...
```

> Tên và vị trí các thư mục dữ liệu có thể thay đổi tùy theo cấu hình trong source code.
