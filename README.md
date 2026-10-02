**QUY TRÌNH CHẠY CODE TRÊN CONDA**

**CLONE REPO**

```bash
git clone https://github.com/Flareons/titans_pytorch.git
cd Embedding
```

**TẠO MÔI TRƯỜNG CONDA**

```bash
conda env create -f enc.yml
```

**ACTIVE CONDA**

```bash
conda activate vulnerability-ai
```
**GỠ CÀI ĐẶT PHIÊN BẢN TORCH HIỆN TẠI**

```bash
pip uninstall torch
```
**CÀI ĐẶT PYTORCH DO PHIÊN BẢN MỚI CHỈ CÓ THỂ CÀI QUA URL**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**TẠO DỮ LIỆU EMBEDDING**

```bash
python Embedding.py
```

**CHẠY MODEL**

```bash
python test_git_model.py
```
