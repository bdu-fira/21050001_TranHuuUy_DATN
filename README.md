# ỨNG DỤNG PHÂN TÍCH CẢM XÚC TỪ MẠNG XÃ HỘI DÀNH CHO TRƯỜNG ĐẠI HỌC BÌNH DƯƠNG

**Sinh viên thực hiện:** Trần Hữu Uy  
**Giảng viên hướng dẫn:** ThS. Nguyễn Văn Thành

## Yêu Cầu Hệ Thống (System Requirements)

- **Python 3.11**
- **pip** (Python Package Manager)
- **SQLite3** (thường có sẵn)
- RAM: Tối thiểu 8GB
- GPU (CUDA 10.2+) - Tùy chọn, giúp tăng tốc độ xử lý

## Cài Đặt (Installation)

### 1. Clone Repository
```bash
git clone https://github.com/bdu-fira/21050001_TranHuuUy_DATN.git
cd 21050001_TranHuuUy_DATN
```

### 2. Tạo Virtual Environment (Khuyến Khích)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3.  Cài Đặt Dependencies
```bash
pip install -r requirements.txt
```
### 4. Tải Word2Vec được đào tạo trước
Tải và giải nén [vi_word2vec.rar](https://drive.google.com/file/d/1RXS8WRIPhHU2q6gTSq387pG5gywfQfdP/view) vào thư mục **Data**
- vi_word2vec.txt: Word2Vec được đào tạo trước với kích thước nhúng là 100 từ [PhoW2V](https://github.com/datquocnguyen/PhoW2V)

## Cấu Hình (Configuration)

### 1.  Tạo File `.env`
Tạo file `.env` trong thư mục gốc của project:
```env
# Flask Config
FLASK_SECRET_KEY=your_secret_key
FLASK_ENV=production
FLASK_DEBUG=False

# Webshare proxy API Keys
WEBSHARE_API_KEY=your_webshare_api

# ID địa điểm trên Google Maps (trường Đại học Bình Dương)
GMAP_PLACE_ID=ChIJncGVJL7RdDERwv9mRNl32a8
```

### 2.  Chuẩn Bị File Cấu Hình
Đảm bảo thư mục `configs/` chứa file:
- `predict_config.yml` - Cấu hình cho mô hình dự đoán

### 3. Chuẩn Bị Dữ Liệu
- File `Data/vietnamese-stopwords.txt`  - Danh sách stopwords tiếng Việt
- File `Data/acronym. txt`              - Danh sách từ viết tắt
- File `Data/improvement_topics.json`   - Các chủ đề cần cải thiện
- File `Data/strengths_topics.json`     - Các chủ đề điểm mạnh

## Chạy Ứng Dụng (Running the Application)

### 1. Chạy Flask Server
```bash
python main.py
```

Ứng dụng sẽ khởi động tại:
```
http://localhost:5000
```

### 2. Truy Cập Web Interface
- Mở trình duyệt web
- Nhập URL: `http://localhost:5000`
- Nhấn Enter để vào trang chủ

## Đăng Nhập (Login)

### Tài Khoản Mặc Định
- **Tên đăng nhập (Username):** `admin`
- **Mật khẩu (Password):** `admin123`

## Các Chức Năng Chính (Main Features)

### 1. **Trang Chủ (Home Page)**
- Route: `/`
- Mô tả: Trang chủ của ứng dụng, cho phép người dùng khách góp ý

### 2. **Dự Đoán Cảm Xúc (Sentiment Prediction)**
- Route: `/predict` (GET/POST)
- Chức năng: Nhập text để dự đoán cảm xúc (Tích cực/Trung tính/Tiêu cực)
- Input: Văn bản cần phân tích
- Output: Loại cảm xúc

### 3. **Thu Thập Đánh Giá (Get Reviews)**
- Route: `/get_reviews` (GET/POST)
- Chức năng: Thu thập và phân tích đánh giá từ:
  - Google Maps
  - TikTok Comments
  - Feedback từ database
- Output: Báo cáo chi tiết với biểu đồ, từ khóa, và gợi ý cải thiện

### 4. **Xem Lịch Sử Phân Tích (Analysis History)**
- Route: `/analysis_history`
- Chức năng: Xem danh sách các báo cáo phân tích đã tạo

### 5. **Xem Chi Tiết Báo Cáo (View Report)**
- Route: `/view_report/<report_id>`
- Chức năng: Xem chi tiết một báo cáo cụ thể

### 6. **Gửi Phản Hồi (Submit Feedback)**
- Route: `/submit_feedback` (POST)
- Chức năng: Người dùng gửi ý kiến phản hồi

### 7. **Xem Phản Hồi (View Feedbacks)**
- Route: `/view_feedbacks`
- Chức năng: Xem danh sách tất cả phản hồi đã gửi

### 8. **Đăng Xuất (Logout)**
- Route: `/logout`
- Chức năng: Đăng xuất khỏi hệ thống

## Cấu Trúc Cơ Sở Dữ Liệu (Database Structure)

Ứng dụng sử dụng SQLite với 3 database:

### 1. `feedback.db`
- Bảng `feedbacks`: Lưu trữ phản hồi từ người dùng

### 2. `analysis_reports.db`
- Bảng `reports`: Lưu trữ báo cáo phân tích chi tiết

### 3. `users.db`
- Bảng `users`: Lưu trữ thông tin tài khoản người dùng

## Quy Trình Hoạt Động (Workflow)

```
1. Đăng Nhập 
        ↓
2. Nhập Văn Bản/Thu Thập Đánh Giá
        ↓                       
3. Tiền Xử Lý Text (Chuẩn Hóa, Loại Bỏ Stopwords, Sửa Lỗi) 
        ↓            
4. Dự Đoán Bằng LSTM Model
        ↓
5. Tạo Biểu Đồ (Wordcloud, N-gram, Venn Diagram)
        ↓
6. Lưu Báo Cáo vào Database 
        ↓
7. Hiển Thị Kết Quả
```

## Lỗi Thường Gặp (Troubleshooting)

### Lỗi: `ModuleNotFoundError: No module named 'torch'`
**Giải pháp:**
```bash
pip install torch torchvision torchaudio
```

### Lỗi: `FileNotFoundError: data/vietnamese-stopwords.txt`
**Giải pháp:**
- Kiểm tra file có tồn tại trong thư mục `data/`
- Nếu không, tạo file hoặc tải từ repository

### Lỗi: `Port 5000 already in use`
**Giải pháp:**
```bash
# Chạy trên port khác
python main.py
# Hoặc thay đổi port trong code
```

### Lỗi: `CUDA out of memory`
**Giải pháp:**
- Giảm `BATCH_SIZE` trong `main.py`
- Hoặc sử dụng CPU thay vì GPU

## Ghi Chú Quan Trọng (Important Notes)

1. **Mô hình AI:** Ứng dụng sử dụng mô hình LSTM được huấn luyện từ dữ liệu thu thập được của khoảng 100 trường Đại học
2. **Hỗ trợ Tiếng Việt:** Sử dụng `underthesea` library để tokenize
3. **Rate Limiting:** Ứng dụng giới hạn 200 request/ngày, 50 request/giờ cho API
4. **Bảo Mật:** Session cookies có flag HTTPOnly để chống XSS attack
5. **CSRF Protection:** Tất cả form POST được bảo vệ bởi CSRF token

## Đường Dẫn Quan Trọng (Important Paths)

| Thư mục | Mục Đích |
|---------|---------|
| `/` | Thư mục gốc chứa main.py |
| `/configs/` | File cấu hình ứng dụng |
| `/data/` | Dữ liệu và stopwords |
| `/templates/` | HTML templates cho Flask |
| `/static/` | CSS, JavaScript, Images |
| `/logs/` | Các file mô hình đã huấn luyện |

## Hỗ Trợ (Support)

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra lại các bước cài đặt
2. Xem log trong thư mục `/logs/`
3. Kiểm tra kết nối internet (nếu cần gọi API Google/TikTok)

---

© 2025 - Đại học Bình Dương - Đồ án tốt nghiệp