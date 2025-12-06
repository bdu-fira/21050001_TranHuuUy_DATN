import json
import time
from deep_translator import GoogleTranslator
from tqdm import tqdm

# Cấu hình tên file
INPUT_FILE = 'final_neutral_reviews.json'
OUTPUT_FILE = 'augmented_neutral_reviews.json'

def back_translate(text, sleep_time=0.2):
    """
    Hàm thực hiện dịch Việt -> Anh -> Việt.
    sleep_time: thời gian nghỉ giữa các lần gọi để tránh bị chặn IP.
    """
    try:
        # Khởi tạo translator
        vi_to_en = GoogleTranslator(source='vi', target='en')
        en_to_vi = GoogleTranslator(source='en', target='vi')

        # Dịch sang tiếng Anh
        english_text = vi_to_en.translate(text)
        time.sleep(sleep_time)

        # Dịch ngược lại tiếng Việt
        vietnamese_text = en_to_vi.translate(english_text)
        time.sleep(sleep_time)
        
        return vietnamese_text
    except Exception as e:
        print(f"\nLỗi khi dịch câu: {text[:30]}... - Lỗi: {e}")
        # Nếu lỗi thì trả về câu gốc để không mất dữ liệu
        return text

def main():
    # 1. Đọc dữ liệu từ file JSON
    print(f"Đang đọc dữ liệu từ {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"Không tìm thấy file {INPUT_FILE}. Hãy kiểm tra lại tên file.")
        return

    print(f"Tổng số bình luận cần xử lý: {len(original_data)}")

    augmented_data = []

    # 2. Thực hiện vòng lặp dịch thuật
    for comment in tqdm(original_data, desc="Đang tăng cường dữ liệu"):
        # Bỏ qua nếu comment quá ngắn hoặc rỗng
        if not comment or len(comment.strip()) < 2:
            continue
            
        new_comment = back_translate(comment)
        
        # Chỉ thêm vào nếu kết quả khác rỗng
        if new_comment:
            augmented_data.append(new_comment)

    # 3. Lưu lại thành file JSON mới
    print(f"Đang lưu kết quả vào {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # ensure_ascii=False để giữ nguyên tiếng Việt không bị lỗi font
        json.dump(augmented_data, f, ensure_ascii=False, indent=4)

    print("Hoàn tất! Kiểm tra file output.")

if __name__ == "__main__":
    main()