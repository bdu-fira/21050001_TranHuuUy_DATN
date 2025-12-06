import json
import os
import re
import emoji
import unicodedata

# --- CẤU HÌNH ---
INPUT_FILE = 'reviews.json'
ACRONYM_FILE = 'acronym.txt'
OUTPUT_FILE = 'cleaned_reviews.json'
# --- --------- ---

# ==============================================================================
# TRÍCH XUẤT VÀ LỌC REVIEW BAN ĐẦU
# ==============================================================================

def process_single_review_list(reviews_list):
    """
    Xử lý một danh sách các review: loại bỏ rating, rtime và các review không hợp lệ.
    """
    filtered_reviews = []
    for review in reviews_list:
        if not isinstance(review, dict):
            continue

        text = review.get('text')
        # Kiểm tra điều kiện text: phải tồn tại, là string và dài hơn 1 ký tự
        if isinstance(text, str) and len(text.strip()) > 1:
            review.pop('rating', None)
            review.pop('rtime', None)
            filtered_reviews.append(review)
    return filtered_reviews

def preprocess_reviews(data):
    """
    Xử lý dữ liệu review thô: xóa 'rating', xóa 'rtime' và các review có 'text' trống.
    Hàm này duyệt qua cấu trúc dữ liệu để tìm danh sách 'reviews'.
    """
    if isinstance(data, list):
        processed_list = []
        for item in data:
            if isinstance(item, dict) and 'reviews' in item and isinstance(item.get('reviews'), list):
                item['reviews'] = process_single_review_list(item['reviews'])
            processed_list.append(item)
        return processed_list
    elif isinstance(data, dict):
        if 'reviews' in data and isinstance(data.get('reviews'), list):
            data['reviews'] = process_single_review_list(data['reviews'])
        return data
    else:
        print("Cảnh báo: Dữ liệu đầu vào không phải là list hoặc dict.")
        return data

def extract_review_texts(data):
    """
    Trích xuất tất cả các giá trị 'text' từ cấu trúc dữ liệu đã được xử lý.
    """
    all_texts = []

    def find_reviews_recursive(current_data):
        if isinstance(current_data, dict):
            reviews_list = current_data.get('reviews')
            if isinstance(reviews_list, list):
                for review in reviews_list:
                    if isinstance(review, dict) and 'text' in review:
                        all_texts.append(review['text'])
        elif isinstance(current_data, list):
            for item in current_data:
                find_reviews_recursive(item)

    find_reviews_recursive(data)
    return all_texts


# ==============================================================================
# LÀM SẠCH VĂN BẢN NÂNG CAO
# ==============================================================================

# Danh sách các ký tự cần loại bỏ hoàn toàn khỏi văn bản
CHARS_TO_REMOVE = ['\\"', '"', '“', '”']

# Định nghĩa các chữ cái tiếng Việt (thường và hoa, có dấu và không dấu)
VIETNAMESE_LETTERS = r'a-zA-ZàáâãèéêìíòóôõùúăđĩũơưỳýạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳýỵỷỹÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯỲÝẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲÝỴỶỸ'

def load_acronyms(filepath):
    """
    Đọc file chứa từ viết tắt và trả về một dictionary.
    """
    acronym_map = {}
    if not os.path.exists(filepath):
        print(f"Cảnh báo: Không tìm thấy file từ viết tắt '{filepath}'. Sẽ không thực hiện thay thế từ viết tắt.")
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = unicodedata.normalize('NFC', line.strip())
                if not line or ':' not in line or line.startswith('#'):
                    continue
                parts = line.split(':', 1)
                if len(parts) != 2:
                    print(f"Cảnh báo: Bỏ qua dòng {line_num+1} không hợp lệ trong '{filepath}': {line}")
                    continue

                acronyms_part, full_form = parts
                full_form = unicodedata.normalize('NFC', full_form.strip())
                acronyms = [unicodedata.normalize('NFC', a.strip()) for a in acronyms_part.split(',')]

                for acronym in acronyms:
                    if acronym:
                        acronym_map[acronym] = full_form
    except Exception as e:
        print(f"Lỗi khi đọc hoặc xử lý file từ viết tắt '{filepath}': {e}")
        return {}

    print(f"Đã tải thành công {len(acronym_map)} quy tắc thay thế từ viết tắt từ '{filepath}'.")
    return acronym_map

def clean_review_text(text, acronym_dict):
    """
    Làm sạch sâu một chuỗi văn bản bình luận.
    """
    if not isinstance(text, str):
        return ""

    try:
        # Chuẩn hóa Unicode
        cleaned_text = unicodedata.normalize('NFC', text)
        # Thay thế '\n' bằng khoảng trắng
        cleaned_text = cleaned_text.replace('\n', ' ')
        # Loại bỏ các ký tự đặc biệt
        for char in CHARS_TO_REMOVE:
            cleaned_text = cleaned_text.replace(char, '')
        # Loại bỏ emojis
        cleaned_text = emoji.replace_emoji(cleaned_text, replace='')
        # Thay thế từ viết tắt
        if acronym_dict:
            sorted_acronyms = sorted(acronym_dict.keys(), key=len, reverse=True)
            for acronym in sorted_acronyms:
                full_form = acronym_dict[acronym]
                pattern = rf"(?<![{VIETNAMESE_LETTERS}])({re.escape(acronym)})(?![{VIETNAMESE_LETTERS}])"
                cleaned_text = re.sub(pattern, full_form, cleaned_text, flags=re.IGNORECASE)
        # Loại bỏ khoảng trắng thừa
        cleaned_text = cleaned_text.strip()
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text
    except Exception as e:
        print(f"Lỗi khi làm sạch văn bản: {text[:100]}... Lỗi: {e}")
        return "" # Trả về chuỗi rỗng nếu có lỗi


# ==============================================================================
# LUỒNG THỰC THI CHÍNH
# ==============================================================================

def main():
    """
    Hàm chính điều khiển toàn bộ quy trình tiền xử lý.
    """
    print("--- BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ REVIEW ---")

    # --- Bước 1: Đọc file JSON gốc ---
    if not os.path.exists(INPUT_FILE):
        print(f"Lỗi: Không tìm thấy file đầu vào '{INPUT_FILE}'. Vui lòng kiểm tra lại.")
        return

    try:
        print(f"\n[1/6] Đang đọc file '{INPUT_FILE}'...")
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            original_data = json.load(f_in)
        print("Đọc file thành công.")

        # --- Bước 2: Lọc review và loại bỏ các trường không cần thiết ---
        print("\n[2/6] Đang xử lý và lọc các review không hợp lệ...")
        processed_data = preprocess_reviews(original_data)
        print("Lọc review hoàn tất.")

        # --- Bước 3: Trích xuất nội dung text của các review ---
        print("\n[3/6] Đang trích xuất nội dung văn bản từ các review đã lọc...")
        extracted_texts = extract_review_texts(processed_data)
        print(f"Đã trích xuất được {len(extracted_texts)} đoạn text review.")
        if not extracted_texts:
            print("Không có review nào được trích xuất. Dừng chương trình.")
            return

        # --- Bước 4: Làm sạch sâu văn bản và loại bỏ review trùng lặp ---
        print("\n[4/6] Đang làm sạch sâu và loại bỏ các review trùng lặp...")
        acronym_dict = load_acronyms(ACRONYM_FILE)

        cleaned_reviews = [clean_review_text(text, acronym_dict) for text in extracted_texts]
        # Loại bỏ các chuỗi rỗng có thể được tạo ra sau khi làm sạch
        cleaned_reviews = [review for review in cleaned_reviews if review]

        # Sử dụng set để loại bỏ trùng lặp hiệu quả
        unique_reviews_set = set(cleaned_reviews)
        unique_reviews_after_dedup = list(unique_reviews_set)

        num_duplicates_removed = len(cleaned_reviews) - len(unique_reviews_after_dedup)
        print(f"Đã loại bỏ {num_duplicates_removed} review trùng lặp sau khi làm sạch.")

        # --- Bước 5: Xóa các bình luận chỉ có một từ ---
        print("\n[5/6] Đang loại bỏ các review chỉ chứa một từ...")
        final_unique_reviews = [review for review in unique_reviews_after_dedup if len(review.split()) > 1]
        num_single_word_removed = len(unique_reviews_after_dedup) - len(final_unique_reviews)
        print(f"Đã loại bỏ {num_single_word_removed} review chỉ chứa một từ.")
        print(f"Số lượng review duy nhất cuối cùng: {len(final_unique_reviews)}")

        # --- Bước 6: Ghi kết quả cuối cùng ra file ---
        print(f"\n[6/6] Đang ghi kết quả vào file '{OUTPUT_FILE}'...")
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            json.dump(final_unique_reviews, f_out, ensure_ascii=False, indent=4)
        print(f"Hoàn thành! Đã lưu các review đã làm sạch vào '{OUTPUT_FILE}'.")
        print("\n--- QUY TRÌNH KẾT THÚC ---")

    except json.JSONDecodeError:
        print(f"Lỗi: File '{INPUT_FILE}' không phải là định dạng JSON hợp lệ.")
    except ImportError:
        print("Lỗi: Thư viện 'emoji' là bắt buộc nhưng chưa được cài đặt.")
        print("Vui lòng cài đặt bằng lệnh: pip install emoji")
    except Exception as e:
        import traceback
        print(f"Đã có lỗi không mong muốn xảy ra: {e}")
        # print(traceback.format_exc()) # Bỏ comment để debug lỗi chi tiết

if __name__ == "__main__":
    main()