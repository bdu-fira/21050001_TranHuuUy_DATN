import json
import os
import unicodedata

def filter_reviews(target_filename, temp_filename="temp.json"):
    """
    Lọc và xóa các review trùng lặp từ file mục tiêu.
    Đã bao gồm bước chuẩn hóa Unicode.

    Hàm này sẽ xóa các review nếu:
    1. Review đó là một bản sao (trùng lặp) trong chính file mục tiêu.
    2. Review đó xuất hiện trong file temp.
    3. Review đó có ít hơn 4 từ.

    Args:
        target_filename (str): Tên của file JSON cần được lọc.
        temp_filename (str): Tên của file JSON chứa các review cần xóa.
    """
    # --- Bước 1: Kiểm tra sự tồn tại của các file ---
    if not os.path.exists(target_filename):
        print(f"Lỗi: Không tìm thấy file '{target_filename}'. Vui lòng kiểm tra lại tên file.")
        return
    if not os.path.exists(temp_filename):
        print(f"Lỗi: Không tìm thấy file '{temp_filename}'. Vui lòng đảm bảo file này tồn tại.")
        return

    try:
        # --- Bước 2: Tải dữ liệu từ các file JSON ---
        print(f"Đang đọc file mục tiêu: '{target_filename}'...")
        with open(target_filename, 'r', encoding='utf-8') as f:
            target_reviews = json.load(f)

        print(f"Đang đọc file tạm: '{temp_filename}'...")
        with open(temp_filename, 'r', encoding='utf-8') as f:
            temp_reviews_list = json.load(f)

        initial_count = len(target_reviews)
        print(f"Số lượng review ban đầu trong file mục tiêu: {initial_count}")

        # --- Bước 2.5: Chuẩn hóa Unicode (NFC) ---
        # Bước này quan trọng để đảm bảo các chuỗi so sánh chính xác
        # Ví dụ: đảm bảo 'Tiếng Việt' dựng sẵn và 'Tiếng Việt' tổ hợp được coi là một
        print("Đang chuẩn hóa Unicode (NFC) cho dữ liệu...")
        target_reviews = [unicodedata.normalize('NFC', r) for r in target_reviews]
        temp_reviews_list = [unicodedata.normalize('NFC', r) for r in temp_reviews_list]

        # Chuyển danh sách review tạm thành một tập hợp (set) để tra cứu nhanh hơn
        temp_reviews_set = set(temp_reviews_list)
        
        # --- Bước 3: Xóa dữ liệu trùng lặp trong file mục tiêu ---
        # Sử dụng dict.fromkeys để xóa trùng lặp mà vẫn giữ nguyên thứ tự ban đầu
        print("Bắt đầu xóa dữ liệu trùng lặp...")
        unique_reviews = list(dict.fromkeys(target_reviews))
        count_after_deduplication = len(unique_reviews)
        duplicates_removed = initial_count - count_after_deduplication
        
        if duplicates_removed > 0:
            print(f"Đã xóa {duplicates_removed} review bị trùng lặp (bao gồm trùng do lỗi Unicode).")
        else:
            print("Không tìm thấy review nào trùng lặp trong file mục tiêu.")

        # --- Bước 4: Lọc dữ liệu theo các điều kiện khác ---
        print("Bắt đầu quá trình lọc (xóa review trong file temp và review quá ngắn)...")
        
        # Lọc dựa trên danh sách đã được xóa trùng lặp
        # Điều kiện 1: review không nằm trong tập hợp review tạm
        # Điều kiện 2: review có 4 từ trở lên
        filtered_reviews = [
            review for review in unique_reviews 
            if review not in temp_reviews_set and len(review.split()) >= 4
        ]
        
        final_count = len(filtered_reviews)
        total_removed = initial_count - final_count

        # --- Bước 5: Ghi lại kết quả vào file gốc ---
        print(f"Đã lọc xong. Ghi lại {final_count} review hợp lệ vào file '{target_filename}'...")
        with open(target_filename, 'w', encoding='utf-8') as f:
            json.dump(filtered_reviews, f, ensure_ascii=False, indent=4)
            
        print("\n--- TÓM TẮT KẾT QUẢ ---")
        print(f"Tổng số review ban đầu: {initial_count}")
        print(f"Số review trùng lặp đã xóa: {duplicates_removed}")
        print(f"Tổng số review đã bị xóa (bao gồm trùng lặp, trong file temp, và quá ngắn): {total_removed}")
        print(f"Tổng số review còn lại: {final_count}")
        print("--------------------------")

    except json.JSONDecodeError as e:
        print(f"Lỗi: File JSON không hợp lệ. Vui lòng kiểm tra lại cấu trúc file. Chi tiết: {e}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")


if __name__ == "__main__":
    # Yêu cầu người dùng nhập tên file JSON "chỉ định"
    target_file = input("Vui lòng nhập tên file JSON bạn muốn lọc (ví dụ: final_positive_reviews.json): ")
    
    # Gọi hàm để thực hiện công việc
    filter_reviews(target_file)