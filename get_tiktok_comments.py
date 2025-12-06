import asyncio
import json
import os
from dotenv import load_dotenv
from datetime import datetime
from TikTokApi import TikTokApi
from TikTokApi.exceptions import TikTokException
import random
import re
from proxyproviders import Webshare

load_dotenv()

# Cấu hình từ khóa tìm kiếm
# Lưu ý: Với Hashtag không cần điền dấu #. Với cụm từ tìm kiếm thì điền bình thường.
SEARCH_KEYWORDS = ["DaiHocBinhDuong", "BDU", "Đại học Bình Dương"]

# Cấu hình giới hạn
VIDEOS_PER_KEYWORD = 10  # Số lượng video lấy cho mỗi từ khóa
COMMENTS_PER_VIDEO = 2500  # Số lượng bình luận lấy tối đa mỗi video

def preprocess_comment(comment):
    """
    Hàm làm sạch văn bản bình luận:
    - Bỏ tag người dùng (ví dụ: @username)
    - Xóa các khoảng trắng thừa ở đầu và cuối.
    """
    if not comment:
        return ""
    comment = re.sub(r'@\S+', '', comment)
    return comment.strip()

async def get_tiktok_comments():
    all_comments = []
    # Giữ phần tử đầu tiên là metadata để tương thích với logic của main.py
    all_comments.append({'info': 'Dữ liệu tổng hợp từ nhiều video theo từ khóa'})
    
    unique_comments_set = set()
    processed_video_ids = set()
    total_extracted = 0

    print(f"--- Bắt đầu quét TikTok theo từ khóa: {SEARCH_KEYWORDS} ---")

    WEBSHARE_API_KEY = os.getenv('WEBSHARE_API_KEY') 
    proxy_provider = None

    if WEBSHARE_API_KEY and WEBSHARE_API_KEY != "YOUR_WEBSHARE_API_KEY":
        print("Đang khởi tạo Webshare Proxy Provider...")
        proxy_provider = Webshare(api_key=WEBSHARE_API_KEY)
    else:
        print("CẢNH BÁO: Không tìm thấy Webshare API Key. Chạy không Proxy.")

    num_browser_sessions = 3

    async with TikTokApi() as api:
        try:
            print(f"Đang tạo {num_browser_sessions} session trình duyệt...")
            await api.create_sessions(
                headless=False, # Để False nếu cần debug
                browser='chromium',
                num_sessions=num_browser_sessions,
                proxy_provider=proxy_provider,
                override_browser_args=["--window-position=2000,0"],
                sleep_after=3
            )

            for keyword in SEARCH_KEYWORDS:
                print(f"\n>>> Đang xử lý từ khóa: '{keyword}'")
                
                # --- LOGIC CHỌN PHƯƠNG THỨC TÌM KIẾM ---
                video_iterator = None
                using_fallback = False

                # Nếu không có dấu cách -> Chắc chắn là Hashtag -> Dùng API Hashtag
                if " " not in keyword:
                    print(f"   -> Chế độ: Hashtag #{keyword}")
                    tag = api.hashtag(name=keyword)
                    video_iterator = tag.videos(count=VIDEOS_PER_KEYWORD)
                
                # Nếu có dấu cách -> Thử dùng Search, nếu lỗi thì fallback về Hashtag
                else:
                    print(f"   -> Chế độ: Search query '{keyword}'")
                    try:
                         # obj_type='item' đại diện cho Video trong TikTok Search
                         video_iterator = api.search.search_type(keyword, obj_type='item', count=VIDEOS_PER_KEYWORD)
                    except Exception as e:
                        print(f"   ! Không gọi được Search API ({e}). Sẽ chuyển sang phương án dự phòng.")
                        using_fallback = True

                # --- VÒNG LẶP QUÉT VIDEO ---
                try:
                    if using_fallback:
                        raise TikTokException("Force fallback")

                    count_videos_found = 0
                    async for video in video_iterator:
                        count_videos_found += 1
                        try:
                            if video.id in processed_video_ids:
                                continue
                            
                            processed_video_ids.add(video.id)
                            print(f"   - Đang quét Video ID: {video.id}...")

                            # Lấy bình luận
                            count_in_video = 0
                            async for comment in video.comments(count=COMMENTS_PER_VIDEO):
                                comment_dict = comment.as_dict
                                user = comment_dict.get('user', {})
                                raw_text = comment_dict.get('text', '')
                                cleaned_text = preprocess_comment(raw_text)

                                if cleaned_text and cleaned_text not in unique_comments_set:
                                    create_ts = comment_dict.get('create_time', 0)
                                    time_str = datetime.fromtimestamp(create_ts).strftime('%d/%m/%Y')

                                    data_entry = {
                                        'user': user.get('unique_id', 'Ẩn danh'),
                                        'comment': cleaned_text,
                                        'create_time': time_str,
                                        'source_keyword': keyword,
                                        'video_id': video.id
                                    }

                                    all_comments.append(data_entry)
                                    unique_comments_set.add(cleaned_text)
                                    count_in_video += 1
                                    total_extracted += 1

                            print(f"     + Đã lấy {count_in_video} bình luận.")
                            await asyncio.sleep(random.uniform(1.0, 2.0))

                        except Exception as e_vid:
                            print(f"     ! Lỗi video {video.id}: {str(e_vid)[:50]}...")
                            continue
                    
                    if count_videos_found == 0 and " " in keyword:
                         print("   ! Search trả về 0 kết quả. Có thể do API hạn chế.")
                         raise TikTokException("Empty search results")

                except (TikTokException, AttributeError, Exception) as e_iter:
                    # --- FALLBACK: NẾU SEARCH LỖI, CHUYỂN SANG HASHTAG ---
                    if " " in keyword:
                        print(f"   ! Gặp vấn đề khi Search ('{str(e_iter)[:100]}').")
                        fallback_tag_name = keyword.replace(" ", "")
                        print(f"   => Đang chuyển sang chế độ dự phòng: Hashtag #{fallback_tag_name}")
                        
                        try:
                            tag_fallback = api.hashtag(name=fallback_tag_name)
                            async for video in tag_fallback.videos(count=VIDEOS_PER_KEYWORD):
                                if video.id in processed_video_ids: continue
                                processed_video_ids.add(video.id)
                                print(f"   [Fallback] - Đang quét Video ID: {video.id}...")
                                
                                count_in_video = 0
                                async for comment in video.comments(count=COMMENTS_PER_VIDEO):
                                    comment_dict = comment.as_dict
                                    cleaned_text = preprocess_comment(comment_dict.get('text', ''))
                                    if cleaned_text and cleaned_text not in unique_comments_set:
                                        data_entry = {
                                            'user': comment_dict.get('user', {}).get('unique_id', 'Ẩn danh'),
                                            'comment': cleaned_text,
                                            'create_time': datetime.fromtimestamp(comment_dict.get('create_time', 0)).strftime('%d/%m/%Y'),
                                            'source_keyword': fallback_tag_name
                                        }
                                        all_comments.append(data_entry)
                                        unique_comments_set.add(cleaned_text)
                                        count_in_video += 1
                                        total_extracted += 1
                                print(f"     + Đã lấy {count_in_video} bình luận.")
                                await asyncio.sleep(1)
                        except Exception as e_fallback:
                            print(f"     ! Fallback cũng thất bại: {e_fallback}")
                    else:
                        print(f"   ! Lỗi khi quét hashtag '{keyword}': {e_iter}")

                # Nghỉ giữa các từ khóa
                await asyncio.sleep(2)

        except TikTokException as e_main:
            print(f"Đã xảy ra lỗi chung TikTokApi: {e_main}")
        
        finally:
            print("\nĐang đóng trình duyệt...")

    print(f"\n=== Hoàn tất! ===")
    print(f"Tổng số video đã quét: {len(processed_video_ids)}")
    print(f"Tổng số bình luận duy nhất thu thập được: {total_extracted}")
    
    return all_comments

if __name__ == "__main__":
    asyncio.run(get_tiktok_comments())