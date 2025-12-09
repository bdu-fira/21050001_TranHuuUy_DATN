from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from bs4 import BeautifulSoup
import time
import os
from dotenv import load_dotenv

load_dotenv()

def scroll_reviews(driver):
    try:
        # Selector cho phần chứa reviews
        scrollable_selector = '#QA0Szd > div > div > div:nth-child(1) > div:nth-child(2) > div > div:nth-child(1) > div > div > div:nth-child(2)'
        
        scroll_script = f"""
        var scrollableElement = document.querySelector('{scrollable_selector}');
        var initialHeight = scrollableElement.scrollHeight;
        scrollableElement.scrollTop = initialHeight;
        return initialHeight;
        """
        
        # Biến theo dõi số lần không thể cuộn thêm
        no_scroll_count = 0
        max_no_scroll = 2  # Số lần không thể cuộn tối đa

        while no_scroll_count < max_no_scroll:
            # Thực hiện cuộn
            initial_height = driver.execute_script(scroll_script)
            
            # Chờ để trang kịp tải thêm nội dung
            time.sleep(1)
            
            # Kiểm tra xem có nội dung mới không
            check_height_script = f"""
            var scrollableElement = document.querySelector('{scrollable_selector}');
            return scrollableElement.scrollHeight;
            """
            new_height = driver.execute_script(check_height_script)
            
            # Nếu chiều cao không thay đổi, tăng biến đếm
            if new_height <= initial_height:
                no_scroll_count += 1
            else:
                no_scroll_count = 0  # Đặt lại nếu vẫn còn nội dung mới
            
        #     print(f"Cuộn: Chiều cao ban đầu {initial_height}, Chiều cao mới {new_height}")
        
        # print(f"Đã cuộn xong sau {no_scroll_count} lần không tải thêm nội dung")
        
        # Chờ thêm để đảm bảo nội dung đã tải
        time.sleep(2)
    
    except Exception as e:
        print(f"Lỗi khi cuộn trang: {e}")


def get_all_google_maps_reviews():
    place_id = os.getenv('GMAP_PLACE_ID')

    if not place_id:
        print("LỖI CRITICAL: Không tìm thấy biến 'GMAP_PLACE_ID' trong file .env!")
        print("Vui lòng thêm dòng: GMAP_PLACE_ID=YourPlaceIDHere vào file .env")
        exit(1)
    
    options = Options()
    options.add_argument("--lang=vi")
    # Ẩn cửa sổ trình duyệt
    options.add_argument("--headless=new")
    options.add_argument("--window-position=-2400,-2400")

    driver = webdriver.Edge(options=options)

    try:
        # Tạo URL từ Place ID
        url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
        
        driver.get(url)

        try:
            # Đợi cho trang tải xong trong vòng 7 giây
            WebDriverWait(driver, 7).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.rogA2c"))
            )
        except TimeoutException:
            raise ValueError("Vui lòng kiểm tra lại địa chỉ hoặc thử lại sau. Không thể tải trang địa điểm.")

        # Lấy tên địa điểm
        try:
            place_name_element = driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div/div[1]/div[1]')
            place_name = place_name_element.text.strip()
            print(f"Tên địa điểm: {place_name}")
        except Exception as e:
            place_name = "Địa điểm không xác định"
            print(f"Không thể lấy tên địa điểm: {e}")

        # Tìm nút "Bài đánh giá" dựa trên các thuộc tính cụ thể
        review_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[role="tab"][aria-label^="Bài đánh giá"]'))
        )

        # Nhấn vào nút "Bài đánh giá"
        try:
            review_button.click()
            print("Đã nhấn nút Bài đánh giá")
        except ElementClickInterceptedException:
            driver.execute_script("arguments[0].click();", review_button)
            print("Đã nhấn nút Bài đánh giá bằng JavaScript")

        # Đợi cho phần tử chứa đánh giá xuất hiện
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.jftiEf'))
        )

        # Cuộn trang để lấy đánh giá
        scroll_reviews(driver)

        # Lấy nội dung HTML
        html_content = driver.page_source
    except TimeoutException:
        print("Timeout waiting for page to load")
        return None
    except NoSuchElementException:
        print("Could not find the reviews section")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    finally:
        driver.quit()

    # Phân tích HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    review_elements = soup.find_all('div', class_='jftiEf')

    reviews = []
    for element in review_elements:
        try:
            # Lấy tác giả
            author_elem = element.find('div', class_='d4r55')
            author = author_elem.text.strip() if author_elem else 'Ẩn danh'

            # Lấy rating
            rating_elem = element.find('span', class_='kvMYJc')
            rating = rating_elem['aria-label'].split()[0] if rating_elem and 'aria-label' in rating_elem.attrs else '0'

            # Lấy nội dung review
            text_elem = element.find('span', class_='wiI7pd')
            text = text_elem.text.strip() if text_elem else ''

            # Lấy thời gian
            rtime_elem = element.find('span', class_='rsqaWe')
            rtime = rtime_elem.text.strip() if rtime_elem else 'Thời gian không xác định'

            # Chỉ thêm review nếu có nội dung
            if text:
                reviews.append({
                    'author': author,
                    'rating': rating,
                    'text': text,
                    'rtime': rtime
                })

        except Exception as e:
            print(f"Lỗi khi xử lý review: {e}")
            continue

    return {
        'place_name': place_name,
        'reviews': reviews
    }