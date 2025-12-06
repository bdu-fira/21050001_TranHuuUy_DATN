from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    JavascriptException,
    NoSuchElementException,
    ElementClickInterceptedException,
    WebDriverException,
)
from bs4 import BeautifulSoup
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def scroll_reviews(driver):
    """Scrolls the review section using the original selector logic."""
    try:
        # Selector cho phần chứa reviews (từ code gốc của bạn)
        # LƯU Ý: Selector này rất cụ thể và có thể dễ bị hỏng nếu Google thay đổi layout.
        scrollable_selector = "#QA0Szd > div > div > div:nth-child(1) > div:nth-child(2) > div > div:nth-child(1) > div > div > div:nth-child(2)"

        logging.info(f"Attempting to use original scroll selector: {scrollable_selector}")

        # Đợi phần tử cuộn xuất hiện (thêm bước kiểm tra nhỏ)
        try:
             WebDriverWait(driver, 10).until(
                 EC.presence_of_element_located((By.CSS_SELECTOR, scrollable_selector))
             )
             logging.info("Scrollable element potentially found with original selector.")
        except TimeoutException:
             logging.error(f"TimeoutException: Could not find the element with the original selector '{scrollable_selector}' within 10 seconds. Scrolling may fail.")
             # Có thể thử một selector dự phòng ở đây nếu muốn, hoặc chỉ báo lỗi
             # Ví dụ thử selector khác:
             # scrollable_selector = 'div[role="main"][aria-label]' # Selector dự phòng
             # logging.warning(f"Falling back to selector: {scrollable_selector}")
             # WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, scrollable_selector)))
             # Nếu không muốn dự phòng, thì return hoặc raise lỗi
             return # Thoát nếu không tìm thấy phần tử cuộn chính

        scroll_script_get_height = f"""
        var scrollableElement = document.querySelector('{scrollable_selector}');
        if (!scrollableElement) {{ return -1; }} // Trả về -1 nếu không tìm thấy phần tử
        return scrollableElement.scrollHeight;
        """

        scroll_script_scroll = f"""
        var scrollableElement = document.querySelector('{scrollable_selector}');
        if (!scrollableElement) {{ return; }} // Không làm gì nếu không tìm thấy
        scrollableElement.scrollTop = scrollableElement.scrollHeight;
        """

        # Biến theo dõi số lần không thể cuộn thêm
        no_scroll_count = 0
        max_no_scroll = 5  # Tăng nhẹ giới hạn để thử nhiều hơn một chút
        last_height = -1 # Khởi tạo chiều cao cuối cùng

        logging.info("Starting review scroll using original logic...")

        # Lấy chiều cao ban đầu
        try:
             last_height = driver.execute_script(scroll_script_get_height)
             if last_height == -1:
                 logging.error("Scrollable element not found via JavaScript during initial height check.")
                 return # Thoát nếu không thể lấy chiều cao ban đầu
             logging.info(f"Initial scroll height: {last_height}")
        except JavascriptException as e:
             logging.error(f"JavaScript error getting initial height: {e}")
             return # Thoát nếu có lỗi JS

        while no_scroll_count < max_no_scroll:
            # Thực hiện cuộn
            try:
                 driver.execute_script(scroll_script_scroll)
                 logging.info("Executed scroll command.")
            except JavascriptException as e:
                 logging.error(f"JavaScript error during scroll execution: {e}")
                 # Có thể nên dừng lại ở đây nếu lệnh cuộn bị lỗi
                 break

            # Chờ để trang kịp tải thêm nội dung - tăng nhẹ thời gian chờ
            time.sleep(2) # Cho 2 giây thay vì 1

            # Kiểm tra xem có nội dung mới không (lấy chiều cao mới)
            current_height = -1
            try:
                 current_height = driver.execute_script(scroll_script_get_height)
                 if current_height == -1:
                     logging.warning("Scrollable element potentially disappeared during scroll check.")
                     # Có thể phần tử đã thay đổi, dừng cuộn
                     break
            except JavascriptException as e:
                 logging.error(f"JavaScript error getting current height: {e}")
                 # Dừng lại nếu không thể kiểm tra chiều cao
                 break

            logging.info(f"Scroll check: Last height={last_height}, Current height={current_height}")

            # Nếu chiều cao không thay đổi đáng kể (hoặc phần tử biến mất), tăng biến đếm
            # Thêm điều kiện current_height > -1 để đảm bảo phần tử còn tồn tại
            if current_height > -1 and current_height <= last_height:
                no_scroll_count += 1
                logging.info(f"Scroll height did not increase significantly. no_scroll_count = {no_scroll_count}")
            elif current_height > last_height:
                last_height = current_height # Cập nhật chiều cao cuối cùng
                no_scroll_count = 0  # Đặt lại nếu vẫn còn nội dung mới
                logging.info(f"Scroll height increased to {current_height}. Resetting no_scroll_count.")
            else: # Trường hợp current_height == -1 (đã log warning ở trên)
                 no_scroll_count += 1 # Coi như không cuộn được nữa
                 logging.warning(f"Assuming no scroll due to element issue. no_scroll_count = {no_scroll_count}")


        logging.info(f"Finished scrolling loop after {no_scroll_count} consecutive no-scroll attempts.")

        # Chờ thêm để đảm bảo nội dung đã tải (nếu có)
        time.sleep(2)

    except JavascriptException as e:
         logging.error(f"A critical JavaScript error occurred during scrolling setup or loop: {e}")
    except Exception as e:
        # Bắt các lỗi không mong muốn khác trong quá trình cuộn
        logging.error(f"An unexpected error occurred in scroll_reviews: {e}", exc_info=True) # Thêm exc_info=True để log traceback


def get_all_google_maps_reviews(url: str):
    """
    Scrapes all reviews for a given Google Maps place URL.

    Args:
        url: The full URL of the Google Maps place.

    Returns:
        A dictionary containing the place name and a list of reviews,
        or None if an error occurs.
        Returns an error message string if input URL is invalid.
    """
    if not url or not url.startswith("https://www.google.com/maps/place/"):
        logging.error(f"Invalid Google Maps URL provided: {url}")
        return "URL không hợp lệ. Vui lòng cung cấp URL Google Maps đầy đủ (bắt đầu bằng https://www.google.com/maps/place/)."

    options = Options()
    options.add_argument("--lang=vi")
    # options.add_argument("--headless=new")
    # Keep window position for debugging if needed, otherwise remove/comment
    # options.add_argument("--window-position=-2400,-2400")
    options.add_argument("--disable-gpu") # Often needed for headless mode
    options.add_argument("--window-size=1920,1080") # Specify window size

    driver = None # Initialize driver to None for finally block
    try:
        # Consider using webdriver-manager for automatic driver management
        # service = Service(EdgeChromiumDriverManager().install())
        # driver = webdriver.Edge(service=service, options=options)
        # --- OR --- Use explicit path if webdriver-manager is not used
        # Make sure msedgedriver.exe is in your PATH or provide the path
        # service = Service(executable_path="path/to/your/msedgedriver.exe")
        driver = webdriver.Edge(options=options)
        driver.set_page_load_timeout(30) # Increase page load timeout

        logging.info(f"Attempting to load URL: {url}")
        driver.get(url)

        # Wait for a key element indicating the place page has loaded (e.g., the place title)
        # Using a more general selector for the title h1 element
        title_selector = "h1.DUwDvf"
        try:
            WebDriverWait(driver, 15).until( # Increased wait time
                EC.presence_of_element_located((By.CSS_SELECTOR, title_selector))
            )
            logging.info("Place page loaded successfully.")
        except TimeoutException:
            logging.error("Timeout waiting for place page main elements to load.")
            # Try checking if it's a consent form page
            try:
                consent_button = driver.find_element(By.XPATH, "//button[.//span[contains(text(), 'Accept all') or contains(text(), 'Chấp nhận tất cả')]]")
                logging.info("Consent form detected. Clicking 'Accept all'.")
                consent_button.click()
                WebDriverWait(driver, 10).until(
                     EC.presence_of_element_located((By.CSS_SELECTOR, title_selector))
                 )
                logging.info("Place page loaded after accepting consent.")
            except Exception as consent_error:
                 logging.error(f"Could not handle consent form or page still didn't load: {consent_error}")
                 raise ValueError(
                     "Không thể tải trang địa điểm. Vui lòng kiểm tra lại URL hoặc thử lại sau."
                 ) from consent_error


        # Get place name
        place_name = "Địa điểm không xác định"
        try:
            # Use the more robust selector found earlier
            place_name_element = WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, title_selector))
            )
            place_name = place_name_element.text.strip()
            logging.info(f"Place name: {place_name}")
        except Exception as e:
            logging.warning(f"Could not retrieve place name: {e}")

        # Find and click the "Reviews" tab
        review_button = None
        possible_review_selectors = [
            'button[role="tab"][aria-label*="Bài đánh giá"]', # Vietnamese
            'button[role="tab"][aria-label*="Reviews"]',     # English
            'button[jsaction*="pane.rating.tab"]',           # Action based
        ]
        for selector in possible_review_selectors:
            try:
                review_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                logging.info(f"Found review button with selector: {selector}")
                break # Stop searching once found
            except TimeoutException:
                logging.info(f"Review button not found with selector: {selector}")
                continue # Try next selector

        if not review_button:
             logging.error("Could not find the 'Reviews' tab/button.")
             raise NoSuchElementException("Không tìm thấy tab 'Bài đánh giá'.")

        # Click the review button (handle potential interception)
        try:
            # Use JS click as it's often more reliable
            driver.execute_script("arguments[0].click();", review_button)
            logging.info("Clicked 'Reviews' tab using JavaScript.")
        except Exception as click_error:
             logging.error(f"Failed to click review button: {click_error}")
             raise ElementClickInterceptedException("Không thể nhấp vào tab 'Bài đánh giá'.") from click_error

        # Wait for review elements to appear
        review_element_selector = "div.jftiEf" # Selector for individual review blocks
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, review_element_selector))
            )
            logging.info("Review section loaded.")
        except TimeoutException:
            logging.warning("Timeout waiting for review elements (div.jftiEf). Maybe no reviews yet?")
            # Check if there's a "Be the first to review" message
            try:
                 no_reviews_msg = driver.find_element(By.XPATH, "//*[contains(text(), 'Hãy là người đầu tiên đánh giá')] | //*[contains(text(), 'Be the first to review')]")
                 if no_reviews_msg:
                     logging.info("No reviews found for this place.")
                     return {"place_name": place_name, "reviews": []} # Return empty list if no reviews
            except NoSuchElementException:
                 logging.error("Timed out waiting for reviews, and no 'no reviews' message found.")
                 raise TimeoutException("Không thể tải mục đánh giá.")


        # Scroll down to load all reviews
        scroll_reviews(driver)

        # Get page source after scrolling
        logging.info("Getting final page source...")
        html_content = driver.page_source

    except WebDriverException as e:
        logging.error(f"WebDriver error occurred: {e}")
        return f"Lỗi trình duyệt: {e}"
    except ValueError as e: # Catch the specific error raised for load failure
        logging.error(f"Page load error: {e}")
        return str(e)
    except TimeoutException as e:
        logging.error(f"Timeout error during scraping: {e}")
        return "Hết thời gian chờ trong quá trình cào dữ liệu."
    except NoSuchElementException as e:
        logging.error(f"Element not found during scraping: {e}")
        return f"Không tìm thấy phần tử cần thiết trên trang: {e}"
    except ElementClickInterceptedException as e:
        logging.error(f"Element click intercepted: {e}")
        return f"Không thể nhấp vào phần tử (có thể bị che khuất): {e}"
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) # Log stack trace
        return f"Lỗi không mong đợi: {e}"
    finally:
        if driver:
            logging.info("Closing browser.")
            driver.quit()

    # Parse HTML content
    logging.info("Parsing HTML content with BeautifulSoup...")
    soup = BeautifulSoup(html_content, "html.parser")
    review_elements = soup.find_all("div", class_="jftiEf") # Find all review blocks

    reviews = []
    logging.info(f"Found {len(review_elements)} potential review elements.")

    for element in review_elements:
        try:
            # Extract author name
            author_elem = element.find("div", class_="d4r55")
            author = author_elem.text.strip() if author_elem else "Ẩn danh"

            # Extract rating - Look for the aria-label on the span inside the rating container
            rating_container = element.find("span", class_="kvMYJc")
            rating = "0"
            if rating_container:
                rating_span = rating_container.find("span", {"aria-label": True})
                if rating_span and "aria-label" in rating_span.attrs:
                    # Extract the number part (e.g., "4 sao" -> "4")
                    label_text = rating_span["aria-label"]
                    rating_parts = label_text.split()
                    if rating_parts and rating_parts[0].isdigit():
                        rating = rating_parts[0]


            # Extract review text - Check for "Tùy chọn khác" (More options) button first
            text = ""
            more_button = element.find("button", class_="w8nwRe") # Button to expand long reviews
            text_elem = element.find("span", class_="wiI7pd") # The actual review text span

            if text_elem:
                text = text_elem.text.strip()
            elif more_button: # If only the button is there, the text might be hidden initially (less common post-scroll)
                 text = "[Nội dung ẩn sau nút 'Thêm']" # Placeholder
                 logging.warning("Found 'More' button, text might be truncated if expansion failed.")
            else:
                 # Sometimes review is just a rating without text
                 pass # No text found, keep text=""


            # Extract review time
            rtime_elem = element.find("span", class_="rsqaWe")
            rtime = rtime_elem.text.strip() if rtime_elem else "Thời gian không xác định"

            # Only add review if it has text or a valid rating (sometimes only rating is given)
            # We will add reviews even without text, as a rating alone is valid feedback.
            reviews.append(
                {"author": author, "rating": rating, "text": text, "rtime": rtime}
            )

        except Exception as e:
            logging.error(f"Error processing a review element: {e}", exc_info=True)
            continue # Skip this review if parsing fails

    logging.info(f"Successfully parsed {len(reviews)} reviews.")
    return {"place_name": place_name, "reviews": reviews}
