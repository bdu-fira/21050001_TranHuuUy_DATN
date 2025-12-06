import flask
from flask import Flask, request, render_template_string, redirect, url_for, flash
import json
import os
from google_map import get_all_google_maps_reviews
import logging

# Configure logging for the web app as well
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
# IMPORTANT: Change this secret key for production environments!
# You can generate one using: python -c 'import os; print(os.urandom(24))'
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

OUTPUT_FILE = 'reviews.json'

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cào dữ liệu Google Maps Reviews</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .container { max-width: 600px; margin: auto; padding: 20px; border: 1px solid #ccc; border-radius: 5px; }
        label { display: block; margin-bottom: 5px; }
        input[type="text"], input[type="url"] { width: 100%; padding: 8px; margin-bottom: 10px; box-sizing: border-box; }
        button { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .flash { padding: 10px; margin-top: 15px; border-radius: 3px; }
        .flash.success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .flash.error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Cào dữ liệu Google Maps Reviews</h1>

        {% with messages = get_flashed_messages(with_categories=true) %}
          {% if messages %}
            {% for category, message in messages %}
              <div class="flash {{ category }}">{{ message }}</div>
            {% endfor %}
          {% endif %}
        {% endwith %}

        <form action="{{ url_for('scrape') }}" method="post">
            <label for="map_url">Nhập link Google Maps:</label>
            <input type="url" id="map_url" name="map_url" required placeholder="https://www.google.com/maps/place/...">
            <button type="submit">Cào dữ liệu</button>
        </form>

        <p style="margin-top: 20px;">Dữ liệu sẽ được lưu vào file <code>{{ output_filename }}</code>.</p>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    """Renders the main page with the input form."""
    return render_template_string(HTML_TEMPLATE, output_filename=OUTPUT_FILE)

@app.route('/scrape', methods=['POST'])
def scrape():
    """Handles the form submission, triggers scraping, and saves data."""
    map_url = request.form.get('map_url')

    if not map_url:
        flash('Vui lòng nhập URL Google Maps.', 'error')
        return redirect(url_for('index'))

    logging.info(f"Received scrape request for URL: {map_url}")
    flash(f"Đang bắt đầu cào dữ liệu cho: {map_url[:50]}...", 'success') # Show feedback immediately

    # Run the scraper function
    # Using a simple direct call here. For long-running tasks, consider background jobs (Celery, RQ).
    try:
        scraped_data = get_all_google_maps_reviews(map_url)

        if isinstance(scraped_data, str):
            flash(f"Lỗi khi cào dữ liệu: {scraped_data}", 'error')
            logging.error(f"Scraping failed for {map_url}: {scraped_data}")
            return redirect(url_for('index'))

        if scraped_data and 'reviews' in scraped_data:
            place_name = scraped_data.get('place_name', 'Không rõ tên')
            num_reviews = len(scraped_data['reviews'])
            logging.info(f"Scraping successful for '{place_name}'. Found {num_reviews} reviews.")

            # --- Load existing data and append ---
            all_data = []
            if os.path.exists(OUTPUT_FILE):
                try:
                    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                        # Handle empty or invalid JSON file
                        content = f.read()
                        if content.strip(): # Check if file is not empty
                            all_data = json.loads(content)
                            if not isinstance(all_data, list): # Ensure it's a list
                                logging.warning(f"Output file '{OUTPUT_FILE}' did not contain a list. Resetting.")
                                all_data = []
                        else: # File is empty
                            all_data = []
                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON from {OUTPUT_FILE}. File might be corrupted. Starting fresh.")
                    # Optionally back up the corrupted file here
                    all_data = []
                except Exception as e:
                    logging.error(f"Error reading {OUTPUT_FILE}: {e}")
                    flash(f"Lỗi khi đọc file JSON hiện có: {e}", 'error')
                    return redirect(url_for('index'))

            # Append the new data as a dictionary for this place
            # Check if this place was already scraped to avoid duplicates (optional, simple check by name)
            existing_names = {item.get('place_name') for item in all_data if isinstance(item, dict)}
            if place_name != 'Địa điểm không xác định' and place_name in existing_names:
                 flash(f"Địa điểm '{place_name}' đã tồn tại trong file JSON. Bỏ qua lần cào này.", 'warning')
                 logging.warning(f"Place '{place_name}' already exists in {OUTPUT_FILE}. Skipping append.")
            else:
                 all_data.append(scraped_data)

                 # --- Save updated data ---
                 try:
                     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                         json.dump(all_data, f, ensure_ascii=False, indent=4)
                     logging.info(f"Successfully appended data for '{place_name}' to {OUTPUT_FILE}.")
                     flash(f"Đã cào thành công {num_reviews} đánh giá cho '{place_name}' và lưu vào {OUTPUT_FILE}.", 'success')
                 except Exception as e:
                     logging.error(f"Error writing to {OUTPUT_FILE}: {e}")
                     flash(f"Lỗi khi ghi dữ liệu vào file JSON: {e}", 'error')

        else:
            flash("Không nhận được dữ liệu hợp lệ từ trình cào.", 'error')
            logging.error(f"Scraper returned None or invalid data for {map_url}")

    except Exception as e:
        logging.exception(f"Unexpected error during scraping process for {map_url}: {e}")
        flash(f"Đã xảy ra lỗi không mong muốn trong quá trình xử lý: {e}", 'error')

    return redirect(url_for('index')) # Redirect back to the main page

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)