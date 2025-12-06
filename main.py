from flask import Flask, render_template, request, jsonify, session, json, redirect, make_response
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import utils
import re
import os
from dotenv import load_dotenv
import numpy as np
import functools
from underthesea import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from io import BytesIO
import base64
from collections import Counter
from nltk.util import ngrams
from get_gmap_reviews import get_all_google_maps_reviews
from get_tiktok_comments import get_tiktok_comments
import asyncio
from torch.nn.utils.rnn import pad_sequence

load_dotenv()

app = Flask(__name__)

# --- CẤU HÌNH BẢO MẬT ---
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key_if_env_missing')

# Cấu hình Cookie an toàn (Session Security)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True, # JavaScript không thể đọc cookie (chống XSS lấy cookie)
    SESSION_COOKIE_SECURE=False,  # Đặt là True nếu chạy HTTPS (production), False nếu chạy localhost
    SESSION_COOKIE_SAMESITE='Lax',# Chống CSRF cơ bản
)

csrf = CSRFProtect(app)

# Rate Limiting (Chống Spam)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"], # Mặc định cho mọi trang
    storage_uri="memory://"
)

# Kích hoạt Secure Headers (Talisman)
# content_security_policy=None: Tạm tắt CSP chặt chẽ để tránh lỗi load script/ảnh từ nguồn ngoài (như Chart.js, FontAwesome)
# force_https=False: Tạm tắt để chạy localhost (Đặt True khi deploy có chứng chỉ SSL)
Talisman(app, content_security_policy=None, force_https=False) 

# --- GLOBAL VARIABLES FOR MODEL LOADING ---
global_model = None
global_vocab = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

REPEATED_CHARS_PATTERN = re.compile(r'(.)\1{1,}')
SPACES_PATTERN = re.compile(r'([.,!?])')
CLEAN_SPACES_PATTERN = re.compile(r'\s+')
SPECIAL_CHARS_PATTERN = re.compile(r'\b\d+\b|\b\w{1}\b|[^\w\s]')

# --- CÁC HÀM HỖ TRỢ (HELPER FUNCTIONS) ---
def load_stopwords(filepath):
    stopwords = set()
    # Các từ phủ định KHÔNG NÊN coi là stopword trong phân tích cảm xúc
    keep_words = {"a ha", "biết bao", "bỏ mẹ", "cha chả", "chao ôi", "chu cha", "chui cha", "chết nỗi", "chết thật", "chết tiệt", "cóc khô",
                "dễ sợ", "nức nở", "oai oái", "oái", "phỉ phui", "quá chừng", "quá lắm", "quá sá", "quá thể", "quá trời", "quá xá", "quá đỗi",
                "sa sả", "than ôi", "thương ôi", "toé khói", "trời đất ơi", "úi", "úi chà", "úi dào", "xiết bao", "á", "á à", "ái", "ái chà", 
                "ái dà", "ô hay", "ô hô", "ô kìa", "ôi chao", "ôi thôi", "ối dào", "ối giời", "ối giời ơi", "ứ hự", "ứ ừ"}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word not in keep_words:
                    word = word.replace(' ', '_') 
                    stopwords.add(word)
    except Exception as e:
        print(f"Lỗi khi load stopwords: {e}")
    return stopwords

stopwords = load_stopwords('./data/vietnamese-stopwords.txt')

def init_model():
    """
    Khởi tạo Model và Vocabulary MỘT LẦN DUY NHẤT khi chạy server.
    Giúp tăng tốc độ vì không phải load lại file model nặng.
    """
    global global_model, global_vocab
    print(f"--- Đang khởi tạo Model trên thiết bị: {device} ---")
    try:
        config_fpath = 'configs/predict_config.yml'
        config = utils.get_config(config_fpath)
        
        global_vocab = torch.load(config["vocab_fpath"])
        global_model = torch.load(config["model_fpath"], map_location=device)
        global_model.eval() # Chuyển sang chế độ đánh giá (inference mode)
        print("--- Đã tải xong Model và Vocabulary thành công! ---")
    except Exception as e:
        print(f"!!! LỖI KHI TẢI MODEL: {e}")

def preprocess_text_for_charts(text):
    text = SPECIAL_CHARS_PATTERN.sub(' ', text)
    text = CLEAN_SPACES_PATTERN.sub(' ', text).strip()
    if text:
        text = word_tokenize(text, format='text')
        text = text.lower()
        
    return text

def generate_wordcloud(text):
    text = preprocess_text_for_charts(text)
    if not text: return ""
    words_list = [w for w in text.split() if w not in stopwords]
    
    if not words_list: return ""
    word_counts = Counter(words_list)
    clean_counts = {word.replace('_', ' '): count for word, count in word_counts.items()}
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(clean_counts)
    img = BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def generate_ngram_chart(ngram_counts, sentiment_type, n=2):
    if not ngram_counts: return ""
    ngrams_list = list(ngram_counts.keys())
    counts = list(ngram_counts.values())
    ngrams_str = [' '.join(ngram) for ngram in ngrams_list]

    plt.figure(figsize=(10, 5))
    plt.bar(ngrams_str, counts, color='skyblue')
    plt.xlabel(f'{n}-grams')
    plt.ylabel('Tần suất')
    plt.title(f'Tần suất {n}-grams trong bình luận {sentiment_type}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='PNG')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return img_base64

def generate_venn_diagram(positive_words, neutral_words, negative_words):
    plt.figure(figsize=(8, 8))
    try:
        venn3([set(positive_words), set(neutral_words), set(negative_words)], ('Tích cực', 'Trung tính', 'Tiêu cực'))
        plt.title('Biểu đồ Venn so sánh từ khóa')
        img = BytesIO()
        plt.savefig(img, format='PNG')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return img_base64
    except Exception:
        plt.close()
        return ""

def analyze_improvement_topics(negative_reviews):
    try:
        with open('data/improvement_topics.json', 'r', encoding='utf-8') as f:
            topics_dict = json.load(f)
    except Exception: return {}

    topic_analysis = {topic: {'count': 0, 'examples': []} for topic in topics_dict}
    for review in negative_reviews:
        review_text = review.get('processed_text', '').lower()
        for topic, keywords in topics_dict.items():
            for keyword in keywords:
                if keyword in review_text:
                    topic_analysis[topic]['count'] += 1
                    if len(topic_analysis[topic]['examples']) < 3 and review.get('text') not in topic_analysis[topic]['examples']:
                        topic_analysis[topic]['examples'].append(review.get('text'))
                    break 
    return dict(sorted([item for item in topic_analysis.items() if item[1]['count'] > 0], key=lambda item: item[1]['count'], reverse=True))

def analyze_strengths(positive_reviews):
    try:
        with open('data/strengths_topics.json', 'r', encoding='utf-8') as f:
            topics_dict = json.load(f)
    except Exception: return {}

    topic_analysis = {topic: {'count': 0, 'examples': []} for topic in topics_dict}
    for review in positive_reviews:
        review_text = review.get('processed_text', '').lower()
        for topic, keywords in topics_dict.items():
            for keyword in keywords:
                if keyword in review_text:
                    topic_analysis[topic]['count'] += 1
                    if len(topic_analysis[topic]['examples']) < 3 and review.get('text') not in topic_analysis[topic]['examples']:
                        topic_analysis[topic]['examples'].append(review.get('text'))
                    break 
    return dict(sorted([item for item in topic_analysis.items() if item[1]['count'] > 0], key=lambda item: item[1]['count'], reverse=True))

# --- CÁC HÀM XỬ LÝ TEXT ---
def remove_repeated_chars(text):
    normalized_text = REPEATED_CHARS_PATTERN.sub(r'\1', text)
    # print("- Sau khi chuẩn hóa chữ lặp:", normalized_text) # Optimized: Comment out
    return normalized_text

def normalized(text):
    text_with_spaces = SPACES_PATTERN.sub(r' \1 ', text)
    text_cleaned = CLEAN_SPACES_PATTERN.sub(' ', text_with_spaces).strip()
    text_lowercase = text_cleaned.lower()
    text_normalized = remove_repeated_chars(text_lowercase)
    
    # Optimized: Comment out prints
    # print("- Sau khi thêm khoảng trắng vào dấu câu:", text_with_spaces)
    # print("- Sau khi chuẩn hóa khoảng trắng:", text_cleaned)
    # print("- Sau khi chuyển thành chữ thường:", text_lowercase)
    return text_normalized

# Cache replacements
cached_replacements = {}
def load_replacements(filename):
    global cached_replacements
    if not cached_replacements:
        replacements = {}
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if ':' in line:
                        parts = line.split(':')
                        if len(parts) == 2:
                            keys = parts[0].strip().split(',')
                            value = parts[1].strip()
                            for key in keys:
                                replacements[key.strip()] = value
            cached_replacements = replacements
        except FileNotFoundError:
            cached_replacements = {}
    return cached_replacements

def replace_text(input_text, replacements):
    # print("\nCác bước xử lý văn bản:") # Optimized
    # print("- Văn bản gốc:", input_text) # Optimized
    
    cleaned_text = normalized(input_text)
    words = [replacements.get(word, word) for word in cleaned_text.split()]
    text_replaced = ' '.join(words)
    # print("- Sau khi thay thế từ viết tắt:", text_replaced) # Optimized
    
    text_cleaned = SPECIAL_CHARS_PATTERN.sub(' ', text_replaced)
    # print("- Sau khi loại bỏ ký tự đặc biệt, số và từ 1 ký tự:", text_cleaned) # Optimized
    return text_cleaned


# --- HÀM DỰ ĐOÁN ---
def predict_single_text(sentence):
    """Dùng cho route /predict (xử lý 1 câu)"""
    if not sentence or sentence.strip() == "":
        return None, {}

    # Sử dụng global_vocab và global_model
    corpus = [sentence]
    # Chuyển text thành tensor
    tensor = global_vocab.corpus_to_tensor(corpus)[0].to(device)
    tensor = tensor.unsqueeze(1) # [seq_len, 1]
    length_tensor = torch.LongTensor([len(tensor)])
    
    with torch.no_grad():
        output = global_model(tensor, length_tensor)
        
    probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    predicted_class_idx = np.argmax(probabilities)
    
    id2sentiment = {0: 'Tiêu cực', 1: 'Trung tính', 2: 'Tích cực'}
    sentiment_type = id2sentiment[predicted_class_idx]
    
    sentiment_probs = {
        'Tiêu cực': float(probabilities[0]),
        'Trung tính': float(probabilities[1]),
        'Tích cực': float(probabilities[2])
    }
    return sentiment_type, sentiment_probs


# --- DATABASE INIT ---
def init_database():
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS feedbacks (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, feedback TEXT NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def init_analysis_reports_database():
    conn = sqlite3.connect('analysis_reports.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS reports (id INTEGER PRIMARY KEY AUTOINCREMENT, place_name TEXT NOT NULL, total_reviews INTEGER, positive_count INTEGER, neutral_count INTEGER, negative_count INTEGER, avg_sentiment_type TEXT, positive_wordcloud TEXT, neutral_wordcloud TEXT, negative_wordcloud TEXT, positive_bigram_chart TEXT, neutral_bigram_chart TEXT, negative_bigram_chart TEXT, venn_diagram TEXT, improvement_suggestions TEXT, strengths_to_promote TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def init_users_database():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE, password TEXT NOT NULL)''')
    # Chỉ tạo user mặc định nếu bảng trống (tránh reset password mỗi lần restart)
    cursor.execute('SELECT count(*) FROM users')
    if cursor.fetchone()[0] == 0:
        admin_pass = os.getenv('ADMIN_PASSWORD', 'admin123') 
        hashed_password = generate_password_hash(admin_pass)
        sample_users = [('admin', hashed_password)]
        cursor.executemany('INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)', sample_users)
        print("--- Đã tạo tài khoản admin mặc định ---")
    conn.commit()
    conn.close()

def nocache(view):
    @functools.wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        return response
    return no_cache

# --- ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/submit_feedback', methods=['POST'])
@limiter.limit("5 per minute")  # Chỉ cho phép gửi 5 góp ý mỗi phút từ 1 IP
def submit_feedback():
    data = request.json
    name = data.get('name', '').strip()
    feedback = data.get('feedback', '').strip()
    if not name or not feedback: return jsonify({'status': 'error', 'message': 'Vui lòng nhập đầy đủ thông tin'}), 400
    if len(name) > 30: return jsonify({'status': 'error', 'message': 'Tên quá dài'}), 400
    if len(feedback) > 500: return jsonify({'status': 'error', 'message': 'Nội dung quá dài'}), 400
    try:
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO feedbacks (name, feedback, created_at) VALUES (?, ?, ?)', (name, feedback, datetime.now()))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Góp ý đã được ghi nhận'}), 200
    except sqlite3.Error as e: return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/view_feedbacks')
@nocache
def view_feedbacks():
    if not session.get('logged_in'): return redirect('/')
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM feedbacks ORDER BY created_at DESC LIMIT 100')
    feedbacks = cursor.fetchall()
    conn.close()
    feedback_list = [{'id': f[0], 'name': f[1], 'feedback': f[2], 'created_at': datetime.strptime(f[3], '%Y-%m-%d %H:%M:%S.%f').strftime('%d/%m/%Y')} for f in feedbacks]
    return render_template('view_feedbacks.html', feedbacks=feedback_list)

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute") # Chống Brute-force mật khẩu
def login():
    data = request.get_json()
    username, password = data.get('username'), data.get('password')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('SELECT password FROM users WHERE username = ?', (username,))
    user_record = cursor.fetchone()
    conn.close()
    if user_record and check_password_hash(user_record[0], password):
        session['logged_in'] = True
        session['username'] = username
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Tên đăng nhập hoặc mật khẩu không đúng'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/predict', methods=['GET', 'POST'])
@limiter.limit("10 per minute") # Giới hạn request AI để tránh treo server
@nocache
def predict_page():
    if not session.get('logged_in'): return redirect('/')
    if request.method == 'GET':
        report_dict = None
        try:
            conn = sqlite3.connect('analysis_reports.db')
            conn.row_factory = sqlite3.Row 
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM reports ORDER BY created_at DESC LIMIT 1')
            report_data = cursor.fetchone()
            conn.close()
            if report_data:
                report_dict = dict(report_data)
                for key in ['improvement_suggestions', 'strengths_to_promote']:
                    if report_dict.get(key):
                        try: report_dict[key] = json.loads(report_dict[key])
                        except: report_dict[key] = {}
        except Exception: pass
        return render_template('predict.html', latest_report=report_dict)
    
    elif request.method == 'POST':
        # Phân tích nhanh 1 câu (Dùng global model)
        replacements = load_replacements('./data/acronym.txt')
        input_text = request.form.get('text', '')
        corrected_text = replace_text(input_text, replacements)
        sentiment_type, sentiment_probs = predict_single_text(corrected_text)
        sentiment_score = sentiment_probs.get(sentiment_type, 0)
        return jsonify({'processed_text': corrected_text, 'sentiment_score': sentiment_score, 'sentiment_type': sentiment_type, 'sentiment_probabilities': sentiment_probs})

@app.route('/get_reviews', methods=['GET', 'POST'])
@nocache
def get_reviews():
    if not session.get('logged_in'): return jsonify({'error': 'Unauthorized'}), 401

    if request.method == 'POST':
        try:
            reviews_data = get_all_google_maps_reviews()
            if reviews_data is None: reviews_data = {'place_name': 'Địa điểm không xác định', 'reviews': []}
            place_name = reviews_data.get('place_name')
            google_reviews = reviews_data.get('reviews', [])
            
            try:
                conn = sqlite3.connect('feedback.db')
                cursor = conn.cursor()
                cursor.execute('SELECT name, feedback, created_at FROM feedbacks ORDER BY created_at DESC')
                db_feedbacks = [{'author': f[0], 'text': f[1], 'rating': 0, 'rtime': datetime.strptime(f[2], '%Y-%m-%d %H:%M:%S.%f').strftime('%d/%m/%Y')} for f in cursor.fetchall()]
                conn.close()
            except: db_feedbacks = []
            
            tiktok_comments = asyncio.run(get_tiktok_comments())
            tiktok_reviews = [{'author': c['user'], 'text': c['comment'], 'rating': 0, 'rtime': c['create_time']} for c in tiktok_comments[1:]]

            all_reviews = db_feedbacks + google_reviews + tiktok_reviews
            replacements = load_replacements('./data/acronym.txt')
            
            # --- TIỀN XỬ LÝ & CHUẨN BỊ BATCH ---
            texts_to_predict = []
            valid_review_indices = []

            for i, review in enumerate(all_reviews):
                review_text = review.get('text', '')
                if review_text and len(review_text.split()) >= 2:
                    corrected_text = replace_text(review_text, replacements)
                    if corrected_text and len(corrected_text.split()) >= 2:
                        texts_to_predict.append(corrected_text)
                        valid_review_indices.append(i)
                        all_reviews[i]['processed_text'] = corrected_text
            
            analyzed_reviews = []
            positive_count = 0
            neutral_count = 0
            negative_count = 0

            # --- BATCH PREDICTION ---
            if texts_to_predict:
                id2sentiment = {0: 'Tiêu cực', 1: 'Trung tính', 2: 'Tích cực'}
                total_samples = len(texts_to_predict)
                for i in range(0, total_samples, BATCH_SIZE):
                    batch_texts = texts_to_predict[i : i + BATCH_SIZE]
                    batch_tensors = global_vocab.corpus_to_tensor(batch_texts)
                    text_lengths = torch.LongTensor([len(t) for t in batch_tensors])
                    padded_input = pad_sequence(batch_tensors, padding_value=global_vocab["<pad>"]).to(device)

                    with torch.no_grad():
                        outputs = global_model(padded_input, text_lengths)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        preds = np.argmax(probs, axis=1)
                    
                    for j, pred_idx in enumerate(preds):
                        original_idx = valid_review_indices[i + j]
                        review_obj = all_reviews[original_idx]
                        
                        sentiment_type = id2sentiment[pred_idx]
                        sentiment_score = float(probs[j][pred_idx])
                        
                        if sentiment_type == 'Tiêu cực': negative_count += 1
                        elif sentiment_type == 'Trung tính': neutral_count += 1
                        else: positive_count += 1
                        
                        analyzed_reviews.append({
                            'author': review_obj.get('author', 'Ẩn danh'),
                            'rating': review_obj.get('rating', 0),
                            'text': review_obj.get('text'),
                            'processed_text': review_obj.get('processed_text'),
                            'sentiment_score': sentiment_score,
                            'sentiment_type': sentiment_type,
                            'rtime': review_obj.get('rtime', 'Không xác định')
                        })

            if not analyzed_reviews:
                return jsonify({'error': 'Không tìm thấy bình luận nào có thể phân tích', 'place_name': place_name}), 404
            
            total_reviews_count = len(analyzed_reviews)
            counts = {'Tích cực': positive_count, 'Trung tính': neutral_count, 'Tiêu cực': negative_count}
            avg_sentiment_type = max(counts, key=counts.get) if total_reviews_count > 0 else 'Không xác định'

            # Xử lý text
            positive_text = " ".join([r['processed_text'] for r in analyzed_reviews if r['sentiment_type'] == 'Tích cực'])
            neutral_text = " ".join([r['processed_text'] for r in analyzed_reviews if r['sentiment_type'] == 'Trung tính'])
            negative_text = " ".join([r['processed_text'] for r in analyzed_reviews if r['sentiment_type'] == 'Tiêu cực'])

            improvement_suggestions = analyze_improvement_topics([r for r in analyzed_reviews if r['sentiment_type'] == 'Tiêu cực'])
            strengths_to_promote = analyze_strengths([r for r in analyzed_reviews if r['sentiment_type'] == 'Tích cực'])
            
            # Wordcloud
            positive_wordcloud = generate_wordcloud(positive_text)
            neutral_wordcloud = generate_wordcloud(neutral_text)
            negative_wordcloud = generate_wordcloud(negative_text)

            # Lấy danh sách từ (có gạch dưới) để lọc stopwords và vẽ Venn
            p_words = [w for w in preprocess_text_for_charts(positive_text).split() if w not in stopwords]
            n_words = [w for w in preprocess_text_for_charts(neutral_text).split() if w not in stopwords]
            neg_words = [w for w in preprocess_text_for_charts(negative_text).split() if w not in stopwords]

            # Hàm hỗ trợ tách từ ghép thành từ đơn (bỏ gạch dưới) để N-gram chỉ có 2 từ đơn
            def flatten_tokens(tokens):
                flat = []
                for t in tokens:
                    # Thay thế _ bằng khoảng trắng rồi tách ra
                    flat.extend(t.replace('_', ' ').split())
                return flat

            # Tạo list từ đơn cho biểu đồ N-gram
            p_words_flat = flatten_tokens(p_words)
            n_words_flat = flatten_tokens(n_words)
            neg_words_flat = flatten_tokens(neg_words)
            
            # Tạo biểu đồ dùng list từ đơn (kết quả sẽ là "từ1 từ2" thay vì "từ1_từ2 từ3")
            positive_bigram_chart = generate_ngram_chart(dict(Counter(list(ngrams(p_words_flat, 2))).most_common(10)), 'Tích cực')
            neutral_bigram_chart = generate_ngram_chart(dict(Counter(list(ngrams(n_words_flat, 2))).most_common(10)), 'Trung tính')
            negative_bigram_chart = generate_ngram_chart(dict(Counter(list(ngrams(neg_words_flat, 2))).most_common(10)), 'Tiêu cực')
            
            # Venn diagram dùng p_words gốc (từ ghép) để hiển thị ý nghĩa đầy đủ hơn
            venn_diagram = generate_venn_diagram(p_words, n_words, neg_words)

            try:
                conn = sqlite3.connect('analysis_reports.db')
                cursor = conn.cursor()
                cursor.execute('''INSERT INTO reports (place_name, total_reviews, positive_count, neutral_count, negative_count, avg_sentiment_type, positive_wordcloud, neutral_wordcloud, negative_wordcloud, positive_bigram_chart, neutral_bigram_chart, negative_bigram_chart, venn_diagram, improvement_suggestions, strengths_to_promote, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (place_name, total_reviews_count, positive_count, neutral_count, negative_count, avg_sentiment_type, positive_wordcloud, neutral_wordcloud, negative_wordcloud, positive_bigram_chart, neutral_bigram_chart, negative_bigram_chart, venn_diagram, json.dumps(improvement_suggestions, ensure_ascii=False), json.dumps(strengths_to_promote, ensure_ascii=False), datetime.now()))
                conn.commit()
                conn.close()
            except Exception as e: print(f"Lỗi lưu báo cáo: {e}")

            return jsonify({
                'positive_wordcloud': positive_wordcloud, 'neutral_wordcloud': neutral_wordcloud, 'negative_wordcloud': negative_wordcloud,
                'positive_bigram_chart': positive_bigram_chart, 'neutral_bigram_chart': neutral_bigram_chart, 'negative_bigram_chart': negative_bigram_chart,
                'venn_diagram': venn_diagram, 'place_name': place_name, 'reviews': analyzed_reviews,
                'avg_sentiment_type': avg_sentiment_type, 'total_reviews': total_reviews_count,
                'positive_count': positive_count, 'neutral_count': neutral_count, 'negative_count': negative_count,
                'improvement_suggestions': improvement_suggestions, 'strengths_to_promote': strengths_to_promote
            })

        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({'error': f"Lỗi server: {str(e)}", 'place_name': 'Địa điểm không xác định'}), 500
    return render_template('reviews.html')

@app.route('/analysis_history')
@nocache
def analysis_history():
    if not session.get('logged_in'): return redirect('/')
    conn = sqlite3.connect('analysis_reports.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, place_name, total_reviews, avg_sentiment_type, created_at FROM reports ORDER BY created_at DESC')
    reports = [{'id': r[0], 'place_name': r[1], 'total_reviews': r[2], 'avg_sentiment_type': r[3], 'created_at': datetime.strptime(r[4], '%Y-%m-%d %H:%M:%S.%f').strftime('%H:%M - %d/%m/%Y')} for r in cursor.fetchall()]
    conn.close()
    return render_template('analysis_history.html', reports=reports)

@app.route('/view_report/<int:report_id>')
@nocache
def view_report(report_id):
    if not session.get('logged_in'): return redirect('/')
    conn = sqlite3.connect('analysis_reports.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reports WHERE id = ?', (report_id,))
    data = cursor.fetchone()
    conn.close()
    if data:
        report = dict(data)
        report['created_at'] = datetime.strptime(report['created_at'], '%Y-%m-%d %H:%M:%S.%f').strftime('%H:%M - %d/%m/%Y')
        for k in ['improvement_suggestions', 'strengths_to_promote']:
            if report.get(k):
                try: report[k] = json.loads(report[k])
                except: report[k] = None
        return render_template('view_report.html', report=report)
    return "Không tìm thấy báo cáo", 404

@app.route('/delete_report/<int:report_id>', methods=['POST'])
def delete_report(report_id):
    if not session.get('logged_in'): return jsonify({'status': 'error'}), 401
    try:
        conn = sqlite3.connect('analysis_reports.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM reports WHERE id = ?', (report_id,))
        conn.commit()
        rows = cursor.rowcount
        conn.close()
        return jsonify({'status': 'success', 'message': 'Đã xóa báo cáo'}) if rows > 0 else jsonify({'status': 'error', 'message': 'Không tìm thấy'}), 404
    except Exception as e: return jsonify({'status': 'error', 'message': str(e)}), 500

# --- KHỞI TẠO ---
init_database()
init_analysis_reports_database()
init_users_database()
init_model()

if __name__ == "__main__":
    app.run(debug=True)