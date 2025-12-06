import torch
import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def prepare_data_for_svm(dataset):
    """
    Hàm phụ trợ để chuyển đổi dữ liệu từ đối tượng IMDBDataset 
    sang định dạng list of strings mà Scikit-learn có thể hiểu được.
    """
    # dataset.tokenized_reviews là một list các list token (vd: [['phim', 'hay'], ...])
    # Ta nối các token lại bằng khoảng trắng để CountVectorizer xử lý
    texts = [" ".join(tokens) for tokens in dataset.tokenized_reviews]
    
    # Lấy nhãn dưới dạng numpy array
    labels = dataset.tensor_label.numpy()
    
    return texts, labels

def main(config_fpath):
    """
    Hàm chính để huấn luyện và đánh giá mô hình SVM với Túi từ (Bag-of-Words).
    """
    # --- Tải cấu hình và chuẩn bị thư mục ---
    config = utils.get_config(config_fpath)
    
    # Tạo thư mục logs
    log_dir = "logs/SVM"
    utils.create_dir(log_dir)
    print(f"Logs directory: {log_dir}")
    # -----------------------

    # --- Tải và Tiền xử lý dữ liệu (Luồng mới từ utils.py) ---
    # Hàm này đã thực hiện: Load JSON -> Normalize -> Tokenize -> Split -> Build Vocab
    # Trả về các đối tượng IMDBDataset
    vocab, train_dataset, valid_dataset, test_dataset = utils.preprocess_and_load_data(config, log_dir)

    print("\nĐang chuẩn bị dữ liệu cho SVM...")
    # Chuyển đổi dữ liệu từ IMDBDataset sang dạng text raw (đã được tokenize và nối chuỗi)
    X_train_text, y_train = prepare_data_for_svm(train_dataset)
    X_test_text, y_test = prepare_data_for_svm(test_dataset)
    
    print(f"Kích thước tập huấn luyện: {len(X_train_text)} mẫu")
    print(f"Kích thước tập kiểm tra: {len(X_test_text)} mẫu")
    
    # In thử một mẫu dữ liệu để kiểm tra
    print(f"\nMẫu dữ liệu huấn luyện (sau khi xử lý): \nText: {X_train_text[0]}\nLabel: {y_train[0]}")

    # --- Vector hóa văn bản với Túi từ (BoW) ---
    print("\nVector hóa văn bản với Túi từ (BoW)...")
    
    # Khởi tạo CountVectorizer
    # Lưu ý: Dữ liệu vào đã được tách từ (bởi underthesea trong utils), 
    # các từ ghép nối bằng '_'. Ta chỉ cần split bằng khoảng trắng đơn giản.
    bow_vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(), # Tách theo khoảng trắng
        token_pattern=None,            # Tắt warning mặc định của sklearn
        max_features=10000,            # Giới hạn số từ vựng (top frequent)
        ngram_range=(1, 2)             # Sử dụng unigram và bigram
    )

    # Học từ vựng từ tập train và transform
    X_train_bow = bow_vectorizer.fit_transform(X_train_text)

    # Chỉ transform tập test (dùng từ vựng đã học từ train)
    X_test_bow = bow_vectorizer.transform(X_test_text)

    print(f"Kích thước ma trận BoW của tập huấn luyện: {X_train_bow.shape}")
    print(f"Kích thước ma trận BoW của tập kiểm tra: {X_test_bow.shape}")

    # --- Huấn luyện mô hình SVM ---
    print("\nHuấn luyện mô hình LinearSVC (SVM)...")
    # Khởi tạo mô hình SVM tuyến tính
    svm_classifier = LinearSVC(C=1.0, random_state=42, max_iter=3000, dual='auto')

    # Huấn luyện mô hình
    svm_classifier.fit(X_train_bow, y_train)
    print("Huấn luyện hoàn tất.")

    # --- Đánh giá mô hình ---
    print("\n----------------------------------------------------")
    print("Đánh giá mô hình trên tập kiểm tra...")
    
    # Dự đoán nhãn cho tập kiểm tra
    y_pred = svm_classifier.predict(X_test_bow)

    # Tính toán các chỉ số
    accuracy = accuracy_score(y_test, y_pred)
    
    # Mapping label id sang tên
    target_names = ['negative', 'neutral', 'positive']
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
    
    print("\nBáo cáo phân loại (Classification Report):")
    print(report)

    # Vẽ và lưu ma trận nhầm lẫn
    plt.figure(figsize=(8, 6))
    utils.plot_confusion_matrix(cm, 
                                classes=target_names, 
                                title='Confusion Matrix - SVM with BoW')
    
    save_path = f"{log_dir}/confusion_matrix_svm_bow.png"
    plt.savefig(save_path)
    print(f"\nĐã lưu ma trận nhầm lẫn vào: {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate SVM model with Bag-of-Words (BoW)")
    parser.add_argument("--config", default="configs/config.yml", help="path to config file", dest="config_fpath")
    args = parser.parse_args()
    main(**vars(args))