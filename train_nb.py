import torch
import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def prepare_data_for_sklearn(dataset):
    """
    Hàm phụ trợ để chuyển đổi dữ liệu từ đối tượng IMDBDataset 
    sang định dạng list of strings.
    """
    texts = [" ".join(tokens) for tokens in dataset.tokenized_reviews]
    labels = dataset.tensor_label.numpy()
    return texts, labels

def main(config_fpath):
    config = utils.get_config(config_fpath)
    
    # Tạo thư mục logs
    log_dir = "logs/NaiveBayes"
    utils.create_dir(log_dir)
    print(f"Logs directory: {log_dir}")

    # --- Tải và Tiền xử lý dữ liệu ---
    vocab, train_dataset, valid_dataset, test_dataset = utils.preprocess_and_load_data(config, log_dir)

    print("\nĐang chuẩn bị dữ liệu cho Naive Bayes...")
    X_train_text, y_train = prepare_data_for_sklearn(train_dataset)
    X_test_text, y_test = prepare_data_for_sklearn(test_dataset)
    
    print(f"Kích thước tập huấn luyện: {len(X_train_text)}")
    
    # --- Vector hóa văn bản ---
    print("\nVector hóa văn bản...")
    
    bow_vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(), 
        token_pattern=None,            
        max_features=5000,
        ngram_range=(1, 1),
        binary=True
    )

    X_train_bow = bow_vectorizer.fit_transform(X_train_text)
    X_test_bow = bow_vectorizer.transform(X_test_text)

    print(f"Kích thước ma trận đặc trưng: {X_train_bow.shape}")

    # --- Huấn luyện mô hình ---
    print("\nHuấn luyện mô hình Naive Bayes...")
    
    nb_classifier = BernoulliNB(alpha=3.0, binarize=0.0)

    nb_classifier.fit(X_train_bow, y_train)
    print("Huấn luyện hoàn tất.")

    # --- Đánh giá mô hình ---
    print("\n----------------------------------------------------")
    print("Đánh giá mô hình trên tập kiểm tra...")
    
    y_pred = nb_classifier.predict(X_test_bow)

    accuracy = accuracy_score(y_test, y_pred)
    target_names = ['negative', 'neutral', 'positive']
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
    print("\nBáo cáo phân loại:")
    print(report)

    # Vẽ và lưu ma trận nhầm lẫn
    plt.figure(figsize=(8, 6))
    utils.plot_confusion_matrix(cm, 
                                classes=target_names, 
                                title='Confusion Matrix - Naive Bayes')
    
    save_path = f"{log_dir}/confusion_matrix.png"
    plt.savefig(save_path)
    print(f"\nĐã lưu ma trận nhầm lẫn vào: {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline")
    parser.add_argument("--config", default="configs/config.yml", help="path to config file", dest="config_fpath")
    args = parser.parse_args()
    main(**vars(args))