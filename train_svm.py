import torch
import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from itertools import cycle

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

def plot_multiclass_roc(y_test, y_score, n_classes, target_names, save_path):
    """
    Hàm vẽ đường cong ROC cho bài toán phân loại đa lớp (One-vs-Rest)
    """
    # Binarize labels
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Tính FPR, TPR, AUC cho từng lớp
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Tính micro-average ROC curve và ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 6))
    lw = 2
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(target_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - SVM')
    plt.legend(loc="lower right")
    
    plt.savefig(save_path)
    print(f"Đã lưu biểu đồ ROC vào: {save_path}")
    plt.close()

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
    # Cấu hình cơ bản cho mô hình
    bow_vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split(), # Tách theo khoảng trắng
        token_pattern=None,            # Tắt warning mặc định của sklearn
        max_features=2000,             # Giới hạn số lượng từ vựng phổ biến nhất
        ngram_range=(1, 1)             # Chỉ sử dụng unigram
    )

    # Học từ vựng từ tập train và transform
    X_train_bow = bow_vectorizer.fit_transform(X_train_text)

    # Chỉ transform tập test (dùng từ vựng đã học từ train)
    X_test_bow = bow_vectorizer.transform(X_test_text)

    print(f"Kích thước ma trận BoW của tập huấn luyện: {X_train_bow.shape}")
    print(f"Kích thước ma trận BoW của tập kiểm tra: {X_test_bow.shape}")

    # --- Huấn luyện mô hình SVM ---
    print("\nHuấn luyện mô hình LinearSVC (SVM)...")
    # Khởi tạo mô hình SVM tuyến tính với các tham số chuẩn hóa cơ bản
    svm_classifier = LinearSVC(C=0.1, random_state=42, max_iter=3000, dual='auto')

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

    # 1. Vẽ và lưu ma trận nhầm lẫn
    plt.figure(figsize=(8, 6))
    utils.plot_confusion_matrix(cm, 
                                classes=target_names, 
                                title='Confusion Matrix - SVM with BoW')
    
    cm_save_path = f"{log_dir}/confusion_matrix_svm_bow.png"
    plt.savefig(cm_save_path)
    print(f"\nĐã lưu ma trận nhầm lẫn vào: {cm_save_path}")
    plt.close()

    # 2. Vẽ và lưu đường cong ROC
    # Lấy điểm số quyết định (decision function scores) thay vì nhãn dự đoán
    # LinearSVC trả về khoảng cách đến siêu phẳng phân cách
    y_score = svm_classifier.decision_function(X_test_bow)
    
    roc_save_path = f"{log_dir}/roc_curve_svm_bow.png"
    plot_multiclass_roc(y_test, y_score, n_classes=3, target_names=target_names, save_path=roc_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate SVM model with Bag-of-Words (BoW)")
    parser.add_argument("--config", default="configs/config.yml", help="path to config file", dest="config_fpath")
    args = parser.parse_args()
    main(**vars(args))