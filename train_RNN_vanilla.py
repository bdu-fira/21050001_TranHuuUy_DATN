import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils
import numpy as np
from RNN_vanilla import RNN_vanilla
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle

def train(model, dataloader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in dataloader:
        optimizer.zero_grad()
        reviews, reviews_lengths = batch["reviews"]
        reviews = reviews.to(device)
        
        predictions = model(reviews, reviews_lengths)
        
        sentiments = batch["sentiments"].to(device)
        loss = criterion(predictions, sentiments)
        
        acc = utils.multiclass_accuracy(predictions, sentiments)
    
        loss.backward()    
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            reviews, reviews_lengths = batch["reviews"]
            reviews = reviews.to(device)
            
            # Logits từ model
            logits = model(reviews, reviews_lengths)
            
            # Tính xác suất (Softmax) cho ROC
            probs = F.softmax(logits, dim=1)
          
            sentiments = batch["sentiments"].to(device)
            loss = criterion(logits, sentiments)  
            
            acc = utils.multiclass_accuracy(logits, sentiments)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            all_preds.append(logits.cpu())
            all_labels.append(sentiments.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    precision, f1, cm = utils.calculate_metrics(all_preds, all_labels)
        
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader), precision, f1, cm, all_probs, all_labels

def plot_multiclass_roc(y_test, y_score, n_classes, target_names, save_path):
    """
    Hàm vẽ đường cong ROC cho bài toán phân loại đa lớp (One-vs-Rest)
    """
    # Binarize labels (One-hot encoding cho nhãn thực tế)
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
    plt.title('Receiver Operating Characteristic (ROC) - Vanilla RNN')
    plt.legend(loc="lower right")
    
    plt.savefig(save_path)
    print(f"Đã lưu biểu đồ ROC vào: {save_path}")
    plt.close()

def get_vanilla_rnn_model(config, vocabulary, embedding_matrix):
    """
    Hàm riêng để tạo mô hình RNN_vanilla với embedding matrix đã tạo.
    """
    embedding_dim = config["model"]["embedding_dim"]
    
    # Thiết lập các tham số để đảm bảo đây là mô hình 'Vanilla' cơ bản
    hidden_dim = 64
    n_layers = 1
    bidirectional = False
    
    dropout = config["model"]["dropout"]
    output_dim = config["model"]["output_dim"]

    input_dim = len(vocabulary)
    pad_idx = vocabulary["<pad>"]

    model = RNN_vanilla(input_dim,
                        embedding_dim,
                        hidden_dim,
                        n_layers,
                        bidirectional,
                        dropout,
                        pad_idx,
                        output_dim)

    # Load pretrained weights từ embedding_matrix vào model
    model.embedding.weight.data.copy_(embedding_matrix)
    
    # Đảm bảo padding vector luôn là 0
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
    
    return model

def main(config_fpath):
    config = utils.get_config(config_fpath)

    print("Create logs folder...")
    # Tạo thư mục log riêng cho mô hình Vanilla RNN
    config["train"]["logs_dir"] = "logs/RNN_vanilla" 
    current_log_dir, state_dir = utils.create_logs_dir(config)
    print(f"The current log dir is {current_log_dir}")

    # --- 1. Load Data & Build Vocab (New Workflow) ---
    vocab, train_dataset, valid_dataset, test_dataset = utils.preprocess_and_load_data(config, current_log_dir)
    
    # --- 2. Load Word2Vec & Create Embedding Matrix ---
    print("Loading Pretrained Word Embeddings...")
    
    word2vec_path = config["model"]["embedding_fpath"] 
    word2vec = utils.get_pretrained_word2vec(word2vec_path)
    
    embedding_dim = config["model"]["embedding_dim"]
    embedding_matrix = utils.create_embedding_matrix(vocab, word2vec, embedding_dim)

    # --- 3. Dataloaders ---
    batch_size = config["train"]["batch_size"]
    n_epochs = config["train"]["n_epochs"]
    save_epoch = config["train"]["save_epoch"]
    state_fpath = config["continue"]["state_fpath"]
    
    print("Creating dataloader...")
    # Sử dụng collate_fn từ chính dataset
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    # --- 4. Initialize Model ---
    print("Creating Vanilla RNN model...")
    model = get_vanilla_rnn_model(config, vocab, embedding_matrix)

    print("Creating optimizer and loss function...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sử dụng SGD
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    print("Creating Learning Rate Scheduler (CosineAnnealingLR)...")
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    # --- 5. Class Weighting ---
    train_labels = train_dataset.tensor_label.numpy()
    class_counts = np.bincount(train_labels)
    
    class_weights = 1. / (class_counts + 1e-6)
    class_weights = class_weights / np.sum(class_weights) * 3
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print(f"Sử dụng trọng số cho các lớp: {class_weights_tensor.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
    model = model.to(device)

    print("Creating SummaryWriter...")
    writer = utils.get_writer(log_dir=current_log_dir)

    print("Training...")
    best_valid_loss = float("inf")
    begin_epoch = 0
    
    # --- BIẾN CHO EARLY STOPPING ---
    patience = 5
    epochs_no_improve = 0
    best_model_path = f"{current_log_dir}/model.pt"
    # ----------------------------------------

    if state_fpath is not None:
        try:
            checkpoint = torch.load(state_fpath)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            begin_epoch = checkpoint["epoch"] + 1
            print(f"Continue after epoch {begin_epoch}")
        except Exception as e:
            print(f"Không thể load checkpoint: {e}. Bắt đầu train từ đầu.")

    for epoch in range(begin_epoch, n_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        valid_loss, valid_acc, valid_precision, valid_f1, _, _, _ = evaluate(model, valid_dataloader, criterion, device)
        end_time = time.time()

        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        
        writer.add_scalars("loss", {"valid": valid_loss, "train": train_loss}, epoch)
        writer.add_scalars("accuracy", {"valid": valid_acc, "train": train_acc}, epoch)
        writer.add_scalar("validation/precision", valid_precision, epoch)
        writer.add_scalar("validation/f1_score", valid_f1, epoch)

        
        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
        print(f"\t Val. Precision: {valid_precision:.3f} |  Val. F1-score: {valid_f1:.3f}")
        
        print(f"\t Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # LOGIC EARLY STOPPING
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, best_model_path)
            epochs_no_improve = 0
            print(f"\t -> New best model saved with validation loss: {best_valid_loss:.3f}")
        else:
            epochs_no_improve += 1
            print(f"\t -> Validation loss did not improve for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break 

        if ((epoch + 1) % save_epoch) == 0:
            print(f"Saving state at epoch {epoch+1}")
            state_fpath = f"{state_dir}/epoch{epoch+1}.pt"
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       state_fpath)
        
        scheduler.step()

    print("\n----------------------------------------------------")
    print("Testing with the best model...")
    if os.path.exists(best_model_path):
        model = torch.load(best_model_path)
    else:
        print("Không tìm thấy model tốt nhất, sử dụng model hiện tại.")

    # Đánh giá trên tập test, lấy thêm probs và labels để vẽ ROC
    test_loss, test_acc, test_precision, test_f1, test_cm, test_probs, test_labels = evaluate(model, test_dataloader, criterion, device)
    
    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
    print(f"Test Precision: {test_precision:.3f} | Test F1-score: {test_f1:.3f}")

    print("\nConfusion Matrix:")
    print(test_cm)

    # Mapping label
    target_names = ['negative', 'neutral', 'positive']

    # 1. Vẽ và lưu Confusion Matrix
    plt.figure(figsize=(8, 6))
    utils.plot_confusion_matrix(test_cm, classes=target_names, title='Confusion matrix - RNN Vanilla')
    plt.savefig(f"{current_log_dir}/confusion_matrix.png")
    print(f"Confusion matrix saved to {current_log_dir}/confusion_matrix.png")
    plt.close()

    # 2. Vẽ và lưu ROC Curve
    roc_save_path = f"{current_log_dir}/roc_curve.png"
    # test_probs là tensor xác suất (sau softmax), test_labels là tensor nhãn thực
    plot_multiclass_roc(test_labels.cpu().numpy(), test_probs.cpu().numpy(), 
                        n_classes=3, target_names=target_names, save_path=roc_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vietnamese Sentiment Analysis model (Vanilla RNN)")
    parser.add_argument("--config", default="configs/config.yml", help="path to config file", dest="config_fpath")
    args = parser.parse_args()
    main(**vars(args))