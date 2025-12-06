import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils
import numpy as np
from RNN_vanilla import RNN_vanilla
import os

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
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            reviews, reviews_lengths = batch["reviews"]
            reviews = reviews.to(device)
            
            predictions = model(reviews, reviews_lengths)
          
            sentiments = batch["sentiments"].to(device)
            loss = criterion(predictions, sentiments)  
            
            acc = utils.multiclass_accuracy(predictions, sentiments)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            all_preds.append(predictions.cpu())
            all_labels.append(sentiments.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    precision, f1, cm = utils.calculate_metrics(all_preds, all_labels)
        
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader), precision, f1, cm

def get_vanilla_rnn_model(config, vocabulary, embedding_matrix):
    """
    Hàm riêng để tạo mô hình RNN_vanilla với embedding matrix đã tạo.
    """
    embedding_dim = config["model"]["embedding_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    n_layers = config["model"]["n_layers"]
    bidirectional = config["model"]["bidirectional"]
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
    
    # --- SỬA LỖI Ở ĐÂY: Dùng key 'embedding_fpath' trong 'model' ---
    word2vec_path = config["model"]["embedding_fpath"] 
    # ---------------------------------------------------------------

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
    optimizer = optim.AdamW(model.parameters(), weight_decay=1e-4)

    print("Creating Learning Rate Scheduler (CosineAnnealingLR)...")
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    # --- 5. Class Weighting ---
    # train_dataset.tensor_label chứa toàn bộ nhãn của tập train
    train_labels = train_dataset.tensor_label.numpy()
    class_counts = np.bincount(train_labels)
    
    # Tính trọng số nghịch đảo tần suất
    class_weights = 1. / (class_counts + 1e-6)
    # Chuẩn hóa để tổng trọng số ~ số lớp (3)
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
        valid_loss, valid_acc, valid_precision, valid_f1, _ = evaluate(model, valid_dataloader, criterion, device)
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
    # Tải mô hình tốt nhất đã được lưu
    if os.path.exists(best_model_path):
        model = torch.load(best_model_path)
    else:
        print("Không tìm thấy model tốt nhất, sử dụng model hiện tại.")

    test_loss, test_acc, test_precision, test_f1, test_cm = evaluate(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
    print(f"Test Precision: {test_precision:.3f} | Test F1-score: {test_f1:.3f}")

    print("\nConfusion Matrix:")
    print(test_cm)

    # Vẽ và lưu ma trận nhầm lẫn
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    utils.plot_confusion_matrix(test_cm, classes=['negative', 'neutral', 'positive'], title='Confusion matrix')
    plt.savefig(f"{current_log_dir}/confusion_matrix.png")
    print(f"Confusion matrix saved to {current_log_dir}/confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vietnamese Sentiment Analysis model (Vanilla RNN)")
    parser.add_argument("--config", default="configs/config.yml", help="path to config file", dest="config_fpath")
    args = parser.parse_args()
    main(**vars(args))