import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils
import numpy as np

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

def main(config_fpath):
    config = utils.get_config(config_fpath)
    current_log_dir, state_dir = utils.create_logs_dir(config)
    print(f"Log dir: {current_log_dir}")

    # 1. Load, Clean, Tokenize, Split & Build Vocab (Only from Train)
    vocab, train_dataset, valid_dataset, test_dataset = utils.preprocess_and_load_data(config, current_log_dir)

    # 2. Load Word2Vec & Create Embedding Matrix (Mix of Pretrained + Random for OOV)
    word_vectors = utils.get_pretrained_word2vec(config["model"]["embedding_fpath"])
    embedding_matrix = utils.create_embedding_matrix(vocab, word_vectors, config["model"]["embedding_dim"])

    # Config params
    batch_size = config["train"]["batch_size"]
    n_epochs = config["train"]["n_epochs"]
    save_epoch = config["train"]["save_epoch"]
    state_fpath = config["continue"]["state_fpath"]
    
    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    # Model Init
    model = utils.get_model(config, vocab, embedding_matrix)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Loss Function with Class Weights
    train_labels = train_dataset.tensor_label
    class_counts = torch.bincount(train_labels)
    class_weights = 1. / (class_counts.float() + 1e-6)
    class_weights = class_weights / class_weights.sum() * 3
    class_weights = class_weights.to(device)
    
    print(f"Class weights: {class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1).to(device)
    
    writer = utils.get_writer(log_dir=current_log_dir)

    # Training Loop
    best_valid_loss = float("inf")
    begin_epoch = 0
    patience = 5
    epochs_no_improve = 0
    best_model_path = f"{current_log_dir}/model.pt"

    if state_fpath is not None:
        checkpoint = torch.load(state_fpath)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        begin_epoch = checkpoint["epoch"] + 1
        print(f"Continue after epoch {begin_epoch}")

    print("Start Training...")
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
        
        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} | Acc: {valid_acc*100:.2f}% | F1: {valid_f1:.3f}")
        print(f"\t Current LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # --- LOGIC EARLY STOPPING ---
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
    print("Testing best model...")
    model = torch.load(best_model_path)
    test_loss, test_acc, test_precision, test_f1, test_cm = evaluate(model, test_dataloader, criterion, device)
    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% | Test F1: {test_f1:.3f}")
    
    print("\nConfusion Matrix:")
    print(test_cm)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    utils.plot_confusion_matrix(test_cm, classes=['negative', 'neutral', 'positive'], title='Confusion matrix')
    plt.savefig(f"{current_log_dir}/confusion_matrix.png")
    print(f"Confusion matrix saved to {current_log_dir}/confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yml", dest="config_fpath")
    args = parser.parse_args()
    main(**vars(args))