import os
import yaml
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torchtext.vocab as vocab
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from RNN import RNN
from Vocabulary import Vocabulary
from IMDBDataset import IMDBDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import json

def get_config(file_path):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_pretrained_word2vec(fpath):
    print(f"Loading Word2Vec from {fpath}...")
    # unk_init=torch.Tensor.normal_ để đảm bảo tính ngẫu nhiên ban đầu
    return vocab.Vectors(name=fpath, unk_init=torch.Tensor.normal_)

def create_embedding_matrix(vocabulary, word_embedding, embedding_dim):
    """
    Tạo embedding matrix:
    - Nếu từ có trong Word2Vec -> Lấy vector pre-trained
    - Nếu từ KHÔNG có (vd: teencode, từ lóng mới) -> Khởi tạo ngẫu nhiên (Learnable)
    """
    print("Creating embedding matrix...")
    matrix_len = len(vocabulary)
    weights_matrix = torch.zeros((matrix_len, embedding_dim))
    
    hits = 0
    misses = 0
    
    for i, word in enumerate(vocabulary.word2id):
        # Bỏ qua padding (đã là zeros)
        if word == "<pad>":
            continue
            
        if word in word_embedding.stoi:
            weights_matrix[i] = word_embedding[word]
            hits += 1
        else:
            # Khởi tạo ngẫu nhiên cho từ OOV
            weights_matrix[i] = torch.normal(mean=0, std=1, size=(embedding_dim,))
            misses += 1
            
    print(f"Embedding Matrix Created: {hits} hits (from W2V), {misses} misses (random init).")
    return weights_matrix

def preprocess_and_load_data(config, log_dir):
    """
    Quy trình chuẩn:
    1. Load JSON
    2. Normalize & Tokenize toàn bộ data
    3. Split indices (Train/Valid/Test)
    4. Build Vocab CHỈ từ tập Train
    5. Tạo Datasets
    """
    data_paths = config["dataset"]["data_paths"]
    acronym_path = "./Data/acronym.txt" 
    
    print("Loading and Preprocessing data...")
    all_reviews = []
    all_sentiments = []
    
    # 1. Load Raw Data
    for sentiment, fpath in data_paths.items():
        with open(fpath, 'r', encoding='utf-8') as f:
            reviews = json.load(f)
            all_reviews.extend(reviews)
            all_sentiments.extend([sentiment] * len(reviews))
            
    # 2. Normalize & Tokenize (Sử dụng hàm mới trong Vocabulary)
    tokenized_reviews = Vocabulary.tokenize_corpus(all_reviews, acronym_path=acronym_path)
    
    # 3. Split Indices (Stratified)
    labels = all_sentiments
    indices = np.arange(len(labels))
    split_rate = config["dataset"]["split_rate"]
    
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, train_size=split_rate, stratify=labels, random_state=42
    )
    valid_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # 4. Build Vocabulary FROM TRAIN ONLY
    print("Building Vocabulary from Train set only...")
    train_reviews_tokenized = [tokenized_reviews[i] for i in train_idx]
    
    vocabulary = Vocabulary()
    vocabulary.add_words_from_corpus(train_reviews_tokenized)
    
    if config["vocab"]["save"]:
        torch.save(vocabulary, f"{log_dir}/vocab.pt")
        
    # 5. Create Dataset Objects
    def create_subset(indices):
        sub_reviews = [tokenized_reviews[i] for i in indices]
        sub_labels = [all_sentiments[i] for i in indices]
        return IMDBDataset(vocabulary, preloaded_reviews=sub_reviews, preloaded_sentiments=sub_labels)

    train_dataset = create_subset(train_idx)
    valid_dataset = create_subset(valid_idx)
    test_dataset = create_subset(test_idx)
    
    print(f"Data split: Train={len(train_dataset)}, Valid={len(valid_dataset)}, Test={len(test_dataset)}")
    
    return vocabulary, train_dataset, valid_dataset, test_dataset

def get_model(config, vocabulary, embedding_matrix):
    embedding_dim = config["model"]["embedding_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    n_layers = config["model"]["n_layers"]
    bidirectional = config["model"]["bidirectional"]
    dropout = config["model"]["dropout"]
    output_dim = config["model"]["output_dim"]

    input_dim = len(vocabulary)
    pad_idx = vocabulary["<pad>"]

    model = RNN(input_dim,
                embedding_dim,
                hidden_dim,
                n_layers,
                bidirectional,
                dropout,
                pad_idx,
                output_dim)

    # Load embedding matrix đã tạo
    model.embedding.weight.data.copy_(embedding_matrix)
    
    # Đảm bảo padding vector luôn là 0
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
    
    return model

# --- Các hàm phụ trợ ---
def create_dir(dpath):
    if not os.path.exists(dpath):
        os.makedirs(dpath)

def create_current_log_dir(logs_dir):
    this_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    current_log_dir = f"{logs_dir}/{this_time}"
    create_dir(current_log_dir)
    state_dir = f"{current_log_dir}/state"
    create_dir(state_dir)
    return current_log_dir, state_dir

def create_logs_dir(config):
    logs_dir = config["train"]["logs_dir"] or "logs"
    create_dir(logs_dir)
    return create_current_log_dir(logs_dir)

def get_writer(log_dir):
    tensorboard_dpath = f"{log_dir}/tensorboard"
    create_dir(tensorboard_dpath)
    return SummaryWriter(log_dir=tensorboard_dpath)

def multiclass_accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    return correct.float() / y.shape[0]

def calculate_metrics(preds, y):
    preds = preds.argmax(1, keepdim=True)
    y = y.cpu().numpy()
    preds = preds.cpu().numpy()
    precision = precision_score(y, preds, average='weighted', zero_division=0)
    f1 = f1_score(y, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(y, preds)
    return precision, f1, cm

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    return int(elapsed_time / 60), int(elapsed_time % 60)