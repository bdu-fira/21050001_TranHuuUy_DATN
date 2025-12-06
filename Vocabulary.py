from itertools import chain
from collections import Counter
import torch
from tqdm import tqdm
from underthesea import word_tokenize
import unicodedata
import re

class Vocabulary:
    """ The Vocabulary class is used to record words, which are used to convert 
        text to numbers and vice versa.
    """

    def __init__(self):
        self.word2id = dict()
        self.word2id['<pad>'] = 0   # Pad Token
        self.word2id['<unk>'] = 1   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id 
    
    def __len__(self):
        return len(self.word2id)

    def id2word_func(self, word_index):
        return self.id2word[word_index]

    def add(self, word):
        """ Add word to vocabulary """
        if word not in self:
            word_index = self.word2id[word] = len(self.word2id)
            self.id2word[word_index] = word
            return word_index
        else:
            return self[word]

    @staticmethod
    def load_acronyms(acronym_path):
        """Load acronyms from file into a dictionary"""
        acronym_dict = {}
        try:
            with open(acronym_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' not in line: continue
                    parts = line.strip().split(':')
                    if len(parts) < 2: continue
                    
                    replacement = parts[1].strip()
                    keys = parts[0].split(',')
                    
                    for key in keys:
                        k = key.strip().lower()
                        if k:
                            acronym_dict[k] = replacement
            print(f"Loaded {len(acronym_dict)} acronyms.")
            return acronym_dict
        except FileNotFoundError:
            print(f"Warning: Acronym file not found at {acronym_path}.")
            return {}

    @staticmethod
    def normalize_text(text, acronym_dict=None):
        """
        Pipeline: 
        1. Normalize Unicode (NFC)
        2. Lowercase
        3. Remove Special Chars & Emojis
        4. Replace Acronyms (Whole word matching)
        5. Word Tokenize
        """
        if not isinstance(text, str):
            return []

        # 1. Unicode Normalize & Lowercase
        text = unicodedata.normalize('NFC', text).lower()
        
        # 2. Remove Special Characters & Emojis
        # Giữ lại chữ cái (bao gồm tiếng Việt có dấu), số và khoảng trắng.
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()

        # 3. Replace Acronyms
        # Thực hiện thay thế trên chuỗi trước khi tokenize
        if acronym_dict:
            # Tạo danh sách các từ để xử lý nhanh hơn regex cho từng từ
            words_raw = text.split()
            words_replaced = [acronym_dict.get(w, w) for w in words_raw]
            text = " ".join(words_replaced)
            
        # 4. Word Tokenize (Underthesea)
        # Thay thế khoảng trắng trong từ ghép bằng _
        tokens = [word.replace(" ", "_") for word in word_tokenize(text)]
            
        return tokens

    @staticmethod
    def tokenize_corpus(corpus, acronym_path=None):
        """Split the documents of the corpus into words with normalization"""
        print("Tokenize and Normalize the corpus...")
        
        acronym_dict = None
        if acronym_path:
            acronym_dict = Vocabulary.load_acronyms(acronym_path)

        tokenized_corpus = list()
        for document in tqdm(corpus):
            tokenized_document = Vocabulary.normalize_text(document, acronym_dict)
            tokenized_corpus.append(tokenized_document)

        return tokenized_corpus

    def corpus_to_tensor(self, corpus, is_tokenized=False):
        if is_tokenized:
            tokenized_corpus = corpus
        else:
            tokenized_corpus = self.tokenize_corpus(corpus)
            
        indicies_corpus = list()
        for document in tqdm(tokenized_corpus):
            indicies_document = torch.tensor(list(map(lambda word: self[word], document)),
                                             dtype=torch.int64)
            indicies_corpus.append(indicies_document)

        return indicies_corpus

    def add_words_from_corpus(self, corpus):
        """ Build vocabulary from list of tokenized documents """
        print("Building vocabulary from corpus...")
        word_freq = Counter(chain(*corpus))
        
        # Sắp xếp theo tần suất giảm dần
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Total unique words found: {len(word_freq)}")
        
        for word, freq in sorted_words:
            self.add(word)
        print(f"Vocabulary size after building: {len(self)}")