import pickle
import faiss
import torch
import numpy as np
import Levenshtein
# from nlgeval import compute_metrics
from tqdm import tqdm

# from transformers import RobertaTokenizer, RobertaModel
# from transformers import AutoTokenizer, T5EncoderModel
from transformers import T5EncoderModel, RobertaTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer as RobertaTokenizerBase
import pandas as pd
from huggingface_hub import hf_hub_download
import json

from bert_whitening import sents_to_vecs, transform_and_normalize

# Custom tokenizer wrapper để bypass lỗi extra_special_tokens
class CodeT5Tokenizer:
    def __init__(self, model_name="Salesforce/codet5-base"):
        # Download vocab và merges files
        vocab_file = hf_hub_download(repo_id=model_name, filename="vocab.json")
        merges_file = hf_hub_download(repo_id=model_name, filename="merges.txt")
        
        # Khởi tạo tokenizer trực tiếp từ vocab files, bỏ qua config
        self.tokenizer = RobertaTokenizerBase(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors='replace',
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            add_prefix_space=False
        )
    
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

# Tạo tokenizer
print("Loading tokenizer...")
tokenizer = CodeT5Tokenizer()
print("Tokenizer loaded successfully!")

dim = 768

df = pd.read_csv("data/train_function_clean.csv", header=None)
train_code_list = df[0].tolist()
df = pd.read_csv("data/train_ast_clean.csv", header=None)
train_ast_list = df[0].tolist()

df = pd.read_csv("data/test_function_clean.csv", header=None)
test_code_list = df[0].tolist()
df = pd.read_csv("data/test_ast_clean.csv", header=None)
test_ast_list = df[0].tolist()

# df = pd.read_csv("data/train_function_clean.csv", header=None)
# train_code_list = df[0].tolist()
# df = pd.read_csv("data/train_ast_clean.csv", header=None)
# train_ast_list = df[0].tolist()

# df = pd.read_csv("data/test_function_clean.csv", header=None)
# test_code_list = df[0].tolist()
# df = pd.read_csv("output_ast.csv", header=None)
# test_ast_list = df[0].tolist()

# tokenizer = RobertaTokenizer.from_pretrained("model/codet5-base")
# model = RobertaModel.from_pretrained("model/codet5-base")
# tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
# model = RobertaModel.from_pretrained("Salesforce/codet5-base")

# Tải model T5Encoder (chỉ lấy phần Encoder để trích xuất đặc trưng)
print("Loading model...")
model = T5EncoderModel.from_pretrained("Salesforce/codet5-base")
print("Model loaded successfully!")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)

def sim_jaccard(s1, s2):
    """jaccard相似度"""
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)
    ret2 = s1.union(s2)
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

class Retrieval(object):
    def __init__(self):
        # Skip loading pre-computed vectors - will compute on-the-fly
        # f = open('model/code_vector.pkl', 'rb')
        # self.bert_vec = pickle.load(f)
        # f.close()
        # f = open('model/a.pkl', 'rb')
        # self.kernel = pickle.load(f)
        # f.close()
        # f = open('model/b.pkl', 'rb')
        # self.bias = pickle.load(f)
        # f.close()
        
        self.bert_vec = None
        self.kernel = None
        self.bias = None

        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None

    def encode_file(self):
        print(f"Encoding {len(train_code_list)} training examples...")
        all_texts = []
        all_ids = []
        all_vecs = []
        
        # Compute vectors for all training data
        batch_size = 32
        for i in range(0, len(train_code_list), batch_size):
            batch_codes = train_code_list[i:i+batch_size]
            batch_vecs = sents_to_vecs(batch_codes, tokenizer, model)
            for j, vec in enumerate(batch_vecs):
                all_texts.append(train_code_list[i+j])
                all_ids.append(i+j)
                all_vecs.append(vec.reshape(1,-1))
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i}/{len(train_code_list)} examples...")
        
        all_vecs = np.concatenate(all_vecs, 0)
        
        # Normalize vectors (skip whitening for simplicity)
        print("Normalizing vectors...")
        all_vecs = transform_and_normalize(all_vecs)
        
        id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
        self.id2text = id2text
        self.vecs = np.array(all_vecs, dtype="float32")
        self.ids = np.array(all_ids, dtype="int64")

    def build_index(self, n_list):
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, min(n_list, self.vecs.shape[0]))
        index.train(self.vecs)
        index.add_with_ids(self.vecs, self.ids)
        self.index = index

    def single_query(self, code, ast, topK):
        body = sents_to_vecs([code], tokenizer, model)
        body = transform_and_normalize(body)  # Skip whitening
        vec = body[[0]].reshape(1, -1).astype('float32')
        _, sim_idx = self.index.search(vec, topK)
        sim_idx = sim_idx[0].tolist()
        max_score = 0
        max_idx = 0
        code_score_list = []
        ast_score_list = []
        for j in sim_idx:
            code_score = sim_jaccard(train_code_list[j].split(), code.split())
            ast_score = Levenshtein.seqratio(str(train_ast_list[j]).split(), str(ast).split())
            code_score_list.append(code_score)
            ast_score_list.append(ast_score)
        for i in range(len(sim_idx)):
            code_score = code_score_list[i]
            ast_score = ast_score_list[i]
            score = 0.7*code_score + 0.3*ast_score
            if score > max_score:
                max_score = score
                max_idx = sim_idx[i]
        return train_code_list[max_idx], train_ast_list[max_idx]

if __name__ == '__main__':
    ccgir = Retrieval()
    print("Sentences to vectors")
    ccgir.encode_file()
    print("加载索引")
    ccgir.build_index(n_list=1)
    ccgir.index.nprobe = 1
    sim_code_list, sim_ast_list = [], []
    data_list = []
    for i in tqdm(range(len(test_code_list))):
        sim_code, sim_ast = ccgir.single_query(test_code_list[i], test_ast_list[i], topK=5)
        sim_code_list.append(sim_code)
        sim_ast_list.append(sim_ast)

    df = pd.DataFrame(sim_code_list)
    df.to_csv("sim_code.csv", index=False, header=None)
    df = pd.DataFrame(sim_ast_list)
    df.to_csv("sim_ast.csv", index=False, header=None)

    # metrics_dict = compute_metrics(hypothesis='sim.csv',references=['nl.csv'],no_skipthoughts=True, no_glove=True)
