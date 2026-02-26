import faiss
import torch
import numpy as np
from tqdm import tqdm

from rank_bm25 import BM25Okapi
from transformers import T5EncoderModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer as RobertaTokenizerBase
import pandas as pd
from huggingface_hub import hf_hub_download
import json

from bert_whitening import sents_to_vecs, transform_and_normalize, compute_whitening

# ==========================================
# 1. SETUP TOKENIZER & MODEL
# ==========================================
class CodeT5Tokenizer:
    def __init__(self, model_name="Salesforce/codet5-base"):
        vocab_file = hf_hub_download(repo_id=model_name, filename="vocab.json")
        merges_file = hf_hub_download(repo_id=model_name, filename="merges.txt")
        self.tokenizer = RobertaTokenizerBase(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors='replace', bos_token="<s>", eos_token="</s>",
            sep_token="</s>", cls_token="<s>", unk_token="<unk>",
            pad_token="<pad>", mask_token="<mask>", add_prefix_space=False
        )
    def __call__(self, *args, **kwargs): return self.tokenizer(*args, **kwargs)
    def __getattr__(self, name): return getattr(self.tokenizer, name)

print("Loading tokenizer and model...")
tokenizer = CodeT5Tokenizer()
model = T5EncoderModel.from_pretrained("Salesforce/codet5-base")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)
dim = 768

# ==========================================
# 2. LOAD JAVA DATASET (Sanity Check)
# ==========================================
print("Loading Java dataset...")
# Đọc file AST
df_ast = pd.read_csv("data/java_ast.csv", header=None)
train_ast_list = df_ast[0].fillna("").astype(str).tolist()

# Đọc file Code từ JSON
with open("data/java_processed.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
    train_code_list = [item.get("func", "") for item in json_data]

# Tạm thời gán Test = Train để test luồng 75 hàm
test_code_list = train_code_list
test_ast_list = train_ast_list
print(f"Loaded {len(train_code_list)} functions.")

# ==========================================
# 3. HYBRID RETRIEVAL (DENSE + SPARSE + RRF)
# ==========================================
class Retrieval(object):
    def __init__(self):
        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None

    def encode_file(self):
        print(f"Encoding {len(train_code_list)} training examples...")
        all_texts, all_ids, all_vecs = [], [], []
        
        batch_size = 32
        for i in range(0, len(train_code_list), batch_size):
            batch_codes = train_code_list[i:i+batch_size]
            batch_vecs = sents_to_vecs(batch_codes, tokenizer, model)
            for j, vec in enumerate(batch_vecs):
                all_texts.append(train_code_list[i+j])
                all_ids.append(i+j)
                all_vecs.append(vec.reshape(1,-1))
        
        all_vecs = np.concatenate(all_vecs, 0)
        print("Normalizing vectors (Whitening)...")
        self.kernel, self.bias = compute_whitening(all_vecs)
        print("Compute Whitening")
        all_vecs = transform_and_normalize(all_vecs, self.kernel, self.bias)
        
        self.id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
        self.vecs = np.array(all_vecs, dtype="float32")
        self.ids = np.array(all_ids, dtype="int64")

    def build_index(self, n_list):
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, min(n_list, self.vecs.shape[0]))
        index.train(self.vecs)
        index.add_with_ids(self.vecs, self.ids)
        self.index = index

    def single_query(self, code, ast, topK=10):
        # 1. SEMANTIC (DENSE RETRIEVAL)
        body = sents_to_vecs([code], tokenizer, model)
        body = transform_and_normalize(body, self.kernel, self.bias)
        vec = body[[0]].reshape(1, -1).astype('float32')
        
        # FAISS trả về topK (mặc định lấy nhiều hơn 1 chút để RRF có không gian xếp hạng)
        # Giới hạn topK không vượt quá số lượng mẫu đang có
        actual_topK = min(topK, len(train_code_list))
        _, sim_idx_matrix = self.index.search(vec, actual_topK)
        sim_idx = sim_idx_matrix[0].tolist()
        
        # Tập ứng viên
        top_k_codes = [train_code_list[j].split() for j in sim_idx]
        top_k_asts = [train_ast_list[j].split() for j in sim_idx]
        
        # 2. LEXICAL (BM25 trên Code)
        bm25_lexical = BM25Okapi(top_k_codes)
        lexical_scores = bm25_lexical.get_scores(code.split())
        lexical_ranks = {j: rank + 1 for rank, (score, j) in enumerate(sorted(zip(lexical_scores, sim_idx), reverse=True))}
        
        # 3. SYNTACTIC (AST 3-Grams + BM25)
        def get_ngrams(sequence, n=3):
            if len(sequence) < n: return ["_".join(sequence)]
            return ["_".join(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
            
        query_ast_ngrams = get_ngrams(str(ast).split(), n=3)
        top_k_ast_ngrams = [get_ngrams(ast_seq, n=3) for ast_seq in top_k_asts]
        
        bm25_syntactic = BM25Okapi(top_k_ast_ngrams)
        syntactic_scores = bm25_syntactic.get_scores(query_ast_ngrams)
        syntactic_ranks = {j: rank + 1 for rank, (score, j) in enumerate(sorted(zip(syntactic_scores, sim_idx), reverse=True))}
        
        # 4. RRF FUSION
        k_rrf = 60
        best_j = sim_idx[0]
        best_rrf_score = -1
        
        for rank_semantic, j in enumerate(sim_idx):
            rank_sem = rank_semantic + 1
            rank_lex = lexical_ranks[j]
            rank_syn = syntactic_ranks[j]
            
            rrf_score = (1.0 / (k_rrf + rank_sem)) + (1.0 / (k_rrf + rank_lex)) + (1.0 / (k_rrf + rank_syn))
            
            if rrf_score > best_rrf_score:
                best_rrf_score = rrf_score
                best_j = j
                
        return train_code_list[best_j], train_ast_list[best_j]

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    ccgir = Retrieval()
    ccgir.encode_file()
    print("Building FAISS index...")
    ccgir.build_index(n_list=1)
    ccgir.index.nprobe = 1
    
    sim_code_list, sim_ast_list = [], []
    
    print("Starting Hybrid Retrieval process...")
    for i in tqdm(range(len(test_code_list))):
        sim_code, sim_ast = ccgir.single_query(test_code_list[i], test_ast_list[i], topK=10)
        sim_code_list.append(sim_code)
        sim_ast_list.append(sim_ast)

    # Xuất kết quả
    df_code = pd.DataFrame(sim_code_list)
    df_code.to_csv("sim_code.csv", index=False, header=None)
    
    df_ast_out = pd.DataFrame(sim_ast_list)
    df_ast_out.to_csv("sim_ast.csv", index=False, header=None)
    
    print("Xong! Đã lưu kết quả truy xuất vào sim_code.csv và sim_ast.csv")