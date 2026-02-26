import torch
import numpy as np

def sents_to_vecs(sents, tokenizer, model, max_len=512):
    """
    Chuyển đổi danh sách các đoạn code/văn bản thành vector sử dụng CodeT5/Roberta.
    """
    # Xác định device dựa trên model
    device = next(model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(sents, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    
    # Chuyển input sang device (GPU/CPU)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Đưa qua model để lấy hidden states
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        
    # --- ĐOẠN QUAN TRỌNG CẦN KIỂM TRA ---
    # CodeT5 (T5Encoder) trả về BaseModelOutput, không có pooler_output mặc định
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        vecs = outputs.pooler_output.cpu().numpy()
    else:
        # Lấy trung bình cộng (Mean Pooling) các token để làm vector đại diện
        hidden_states = outputs.last_hidden_state
        # Chú ý: Cần tính trung bình có trọng số dựa trên attention_mask để không tính padding
        # Cách đơn giản:
        vecs = torch.mean(hidden_states, dim=1).cpu().numpy()
        
    return vecs

def transform_and_normalize(vecs, kernel, bias):
    """
    Áp dụng biến đổi Whitening (nếu có kernel/bias) và chuẩn hóa vector.
    Công thức: y = (x + bias).dot(kernel)
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    
    # Chuẩn hóa về độ dài đơn vị (Normalize to unit length)
    # Tránh chia cho 0
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / (norms + 1e-8)

def compute_whitening(vecs):
    """
    vecs: numpy array (N, D)
    Trả về:
        kernel (D, D)
        bias (1, D)
    """
    mu = np.mean(vecs, axis=0, keepdims=True)
    X = vecs - mu

    cov = np.dot(X.T, X) / X.shape[0]

    U, S, Vt = np.linalg.svd(cov)

    # Tránh chia cho 0
    k = 256
    W = np.dot(U[:, :k], np.diag(1.0 / np.sqrt(S[:k] + 1e-12)))

    return W, -mu