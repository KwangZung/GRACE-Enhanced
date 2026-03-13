from google import genai
from google.genai import types
import pandas as pd
from tqdm import tqdm
import json
import csv
import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

# ======================
# CONFIG
# ======================
MODEL = "gemini-2.5-flash"
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=API_KEY)

REQUESTS_PER_MIN = 12  # An toàn với free tier (limit 15/phút)
MIN_INTERVAL = 60 / REQUESTS_PER_MIN
BATCH_SIZE = 3
last_call = 0

# ======================
# RATE LIMIT
# ======================
def rate_limit():
    global last_call
    elapsed = time.time() - last_call
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    last_call = time.time()

# ======================
# UTILS
# ======================
def clean_text(text):
    return " ".join(str(text).split())

def extract_labels(text, expected_count):
    """Lọc lấy đúng N con số 0 hoặc 1 cuối cùng để chống nhiễu"""
    matches = re.findall(r"\b(0|1)\b", text)
    # Lấy các con số từ dưới lên (vì kết luận thường nằm ở cuối)
    extracted = [int(x) for x in matches[-expected_count:]] if len(matches) >= expected_count else [int(x) for x in matches]
    
    # Bù số 2 (Lỗi) nếu AI trả thiếu
    while len(extracted) < expected_count:
        extracted.append(2)
    return extracted

def calculate_metrics(predictions, ground_truth):
    tp = sum(1 for p, t in zip(predictions, ground_truth) if p == t == 1)
    tn = sum(1 for p, t in zip(predictions, ground_truth) if p == t == 0)
    fp = sum(1 for p, t in zip(predictions, ground_truth) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(predictions, ground_truth) if p == 0 and t == 1)

    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

# ======================
# CALL GEMINI
# ======================
def call_gemini(prompt):
    rate_limit()
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0
        )
    )
    return response.text

# ======================
# PROMPT BUILDER (Fixed RAG Logic)
# ======================
def build_prompt(batch_targets, batch_examples, batch_labels):
    prompt = f"""You are an elite Java security expert detecting OS Command Injection (CWE-78).
I will give you {len(batch_targets)} separate cases to analyze. 
For EACH case, you have a Reference Example (with its true label) and a Target Function.

"""
    for i in range(len(batch_targets)):
        target = batch_targets[i]
        label_txt = "Vulnerable (1)" if batch_labels[i] == 1 else "Non-vulnerable (0)"
        
        prompt += f"""--- CASE {i+1} ---
[Reference Example]
Code: {batch_examples[i][:1000]}
Label: {batch_labels[i]} ({label_txt})

[Target to Analyze]
Code: {target['func'][:1000]}
Nodes: {target['node'][:400]}
Edges: {target['edge'][:400]}

"""
    prompt += f"""CRITICAL INSTRUCTION: Analyze data flows based on references. 
Output ONLY exactly {len(batch_targets)} digits (0 or 1) separated by spaces.
Example format for 3 cases: 1 0 1"""
    return prompt

# ======================
# MAIN
# ======================
def main():
    print("🚀 GRACE + Gemini Batch-Optimized Pipeline")

    with open("data/test_processed.json") as f:
        test_data = json.load(f)
    with open("data/train_processed.json") as f:
        train_data = json.load(f)
    
    df_sim = pd.read_csv("sim_code.csv", header=None)
    retrieved_codes = df_sim[0].fillna("").tolist()

    train_label_dict = {clean_text(x["func"]): x["target"] for x in train_data}

    # Setup file lưu kết quả
    csv_file = 'grace_batch_predictions.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Query_Index', 'LLM_Response', 'Prediction', 'GroundTruth'])

    predictions = []
    labels = []

    for i in tqdm(range(0, len(test_data), BATCH_SIZE), desc="Đang quét Batch"):
        batch_targets = test_data[i : i+BATCH_SIZE]
        batch_examples = retrieved_codes[i : i+BATCH_SIZE]
        batch_labels = [train_label_dict.get(clean_text(ex), 0) for ex in batch_examples]

        prompt = build_prompt(batch_targets, batch_examples, batch_labels)

        # Bọc giáp Try-Except chống crash
        max_retries = 3
        preds = []
        raw_resp = ""
        
        for attempt in range(max_retries):
            try:
                raw_resp = call_gemini(prompt)
                preds = extract_labels(raw_resp, len(batch_targets))
                break
            except Exception as e:
                print(f"\n⚠️ Lỗi ở batch {i}: {e}")
                if attempt == max_retries - 1:
                    raw_resp = f"Error: {e}"
                    preds = [2] * len(batch_targets) # Đánh rớt (nhãn 2) nếu thử 3 lần đều xịt
                time.sleep(5)

        # Lưu kết quả an toàn
        for j, item in enumerate(batch_targets):
            pred = preds[j]
            truth = item["target"]
            predictions.append(pred)
            labels.append(truth)
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([i + j, raw_resp.replace('\n', ' '), pred, truth])

    # Chấm điểm
    acc, prec, rec, f1 = calculate_metrics(predictions, labels)
    print("\n" + "="*40)
    print("🏆 BÁO CÁO KẾT QUẢ BATCH-RAG")
    print("="*40)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

if __name__ == "__main__":
    main()