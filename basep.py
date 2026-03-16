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
# CONFIG (GIỮ NGUYÊN)
# ======================
MODEL = "gemini-2.5-flash"

API_KEYS = os.getenv("GEMINI_API_KEYS")

if not API_KEYS:
    raise ValueError("❌ GEMINI_API_KEYS not found in .env")

API_KEYS = [k.strip() for k in API_KEYS.split(",")]

current_key_index = 0

def create_client():
    global current_key_index
    return genai.Client(api_key=API_KEYS[current_key_index])

client = create_client()

REQUESTS_PER_MIN = 12
MIN_INTERVAL = 60 / REQUESTS_PER_MIN
BATCH_SIZE = 3
last_call = 0


# ======================
# RATE LIMIT (GIỮ NGUYÊN)
# ======================
def rate_limit():
    global last_call
    elapsed = time.time() - last_call
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    last_call = time.time()


# ======================
# SWITCH API (GIỮ NGUYÊN)
# ======================
def switch_api():
    global current_key_index, client

    current_key_index = (current_key_index + 1) % len(API_KEYS)

    print(f"🔁 Switching to API key #{current_key_index}")

    client = create_client()


# ======================
# UTILS (GIỮ NGUYÊN)
# ======================
def clean_text(text):
    return " ".join(str(text).split())


def extract_labels(text, expected_count):

    matches = re.findall(r"\b(0|1)\b", text)

    extracted = (
        [int(x) for x in matches[-expected_count:]]
        if len(matches) >= expected_count
        else [int(x) for x in matches]
    )

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
# CALL GEMINI (GIỮ NGUYÊN)
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
# PROMPT BUILDER (BASELINE)
# ======================
def build_prompt(batch_targets, batch_examples, batch_labels):

    prompt = f"""You are an elite Java security expert detecting OS Command Injection (CWE-78).

Analyze the following {len(batch_targets)} Java functions.

For EACH function determine whether it is vulnerable.

Output ONLY exactly {len(batch_targets)} digits (0 or 1) separated by spaces.

1 = Vulnerable
0 = Non-vulnerable

"""

    for i in range(len(batch_targets)):

        target = batch_targets[i]

        prompt += f"""--- CASE {i+1} ---
Code:
{target['func'][:1000]}

"""

    prompt += f"""Output format example for {len(batch_targets)} cases:
1 0 1"""

    return prompt


# ======================
# MAIN (GIỮ NGUYÊN)
# ======================
def main():

    print("🚀 BASELINE + Gemini Batch Pipeline")

    with open("data/test_processed.json") as f:
        test_data = json.load(f)

    with open("data/train_processed.json") as f:
        train_data = json.load(f)

    df_sim = pd.read_csv("sim_code.csv", header=None)
    retrieved_codes = df_sim[0].fillna("").tolist()

    train_label_dict = {clean_text(x["func"]): x["target"] for x in train_data}

    csv_file = "baseline_predictions.csv"

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Query_Index", "LLM_Response", "Prediction", "GroundTruth"]
        )

    predictions = []
    labels = []

    for i in tqdm(range(0, len(test_data), BATCH_SIZE), desc="Đang quét Batch"):

        batch_targets = test_data[i : i + BATCH_SIZE]
        batch_examples = retrieved_codes[i : i + BATCH_SIZE]
        batch_labels = [
            train_label_dict.get(clean_text(ex), 0) for ex in batch_examples
        ]

        prompt = build_prompt(batch_targets, batch_examples, batch_labels)

        max_retries = 3
        preds = []
        raw_resp = ""

        for attempt in range(max_retries):

            try:

                raw_resp = call_gemini(prompt)

                preds = extract_labels(raw_resp, len(batch_targets))

                break

            except Exception as e:

                if "RESOURCE_EXHAUSTED" in str(e):

                    print("⚠️ Quota exceeded → switching API key")

                    switch_api()

                    time.sleep(1)

                    continue

        if len(preds) < len(batch_targets):

            preds += [2] * (len(batch_targets) - len(preds))

        for j, item in enumerate(batch_targets):

            pred = preds[j]

            truth = item["target"]

            predictions.append(pred)

            labels.append(truth)

            with open(csv_file, "a", newline="", encoding="utf-8") as f:

                writer = csv.writer(f)

                writer.writerow(
                    [i + j, raw_resp.replace("\n", " "), pred, truth]
                )

    acc, prec, rec, f1 = calculate_metrics(predictions, labels)

    print("\n" + "=" * 40)

    print("🏆 BASELINE RESULTS")

    print("=" * 40)

    print(f"Accuracy  : {acc * 100:.2f}%")

    print(f"Precision : {prec * 100:.2f}%")

    print(f"Recall    : {rec * 100:.2f}%")

    print(f"F1 Score  : {f1 * 100:.2f}%")


if __name__ == "__main__":
    main()