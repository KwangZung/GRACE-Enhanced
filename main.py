import os
import json
import csv
import re
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import dspy

# Load environment variables
load_dotenv()

# ======================
# DSPy Setup Config
# ======================
API_KEYS = os.getenv("DEEPSEEK_API_KEYS")
if not API_KEYS:
    raise ValueError("❌ DEEPSEEK_API_KEYS not found in .env")

# Taking the first API key for DSPy configuration
api_key = API_KEYS.split(",")[0].strip()

# Configure the DSPy LM Client for DeepSeek
lm = dspy.LM(
    model="openai/deepseek-chat",
    api_key=api_key,
    api_base="https://api.deepseek.com",
    max_tokens=100,
    temperature=0.0
)
dspy.configure(lm=lm)

# ======================
# DSPy Signature & Module
# ======================
class DetectOSCommandInjection(dspy.Signature):
    """You are an elite Java security expert detecting OS Command Injection (CWE-78). Analyze the data flows based on references to determine if the target function is vulnerable."""
    
    reference_code: str = dspy.InputField(desc="Reference Java code snippet for comparison.")
    reference_label: str = dspy.InputField(desc="The vulnerability status of the reference code: 'Vulnerable (1)' or 'Non-vulnerable (0)'.")
    
    target_code: str = dspy.InputField(desc="The target Java function code to analyze.")
    target_nodes: str = dspy.InputField(desc="Graph nodes representing the control or data flow of the target function.")
    target_edges: str = dspy.InputField(desc="Graph edges representing relationships in the target function's structure.")
    
    prediction: str = dspy.OutputField(desc="Output strictly '1' if the target is Vulnerable, or '0' if Non-vulnerable.")

class VulnerabilityDetector(dspy.Module):
    def __init__(self):
        super().__init__()
        # Using Predict for straightforward reasoning, similar to the original naive prompt.
        # Alternatively, dspy.ChainOfThought(DetectOSCommandInjection) could be used to improve accuracy.
        self.predictor = dspy.Predict(DetectOSCommandInjection)

    def forward(self, reference_code, reference_label, target_code, target_nodes, target_edges):
        result = self.predictor(
            reference_code=reference_code,
            reference_label=reference_label,
            target_code=target_code,
            target_nodes=target_nodes,
            target_edges=target_edges
        )
        return result

# ======================
# UTILS
# ======================
def clean_text(text):
    return " ".join(str(text).split())

def extract_label(text):
    """Extracts exactly one 0 or 1 label from the text output"""
    matches = re.findall(r"\b(0|1)\b", str(text))
    if matches:
        return int(matches[-1])
    return 2 # Represents an error or unparsable model response

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
# MAIN RUNNER
# ======================
def main():
    print("🚀 GRACE + DSPy Optimized Pipeline")

    # 1. Setup output folder
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, 'grace_dspy_predictions.csv')

    # 2. Load Data
    with open("data/test_processed.json") as f:
        test_data = json.load(f)
    with open("data/train_processed.json") as f:
        train_data = json.load(f)
    
    df_sim = pd.read_csv("sim_code.csv", header=None)
    retrieved_codes = df_sim[0].fillna("").tolist()

    train_label_dict = {clean_text(x["func"]): x["target"] for x in train_data}

    # Setup CSV Writer
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Query_Index', 'DSPy_Prediction', 'Prediction_Parsed', 'GroundTruth'])

    detector = VulnerabilityDetector()
    
    predictions = []
    labels = []

    # 3. Predict per example
    for i in tqdm(range(len(test_data)), desc="Evaluating Targets in DSPy"):
        target = test_data[i]
        ref_code = retrieved_codes[i]
        
        # Look up reference label
        true_ref_label_val = train_label_dict.get(clean_text(ref_code), 0)
        ref_label_txt = f"Vulnerable (1)" if true_ref_label_val == 1 else f"Non-vulnerable (0)"

        # Prepare payload truncating inputs to avoid context limit bloat
        payload = {
            "reference_code": str(ref_code)[:1000],
            "reference_label": ref_label_txt,
            "target_code": str(target['func'])[:1000],
            "target_nodes": str(target['node'])[:400],
            "target_edges": str(target['edge'])[:400]
        }

        ground_truth = target["target"]
        labels.append(ground_truth)
        
        # DSPy Prediction execution
        try:
            # Invoking DSPy programmatic forward method
            prediction_res = detector(**payload).prediction
            parsed_pred = extract_label(prediction_res)
        except Exception as e:
            print(f"\n❌ Prediction failed at index {i}: {e}")
            prediction_res = "ERROR"
            parsed_pred = 2 # Error state
            
        predictions.append(parsed_pred)

        # Write incrementally to output
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([i, prediction_res.replace('\n', ' '), parsed_pred, ground_truth])

    # 4. Metrics Reporting
    acc, prec, rec, f1 = calculate_metrics(predictions, labels)
    
    print("\n" + "="*40)
    print("🏆 REPORT: GRACE DSPy BATCH PIPELINE")
    print("="*40)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print(f"\n📁 Benchmark output successfully exported to -> {csv_file}")

if __name__ == "__main__":
    main()
