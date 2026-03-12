import json
import pandas as pd

def clean_text(text):
    return str(text).strip()

def main():
    print("🚀 BẮT ĐẦU CHẤM ĐIỂM HỆ THỐNG (ABLATION STUDY)")
    
    # 1. Tải đáp án gốc (Ground Truth) từ tập Test
    with open("data/test_processed.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    test_targets = [item.get("target", 0) for item in test_data]
    total_queries = len(test_targets)
    print(f"Đã nạp {total_queries} câu truy vấn từ tập Test.")

    # 2. Tải Kho dữ liệu Train để tra cứu nhãn của ứng viên
    with open("data/train_processed.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
        
    # Tạo từ điển map giữa "Mã nguồn" và "Nhãn lỗ hổng" của tập Train
    # Lưu ý: Vì hiện tại "func" đang bị trùng lặp nhiều (đều là public void bad()), 
    # cách map này có thể hơi mỏng, nhưng đủ xài cho bước test nhanh này.
    train_label_dict = {}
    for item in train_data:
        code_str = clean_text(item.get("func", ""))
        train_label_dict[code_str] = item.get("target", 0)

    # 3. Tải kết quả do AI (genexample.py) vừa tìm ra
    try:
        df_sim = pd.read_csv("sim_code.csv", header=None)
        retrieved_codes = df_sim[0].fillna("").tolist()
    except FileNotFoundError:
        print("❌ LỖI: Không tìm thấy file sim_code.csv!")
        return

    if len(retrieved_codes) != total_queries:
        print("❌ LỖI: Số lượng kết quả tìm kiếm không khớp với số câu hỏi!")
        return

    # 4. Chấm điểm Top-1 Accuracy
    correct_hits = 0
    misses = []

    for i in range(total_queries):
        expected_label = test_targets[i]
        
        # Tra cứu xem đoạn code mà hệ thống tìm về mang nhãn gì
        candidate_code = clean_text(retrieved_codes[i])
        actual_label = train_label_dict.get(candidate_code, -1) 
        
        if expected_label == actual_label:
            correct_hits += 1
        else:
            misses.append({
                "query_index": i,
                "expected": expected_label,
                "got": actual_label,
                "code_found": candidate_code[:50] + "..." # In một chút code để dễ nhìn
            })

    # 5. Báo cáo kết quả
    accuracy = (correct_hits / total_queries) * 100
    
    print("\n=========================================")
    print("📊 KẾT QUẢ ĐÁNH GIÁ (TOP-1 HIT RATE)")
    print("=========================================")
    print(f"Tổng số mẫu test:   {total_queries}")
    print(f"Số lần bắn trúng:   {correct_hits}")
    print(f"Số lần trượt:       {total_queries - correct_hits}")
    print(f"TỶ LỆ CHÍNH XÁC:    {accuracy:.2f}%")
    print("=========================================")
    
    if len(misses) > 0:
        print("\n⚠️ 5 Ca tìm kiếm thất bại đầu tiên để phân tích:")
        for m in misses[:5]:
            print(f"- Mẫu #{m['query_index']}: Cần nhãn {m['expected']} nhưng lại lấy về mã có nhãn {m['got']} -> {m['code_found']}")

if __name__ == "__main__":
    main()