import os
import subprocess
import json
import pandas as pd
import shutil

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN TRAIN VÀ TEST
# ==========================================
CONFIG = {
    "train": {
        "input_dir": "data/java_train",       # Thư mục chứa code để huấn luyện/xây FAISS
        "out_json": "data/train_processed.json",
        "out_csv": "data/train_ast.csv"
    },
    "test": {
        "input_dir": "data/java_test",        # Thư mục chứa code dùng làm câu truy vấn (Query)
        "out_json": "data/test_processed.json",
        "out_csv": "data/test_ast.csv"
    }
}

JOERN_SCRIPT = 'batch_query.sc'

def create_and_run_joern(target_dir):
    script_content = r"""
    @main def exec() = {
        try {
            importCode.java("TARGET_DIR")
        } catch {
            case e: Exception => println("ERROR_IMPORTING: " + e.getMessage)
        }
        
        try {
            cpg.method.internal.filterNot(m => m.name == "<init>" || m.name == "<clinit>").foreach { method =>
                try {
                    val methodName = method.name
                    val code = method.code.replaceAll("\r\n|\r|\n", " ")
                    val astSeq = method.ast.map(_.label).l.mkString(" ")
                    val nodes = method.ast.map(n => "(" + n.id + ", " + n.label + ", " + n.code.replaceAll("\r\n|\r|\n", " ") + ")").l.mkString("; ")
                    
                    val edges = method.ast.outE.map(e => "(" + e.src.id + "->" + e.dst.id + ", " + e.label + ")").l.mkString("; ")
                    
                    println("RESULT_START|_SEP_|" + methodName + "|_SEP_|" + code + "|_SEP_|" + astSeq + "|_SEP_|" + nodes + "|_SEP_|" + edges + "|_SEP_|RESULT_END")
                } catch {
                    case e: Exception => // Bỏ qua lỗi cục bộ
                }
            }
        } catch {
            case e: Exception => println("ERROR_PROCESSING_CPG: " + e.getMessage)
        }
    }
    """.replace("TARGET_DIR", target_dir.replace("\\", "/"))
    
    with open(JOERN_SCRIPT, 'w', encoding='utf-8') as f:
        f.write(script_content)
        
    print(f"⏳ Đang khởi động Joern và phân tích thư mục: {target_dir} ...")
    
    result = subprocess.run(
        ['joern', '--script', JOERN_SCRIPT], 
        capture_output=True, 
        text=True, 
        encoding='utf-8'
    )
    return result.stdout, result.stderr

def process_directory(input_dir, out_json, out_csv):
    """Hàm xử lý cho từng thư mục độc lập"""
    if not os.path.exists(input_dir):
        print(f"❌ Lỗi: Không tìm thấy thư mục '{input_dir}'. Bỏ qua khâu này.")
        return

    stdout, stderr = create_and_run_joern(input_dir)
    
    processed_json_data = []
    ast_list = []
    
    for line in stdout.splitlines():
        if "RESULT_START" in line and "RESULT_END" in line:
            try:
                content = line.split("RESULT_START|_SEP_|")[1].split("|_SEP_|RESULT_END")[0]
                parts = content.split("|_SEP_|")
                
                if len(parts) >= 5:
                    method_name = parts[0]
                    func_code = parts[1]
                    ast_seq = parts[2]
                    nodes = parts[3]
                    edges = parts[4]
                    
                    target = 1 if "bad" in method_name.lower() else 0
                    
                    ast_list.append(ast_seq)
                    processed_json_data.append({
                        "func": func_code,
                        "target": target,
                        "node": nodes,
                        "edge": edges,
                        "example": ""
                    })
            except Exception as e:
                continue

    if len(processed_json_data) > 0:
        pd.DataFrame(ast_list).to_csv(out_csv, index=False, header=False)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(processed_json_data, f, indent=4)
            
        print(f"✅ THÀNH CÔNG! Đã trích xuất {len(processed_json_data)} hàm và lưu vào {out_json}")
    else:
        print(f"⚠️ Không có dữ liệu được trích xuất từ {input_dir}.")
        print("Chi tiết lỗi (nếu có):", stderr[:500]) # Chỉ in 500 ký tự đầu cho đỡ rối

def main():
    # Chạy vòng lặp qua cả tập Train và Test
    for split_name, paths in CONFIG.items():
        print(f"\n{'='*50}")
        print(f"🚀 BẮT ĐẦU XỬ LÝ TẬP DỮ LIỆU: {split_name.upper()}")
        print(f"{'='*50}")
        process_directory(paths["input_dir"], paths["out_json"], paths["out_csv"])

    # Dọn dẹp chiến trường sau khi hoàn tất cả 2 tập
    print("\n🧹 Đang dọn dẹp file tạm...")
    if os.path.exists(JOERN_SCRIPT): os.remove(JOERN_SCRIPT)
    if os.path.exists("workspace"): 
        shutil.rmtree("workspace", ignore_errors=True)
    print("🎉 Hoàn tất toàn bộ quy trình tiền xử lý!")

if __name__ == "__main__":
    main()