import os
import subprocess
import json
import pandas as pd

JAVA_DIR = 'data/java_test'
OUTPUT_JSON = 'data/java_processed.json'
OUTPUT_AST_CSV = 'data/java_ast.csv'
JOERN_SCRIPT = 'batch_query.sc'

def create_and_run_joern():
    target_dir = JAVA_DIR
    
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
                    
                    // SỬA LỖI Ở ĐÂY: Đổi inNode thành dst (đích) và outNode thành src (nguồn) cho tương thích Joern FlatGraph
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
        
    print(f"Đang khởi động Joern và phân tích thư mục {target_dir} ...")
    
    result = subprocess.run(
        ['joern', '--script', JOERN_SCRIPT], 
        capture_output=True, 
        text=True, 
        encoding='utf-8'
    )
    return result.stdout, result.stderr

def main():
    if not os.path.exists(JAVA_DIR):
        print(f"Lỗi: Không tìm thấy thư mục '{JAVA_DIR}'.")
        return

    stdout, stderr = create_and_run_joern()
    
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
        pd.DataFrame(ast_list).to_csv(OUTPUT_AST_CSV, index=False, header=False)
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(processed_json_data, f, indent=4)
            
        print(f"✅ THÀNH CÔNG! Đã trích xuất {len(processed_json_data)} hàm Java.")
    else:
        print("❌ LỖI TỪ JOERN. ĐÂY LÀ NGUYÊN NHÂN THỰC SỰ:")
        print("========= STDERR (Lỗi chi tiết) =========")
        print(stderr)
        print("========= STDOUT (Kết quả in ra) =========")
        print(stdout)
        print("=======================================")

    # Dọn dẹp file thừa
    if os.path.exists(JOERN_SCRIPT): os.remove(JOERN_SCRIPT)
    if os.path.exists("workspace"): 
        import shutil
        shutil.rmtree("workspace", ignore_errors=True)

if __name__ == "__main__":
    main()