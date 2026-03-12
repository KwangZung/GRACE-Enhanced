import json
import pandas as pd

TEST_JSON = "data/java_processed.json"
SIM_CODE = "sim_code.csv"
SIM_AST = "sim_ast.csv"

OUTPUT = "processed_for_llmpre.json"


def main():

    print("Loading data...")

    with open(TEST_JSON, "r", encoding="utf-8") as f:
        code_data = json.load(f)

    sim_code_df = pd.read_csv(SIM_CODE, header=None)
    sim_ast_df = pd.read_csv(SIM_AST, header=None)

    results = []

    total = len(code_data)

    for i in range(total):

        item = {}

        item["func"] = code_data[i]["func"]
        item["target"] = code_data[i]["target"]

        item["node"] = str(code_data[i]["node"])
        item["edge"] = str(code_data[i]["edge"])

        example_code = str(sim_code_df.iloc[i, 0])
        example_ast = str(sim_ast_df.iloc[i, 0])

        item["example"] = example_code + "\nAST:\n" + example_ast

        results.append(item)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Done.")
    print("Generated:", OUTPUT)
    print("Total samples:", len(results))


if __name__ == "__main__":
    main()