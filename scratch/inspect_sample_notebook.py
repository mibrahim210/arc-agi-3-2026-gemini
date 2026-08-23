import json
import glob

notebook_files = glob.glob("reference/stochastic-goose/*.ipynb")
print("Found notebook files:", notebook_files)

for nb_path in notebook_files:
    print(f"\n==================== {nb_path} ====================")
    with open(nb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, cell in enumerate(data.get("cells", [])):
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))
        print(f"\n--- Cell {i} ({cell_type}) ---")
        print(source[:1000])
