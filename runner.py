"""
Runner script:
- Iterates through TN and WA contracts
- Runs retrieval pipeline
- Runs classifier pipeline with region-specific standard template
- Saves results under /result/ with per-contract CSVs
"""

import os
from extract import TextExtractor, flat_entries_to_df
from retrieval import WorldClassContractClassifier, extract_best_attributes
from classifier import StandardContractSearcher

# ------------------ Paths ------------------
BASE_DIR = "data"
CONTRACTS_DIR = os.path.join(BASE_DIR, "Contracts")
STANDARDS_DIR = os.path.join(BASE_DIR, "Standard Templates")
ATTR_FILE = "Attribute Dictionary.xlsx"

# Mapping regions → standard template
STANDARD_MAP = {
    "TN": os.path.join(STANDARDS_DIR, "TN_Standard_Template_Redacted"),
    "WA": os.path.join(STANDARDS_DIR, "WA_Standard_Redacted"),
}

RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

# ------------------ Retrieval ------------------
def run_retrieval(contract_path: str, extractor: TextExtractor) -> str:
    """Run retrieval pipeline and return CSV path"""
    contract_name = os.path.splitext(os.path.basename(contract_path))[0]
    print(f"\n=== Retrieval for {contract_name} ===")

    wc = WorldClassContractClassifier()
    wc.stage1_load_attributes(ATTR_FILE)

    flat = extractor.pdf_to_json(contract_path, output_format="flat")
    contract_df = flat_entries_to_df(flat["entries"])

    wc.stage2_load_contract(contract_df)
    wc.stage3_classify_clauses(top_k=5)

    csv_file, json_file, _ = wc.stage5_generate_outputs()

    # Post-process attributes
    _ = extract_best_attributes(input_file=csv_file, output_file=csv_file)

    return csv_file

# ------------------ Classifier ------------------
def run_classifier(retrieval_csv: str, contract_path: str, scs: StandardContractSearcher, standard_path: str):
    """Run semantic + Ollama classifier"""
    contract_name = os.path.splitext(os.path.basename(contract_path))[0]
    output_csv = os.path.join(RESULT_DIR, f"final_classifier_results_{contract_name}.csv")

    final_df = scs.run(
        retrieval_csv=retrieval_csv,
        output_csv=output_csv,
        standard_pdf_name=standard_path,
        top_k=3,
        alpha=0.6,
    )
    print(f"✅ Final results saved: {output_csv}")
    return final_df

# ------------------ Main Runner ------------------
if __name__ == "__main__":
    extractor = TextExtractor(["en"])

    for region in ["TN", "WA"]:
        standard_path = STANDARD_MAP[region]

        # Warm cache with standard template
        _ = extractor.pdf_to_json(standard_path, output_format="flat")

        # Init classifier with region's standard index
        scs = StandardContractSearcher(
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
            ollama_model="gemma3:1b"
        )
        scs.build_standard_index(standard_path)

        region_dir = os.path.join(CONTRACTS_DIR, region)
        for fname in sorted(os.listdir(region_dir)):
            if fname.endswith(".pdf"):
                contract_path = os.path.join(region_dir, fname)
                try:
                    retrieval_csv = run_retrieval(contract_path, extractor)
                    _ = run_classifier(retrieval_csv, contract_path, scs, standard_path)
                except Exception as e:
                    print(f"❌ Error on {fname}: {e}")
