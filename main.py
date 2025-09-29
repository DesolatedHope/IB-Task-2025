"""
Updated main pipeline:
1) Run your existing pipeline (attributes -> stage2_load_contract -> stage3_classify_clauses -> outputs)
   - this uses WorldClassContractClassifier from retrieval.py
2) Feed its CSV output to classifier_module.StandardContractSearcher
3) Produce final 'classifier_results.csv' with Ollama-powered comparisons
"""

from extract import TextExtractor, flat_entries_to_df
from retrieval import WorldClassContractClassifier, extract_best_attributes
from variables import TN_STANDARD_TEMPLATE
from classifier import StandardContractSearcher
import os

# -------------- Stage A: existing retrieval pipeline --------------
print("=== STAGE A: RUN  RETRIEVAL PIPELINE ===")
extractor = TextExtractor(["en"])

# Extract standard template (this is used later by the classifier, but running extractor here also warms caches)
print("Extracting standard template (for preview / debug)...")
_ = extractor.pdf_to_json(TN_STANDARD_TEMPLATE, output_format="flat")

# Run retrieval pipeline
wc = WorldClassContractClassifier()
wc.stage1_load_attributes("Attribute Dictionary.xlsx")

# Load a specific contract (change filename if needed)
flat = extractor.pdf_to_json("TN_Contract5_Redacted", output_format="flat")
contract_df = flat_entries_to_df(flat["entries"])

wc.stage2_load_contract(contract_df)     # accepts DataFrame
wc.stage3_classify_clauses(top_k=5)

# Stage 5 generate outputs (uses your retrieval.run_complete_analysis style)
csv_file, json_file, world_results_df = wc.stage5_generate_outputs()
print("World-class pipeline produced:", csv_file, json_file)

# Optionally post-process best attributes
best_results = extract_best_attributes(
    input_file=csv_file,
    output_file=csv_file
)
print("✅ Best attributes extracted and saved!")

# -------------- Stage B: run the new classifier that ties retrieval -> standard DB -> Ollama --------------
print("=== STAGE B: RUN NEW CLASSIFIER (FUZZY + SEMANTIC + OLLAMA) ===")
scs = StandardContractSearcher(
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    ollama_model="llama3.2:1b"   # change if you want to use a different Ollama model
)

# build standard index from the standard pdf (TN_STANDARD_TEMPLATE is from variables.py)
scs.build_standard_index(TN_STANDARD_TEMPLATE)

# feed the retrieval CSV produced by your earlier pipeline
retrieval_csv = csv_file
final_output_csv = "final_classifier_results.csv"

final_df = scs.run(
    retrieval_csv=retrieval_csv,
    output_csv=final_output_csv,
    standard_pdf_name=TN_STANDARD_TEMPLATE,
    top_k=3,
    alpha=0.6,
)

print(f"Final classifier output saved to {final_output_csv}")
print(final_df.head(10).to_string(index=False))
