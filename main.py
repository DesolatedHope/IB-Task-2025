"""
Updated main pipeline:
1) Run your existing pipeline (attributes -> stage2_load_contract -> stage3_classify_clauses -> outputs)
   - this uses WorldClassContractClassifier from retrieval.py
2) Feed its CSV output to classifier_module.StandardContractSearcher
3) Produce final 'classifier_results.csv' with Ollama-powered comparisons
"""

from extract import TextExtractor, flat_entries_to_df
from retrieval import WorldClassContractClassifier, extract_best_attributes
from variables import TN_STANDARD_TEMPLATE, TNContract, WA_STANDARD_TEMPLATE, WAContract
from classifier import StandardContractSearcher
import os

wc = WorldClassContractClassifier()
wc.stage1_load_attributes("Attribute Dictionary.xlsx")


print("=== STAGE A: RUN  RETRIEVAL PIPELINE ===")
extractor = TextExtractor(["en"])

print(f"Extracting standard template (for {TN_STANDARD_TEMPLATE.value} preview / debug)...")
_ = extractor.pdf_to_json(TN_STANDARD_TEMPLATE, output_format="flat")

for contract in TNContract:
    flat = extractor.pdf_to_json(contract.value, output_format="flat")
    contract_df = flat_entries_to_df(flat["entries"])

    wc.stage2_load_contract(contract_df)
    wc.stage3_classify_clauses(top_k=5)

    csv_file, json_file, world_results_df = wc.stage5_generate_outputs()

    best_results = extract_best_attributes(
        input_file=csv_file,
        output_file=csv_file
    )
    print("✅ Best attributes extracted and saved!")

    print("=== STAGE B: RUN NEW CLASSIFIER (FUZZY + SEMANTIC + OLLAMA) ===")
    scs = StandardContractSearcher(
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        ollama_model="llama3.2:1b" 
    )

    scs.build_standard_index(TN_STANDARD_TEMPLATE)

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


print(f"Extracting standard template (for {WA_STANDARD_TEMPLATE.value} preview / debug)...")
_ = extractor.pdf_to_json(WA_STANDARD_TEMPLATE, output_format="flat")

for contract in WAContract:
    flat = extractor.pdf_to_json(contract.value, output_format="flat")
    contract_df = flat_entries_to_df(flat["entries"])

    wc.stage2_load_contract(contract_df)
    wc.stage3_classify_clauses(top_k=5)

    csv_file, json_file, world_results_df = wc.stage5_generate_outputs()

    best_results = extract_best_attributes(
        input_file=csv_file,
        output_file=csv_file
    )
    print("✅ Best attributes extracted and saved!")

    print("=== STAGE B: RUN NEW CLASSIFIER (FUZZY + SEMANTIC + OLLAMA) ===")
    scs = StandardContractSearcher(
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        ollama_model="llama3.2:1b"   # change if you want to use a different Ollama model
    )

    scs.build_standard_index(TN_STANDARD_TEMPLATE)

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
