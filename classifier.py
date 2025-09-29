"""
classifier_module.py
- Build standard clause index (extract -> flat_entries_to_df)
- For every clause in a retrieval CSV, run fuzzy + semantic search against standard clauses
- Compare best match(s) with Ollama structured outputs to provide label, confidence, reasoning and differences
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Literal

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer

from ollama import chat
from pydantic import BaseModel, Field, ValidationError

# reuse your extractor utilities
from extract import TextExtractor, flat_entries_to_df

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClassifierModule")


# -------- Pydantic schema for structured output --------
class ComparisonResult(BaseModel):
    label: Literal["STANDARD", "NON-STANDARD"]
    confidence: float
    reasoning: str
    differences: List[str] = Field(default_factory=list)


class StandardContractSearcher:
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        ollama_model: str = "llama3.1",
    ):
        """
        embed_model: SentenceTransformer model for semantic search
        ollama_model: Ollama model name (must be pulled locally, e.g., "llama3.1" or "mistral")
        """
        self.embed_model_name = embed_model
        self.ollama_model = ollama_model

        logger.info(f"Loading embedder: {embed_model} ...")
        self.embedder = SentenceTransformer(embed_model)

        # placeholders populated later
        self.standard_df: Optional[pd.DataFrame] = None
        self.standard_texts: List[str] = []
        self.standard_embeddings: Optional[np.ndarray] = None

    # -------- Standard index --------
    def build_standard_index(self, standard_pdf_name: str, output_format: str = "flat"):
        logger.info(f"Extracting standard document: {standard_pdf_name}.pdf")
        extractor = TextExtractor(["en"])
        flat = extractor.pdf_to_json(standard_pdf_name, output_format=output_format)
        df = flat_entries_to_df(flat["entries"])

        if "Content" not in df.columns:
            raise ValueError("Standard df missing 'Content' column after extraction.")

        self.standard_df = df.copy()
        self.standard_texts = df["Content"].fillna("").astype(str).tolist()

        logger.info(f"Encoding {len(self.standard_texts)} standard clauses with {self.embed_model_name} ...")
        embeddings = self.embedder.encode(
            self.standard_texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        self.standard_embeddings = embeddings / norms

        logger.info("Standard index built.")
        return self.standard_df

    # -------- Retrieval CSV loader --------
    def load_retrieval_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        text_col = None
        for candidate in ["Clause_Text", "Content", "RawText", "clause_text", "Clause_Text_Extracted"]:
            if candidate in df.columns:
                text_col = candidate
                break
        if text_col is None:
            text_cols = [c for c in df.columns if df[c].dtype == object]
            if not text_cols:
                raise ValueError("No text column detected in retrieval CSV.")
            text_col = text_cols[0]

        df = df.copy()
        df["clause_text"] = df[text_col].fillna("").astype(str)
        if "Clause_ID" not in df.columns:
            df["Clause_ID"] = df.index.astype(str)

        logger.info(f"Loaded retrieval CSV '{path}' with {len(df)} rows. Using text column '{text_col}'.")
        return df

    # -------- Search helpers --------
    def _semantic_search(self, clause_text: str, top_k: int = 5):
        emb = self.embedder.encode([clause_text], convert_to_numpy=True)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        sims = (self.standard_embeddings @ emb[0]).astype(float)
        top_idx = np.argsort(-sims)[:top_k]
        return [(int(idx), float(sims[idx])) for idx in top_idx]

    def _fuzzy_search(self, clause_text: str, top_k: int = 5):
        results = process.extract(
            clause_text, self.standard_texts, scorer=fuzz.token_sort_ratio, limit=top_k
        )
        return [(int(r[2]), float(r[1]) / 100.0) for r in results]

    def find_candidates(self, clause_text: str, top_k: int = 5, alpha: float = 0.6):
        sem = self._semantic_search(clause_text, top_k=top_k)
        fuz = self._fuzzy_search(clause_text, top_k=top_k)

        cand = {}
        for idx, s in sem:
            cand[idx] = {"sem": s, "fuz": 0.0}
        for idx, f in fuz:
            if idx in cand:
                cand[idx]["fuz"] = f
            else:
                cand[idx] = {"sem": 0.0, "fuz": f}

        cand_list = []
        for idx, scores in cand.items():
            combined = alpha * scores["sem"] + (1 - alpha) * scores["fuz"]
            cand_list.append(
                {
                    "idx": idx,
                    "text": self.standard_texts[idx],
                    "semantic": float(scores["sem"]),
                    "fuzzy": float(scores["fuz"]),
                    "combined": float(combined),
                }
            )
        cand_list.sort(key=lambda x: x["combined"], reverse=True)
        return cand_list[:top_k]

    # -------- Ollama comparator --------
    def compare_with_ollama(self, contract_text: str, standard_text: str) -> ComparisonResult:
        prompt = (
            "You are a legal contract comparator. Compare the contract clause and the standard clause.\n"
            """1. Exact Structural Match (Standard)
• Contract text matches the standard template in structure, phrasing, and intent.
• Placeholders (e.g., XX%, [Fee Schedule]) are replaced with actual values.
• Example:
o [(XX%)] → 100% ✅ Classify as Standard
2. Value Substitution (Still Standard)
• Percentage values, fee schedule names, or similar placeholders differ but follow the
same formula/intent.
• Example:
o Standard: “[(XX%)] of the Fee Schedule”
o Extracted: “95% of the Fee Schedule” ✅ Classify as Standard
3. Minor Wording Differences (Standard)
• Stylistic or language changes without altering meaning.
• Example:
o Standard: "in effect on the date of service"
o Extracted: "as in force at the time services are rendered" ✅ Classify as
Standard
4. Structural or Conditional Changes (Non-Standard)
• Additional conditions, carve-outs, or exceptions not in the standard.
• Reimbursement tied to something other than the specified Fee Schedule.
• Example:
o "Shall be 100% of the Fee Schedule except for cardiology services, which
will be 80%." ❌ Classify as Non-Standard
5. Reference to Different Methodologies (Non-Standard)
• ❌ Classify as Non-Standard
"""
            "Output must strictly follow the JSON schema with keys: label, confidence, reasoning, differences.\n\n"
            "Write label as STANDARD/NON-STANDARD only.\n"
            "Contract clause:\n---\n"
            f"{contract_text}\n---\n"
            "Standard clause:\n---\n"
            f"{standard_text}\n---"
        )
        schema = ComparisonResult.model_json_schema()

        try:
            resp = chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                format=schema,
                options={"temperature": 0},
            )
            content = resp.message.content
            comp = ComparisonResult.model_validate_json(content)
            return comp
        except (ValidationError, Exception) as e:
            sem_score = self._semantic_search(contract_text, top_k=1)[0][1]
            label = "MATCH" if sem_score > 0.82 else ("NON-STANDARD")
            return ComparisonResult(
                label=label,
                confidence=sem_score,
                reasoning=f"Fallback heuristic due to Ollama error: {repr(e)}",
                differences=[],
            )

    # -------- End-to-end runner --------
    def run(
        self,
        retrieval_csv: str,
        output_csv: str,
        standard_pdf_name: str,
        top_k: int = 3,
        alpha: float = 0.6,
    ) -> pd.DataFrame:
        if self.standard_df is None:
            self.build_standard_index(standard_pdf_name, output_format="flat")

        src_df = self.load_retrieval_csv(retrieval_csv)

        out_rows = []
        for i, row in src_df.iterrows():
            clause_text = str(row["clause_text"])
            clause_id = row.get("Clause_ID", str(i))

            candidates = self.find_candidates(clause_text, top_k=top_k, alpha=alpha)
            best = candidates[0] if candidates else None

            if best:
                std_idx = best["idx"]
                standard_text = best["text"]
                comp_res = self.compare_with_ollama(clause_text, standard_text)
            else:
                std_idx = None
                standard_text = ""
                comp_res = ComparisonResult(
                    label="UNMATCHED",
                    confidence=0.0,
                    reasoning="No candidate found",
                    differences=[],
                )

            out_rows.append(
                {
                    "Clause_ID": clause_id,
                    "Clause_Text": clause_text,
                    "Matched_Standard_Index": std_idx,
                    "Matched_Standard_Text": standard_text,
                    "Semantic_Score": best["semantic"] if best else 0.0,
                    "Fuzzy_Score": best["fuzzy"] if best else 0.0,
                    "Combined_Score": best["combined"] if best else 0.0,
                    "LLM_Label": comp_res.label,
                    "LLM_Confidence": comp_res.confidence,
                    "LLM_Reasoning": comp_res.reasoning,
                    "LLM_Differences": json.dumps(comp_res.differences),
                }
            )

        out_df = pd.DataFrame(out_rows)
        out_df.to_csv(output_csv, index=False)
        logger.info(f"Saved classifier output to {output_csv}")
        return out_df
