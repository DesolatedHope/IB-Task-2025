# Healthcare Contract Clause Extraction & Classification  

## 🏆 Competition Submission  

This project was developed as part of a challenge to build a system that can automatically process healthcare contracts and classify clauses into **Standard** or **Non-Standard** categories.  

The main goal was to **extract, align, and compare** clauses from masked healthcare contracts against a state-specific standard template. The difficulty comes from noisy contract data (scanned PDFs, OCR errors, non-uniform headings) and the fact that clauses may vary in wording while preserving or changing intent.  

Our solution combines **advanced text extraction**, **semantic understanding**, and **hybrid retrieval strategies** to achieve robust and accurate results.  

---

## ⚙️ Core Approach  

### 1️⃣ Contract Text Parsing & Clause Extraction  

Contracts are inconsistent in structure, so we designed a **multi-pattern extraction pipeline**:  

- **Heading Detection Tricks**  
  - Regex patterns for multiple formats:  
    - Numeric for clauses: `1.1`, `2.3.4`  
    - Generic style: `ARTICLE I`, `SECTION 2.1`  
  - This ensures we don’t miss clauses hidden under different formatting conventions.  

- **OCR Preprocessing** *(only when contracts are scanned PDFs)*  
  - **Denoising & Deskewing** → Improves OCR recognition accuracy
  - **Adaptive Thresholding** → Handles uneven lighting and background noise  
  - **EasyOCR + PyMuPDF + Pillow** → Combination proved most effective for noisy legal scans

- **Structured Extraction**  
  - Each clause is tagged with **metadata** (section, subsection, page number, contract position).  
  - This metadata helps later in aligning clauses across different contracts.  

---

### 2️⃣ Attribute-Based Clause Retrieval  

We were given an **Excel Attribute Dictionary** that defines 5 attributes to extract.  

To maximize coverage, we combined **three strategies** for mapping clauses to attributes:  

1. **Exact Keyword Match** – Fast and precise when exact terms appear.  
2. **Fuzzy Matching** – Used RapidFuzz (Levenshtein distance) to capture spelling variations and OCR errors.  
3. **Semantic Search with Embeddings** – Used `all-MiniLM-L6-v2` (Sentence-Transformers) to embed both attribute definitions and clauses, and matched them via cosine similarity

- **Ranking Trick**  
  - Final relevance = weighted combination of fuzzy score + cosine similarity
  - This ensured robustness: exact matches ranked highest, but semantic matches filled gaps when the language varied

---

### 3️⃣ Clause Comparison & Classification  

Once the right clauses were mapped, the next challenge was **deciding whether they were Standard or Non-Standard**.  

- **Semantic Similarity Scoring**  
  - Contract clause vs. Standard template clause → embeddings via `all-MiniLM-L6-v2`.  
  - Cosine similarity threshold chosen empirically to distinguish Standard from Modified vs. Non-Standard

- **Zero-Shot LLM Classification (LLaMA 3.1)**  
  - Used to verify whether a clause *retains intent* even when the wording changes significantly.  
  - This helped reduce false negatives when clauses were rephrased but semantically identical

- **Multi-Level Classification**  
  - **Standard** → Matches or near-matches that are highly similar with slight modifications
  - **Non-Standard** → Deviations which contain risk

- **Confidence Scoring Trick**  
  - Each decision carries a confidence score derived from similarity + model probability.  
  - This makes the system more transparent for reviewers.  


## 🛠️ Key Techniques Used

- **Hybrid Retrieval (Exact + Fuzzy + Semantic)** → Prevented misses when attributes were expressed in a different way
- **OCR Preprocessing Tricks** → Deskewing and adaptive thresholding to remove unnecessary content and enhance the image for accurate character recognition
- **Sentence Embeddings (`all-MiniLM-L6-v2`)** → Important for fast processing with good semantic accuracy
- **Confidence Scores** → Made results interpretable, which is critical in legal/contract reviews
- **Lightweight Models** → Ensured faster inference while providing good quality results


## 📦 Installation  

```bash
# Clone repository
git clone https://github.com/your-repo/contract-classifier.git
cd contract-classifier

# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
ollama pull llama3.2:1b ## Model download
```

Ensure that all the Standard Template PDFs and all Contract PDFs and the Attribute Dictionary is present in the same directory before running the code

```bash
python main.py
```
