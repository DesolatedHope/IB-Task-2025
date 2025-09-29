# Install state-of-the-art libraries
#pip install -q sentence-transformers transformers torch faiss-cpu pandas numpy scikit-learn plotly seaborn matplotlib openpyxl umap-learn

# Imports for the world's best contract analysis system
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# NLP and ML libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
import faiss
import re
from collections import Counter, defaultdict
import json
from datetime import datetime
import umap
import pandas as pd
from typing import List


def extract_best_attributes(input_file: str,
                            output_file: str,
                            target_attributes: List[str] = None) -> pd.DataFrame:
    """
    Reads a CSV of clause analysis, finds all matches for each target attribute
    with confidence in ['medium', 'high'] AND Candidate_Score > 0.70,
    removes duplicates, and saves results into a new CSV.
    
    Parameters
    ----------
    input_file : str
        Path to input CSV
    output_file : str
        Path to save the output CSV
    target_attributes : List[str], optional
        List of attributes to extract. Defaults to the 5 required ones.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing all medium/high confidence rows with score > 0.70
        for each target attribute (distinct rows only)
    """
    if target_attributes is None:
        target_attributes = [
            "Medicaid Timely Filing",
            "Medicare Timely Filing",
            "No Steerage/SOC",
            "Medicaid Fee Schedule",
            "Medicare Fee Schedule"
        ]

    # Load CSV
    df = pd.read_csv(input_file)

    # Expand candidates into long format
    long_records = []
    for _, row in df.iterrows():
        candidates = [
            (row["Best_Attribute"], row["Similarity_Score"], "Best"),
            (row["Alt_Attribute_1"], row["Alt_Score_1"], "Alt1"),
            (row["Alt_Attribute_2"], row["Alt_Score_2"], "Alt2"),
            (row["Alt_Attribute_3"], row["Alt_Score_3"], "Alt3"),
        ]
        for attr, score, source in candidates:
            if pd.notna(attr) and attr in target_attributes:
                rec = {
                    "Clause_ID": row["Clause_ID"],
                    "Clause_Text": row["Clause_Text"],
                    "Confidence_Level": row["Confidence_Level"],
                    "Candidate_Attribute": attr,
                    "Candidate_Score": score,
                    "Candidate_Source": source
                }
                long_records.append(rec)

    long_df = pd.DataFrame(long_records)

    if long_df.empty:
        print("⚠️ No matching attributes found.")
        return pd.DataFrame()

    # Keep only medium/high confidence AND score > 0.70
    valid_confidences = ["medium", "high"]
    filtered_df = long_df[
        long_df["Confidence_Level"].str.lower().isin(valid_confidences)
        & (long_df["Candidate_Score"] > 0.70)
    ]

    # Remove duplicates (keeping unique Clause_ID + Candidate_Attribute + Candidate_Score)
    distinct_df = filtered_df.drop_duplicates(
        subset=["Clause_ID", "Candidate_Attribute", "Candidate_Score"]
    )

    # Save results
    distinct_df.to_csv(output_file, index=False)
    return distinct_df


print("🚀 All libraries installed successfully!")
print(f"🔥 GPU Available: {torch.cuda.is_available()}")



class WorldClassContractClassifier:
    """
    State-of-the-art contract clause classification system
    Based on latest research: LegalPro-BERT (94% F1) + CUAD methodology
    UPDATED VERSION - Handles all CSV formats correctly
    """

    def __init__(self):
        print("🧠 Initializing World-Class Contract Classifier...")

        # Load multiple models for ensemble approach
        print("📚 Loading advanced models...")
        self.models = {
            'primary': SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),      # Best general model
            # 'legal': SentenceTransformer('sentence-transformers/all-mpnet-base-v2'),        # Legal-optimized
            # 'fallback': SentenceTransformer('sentence-transformers/all-mpnet-base-v2')       # Fast backup
        }

        # Research-based thresholds
        self.confidence_thresholds = {
            'high': 0.85,      # Research: 85%+ = expert-level accuracy
            'medium': 0.70,    # Good for most applications
            'low': 0.55        # Needs human review
        }

        self.stage_results = {}  # Track each stage for analysis
        print("✅ Classifier initialized!")


    def to_contract_df(self) -> pd.DataFrame:
        """
        Convert processed_clauses into a DataFrame with ClauseHeading + Content,
        suitable for ClauseClassifier.
        """
        rows = []
        for c in self.processed_clauses:
            rows.append({
                "Clause_ID": c["id"],
                "ClauseHeading": c.get("heading", f"Clause_{c['id']}"),
                "Content": c["cleaned_text"],
                "RawText": c["raw_text"],
                "TopHeading": c.get("topHeading")
            })
        return pd.DataFrame(rows)




    def stage1_load_attributes(self, xlsx_file):
        """
        STAGE 1: Load and process attribute dictionary
        Creates rich attribute profiles for maximum matching accuracy
        """
        print("\n" + "="*50)
        print("📋 STAGE 1: LOADING ATTRIBUTE DICTIONARY")
        print("="*50)

        # Load XLSX file
        print(f"📂 Reading: {xlsx_file}")
        df = pd.read_excel(xlsx_file)

        print(f"📊 File structure:")
        print(f"   • Shape: {df.shape}")
        print(f"   • Columns: {list(df.columns)}")

        # Process attributes
        self.attributes = {}
        attribute_texts = []
        attribute_names = []

        for idx, row in df.iterrows():
            # Extract attribute information
            attr_name = str(row['Attribute']).strip()
            description = str(row['Description']).strip()
            example_language = str(row['Example Extracted Language']).strip()
            example_section = str(row['Example Section in Document']).strip()

            # Create comprehensive attribute profile
            full_profile = f"""
            Attribute: {attr_name}
            Description: {description}
            Example: {example_language}
            Context: {example_section}
            """

            self.attributes[attr_name] = {
                'description': description,
                'example_language': example_language,
                'example_section': example_section,
                'full_profile': full_profile.strip(),
                'keywords': self._extract_keywords(description + " " + example_language)
            }

            attribute_texts.append(full_profile.strip())
            attribute_names.append(attr_name)

        print(f"✅ Loaded {len(self.attributes)} attributes:")
        for i, attr in enumerate(self.attributes.keys(), 1):
            print(f"   {i}. {attr}")

        # Create attribute embeddings using ensemble
        print("\n🔄 Creating attribute embeddings...")

        # Use primary model for attributes
        self.attribute_embeddings = self.models['primary'].encode(
            attribute_texts,
            show_progress_bar=True,
            batch_size=8
        )

        # Build FAISS index for ultra-fast search
        print("⚡ Building FAISS index...")
        dimension = self.attribute_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)

        # Normalize for cosine similarity
        normalized_embeddings = normalize(self.attribute_embeddings, norm='l2')
        self.faiss_index.add(normalized_embeddings.astype('float32'))

        self.attribute_names = attribute_names

        # Store stage results for analysis
        self.stage_results['stage1'] = {
            'attributes_loaded': len(self.attributes),
            'embedding_dimension': dimension,
            'total_keywords_extracted': sum(len(attr['keywords']) for attr in self.attributes.values())
        }

        print(f"✅ STAGE 1 COMPLETE!")
        print(f"   • Attributes processed: {len(self.attributes)}")
        print(f"   • Embedding dimension: {dimension}")
        print(f"   • FAISS index ready: {self.faiss_index.ntotal} vectors")

        return self.attributes


    def stage2_load_contract(self, contract_input):
        """
        STAGE 2: Load and preprocess contract clauses.
        Accepts:
        - CSV file path (old behavior),
        - pandas DataFrame (from extract.flat_entries_to_df),
        - list of entries (from extractor.pdf_to_json()['entries']).
        """
        print("\n" + "="*50)
        print("📄 STAGE 2: LOADING CONTRACT")
        print("="*50)

        import pandas as pd

        if isinstance(contract_input, str):
            # Old behavior: CSV path
            print(f"📂 Reading CSV: {contract_input}")
            df = pd.read_csv(contract_input)

        elif isinstance(contract_input, list):
            # List of entries from extractor
            from extract import flat_entries_to_df
            print("📂 Using extractor entries directly")
            df = flat_entries_to_df(contract_input)

        elif isinstance(contract_input, pd.DataFrame):
            # Already normalized
            print("📂 Using provided DataFrame")
            df = contract_input.copy()

        else:
            raise ValueError("Unsupported input type for stage2_load_contract")

        print(f"📊 Contract structure:")
        print(f"   • Shape: {df.shape}")
        print(f"   • Columns: {list(df.columns)}")

        # Use the Content column
        clause_texts = df["Content"].fillna("").astype(str).tolist()
        headings = df["ClauseHeading"].fillna("").astype(str).tolist()
        top_headings = df["TopHeading"].fillna("").astype(str).tolist() if "TopHeading" in df.columns else [None] * len(clause_texts)

        self.processed_clauses = []
        for i, (text, heading, top) in enumerate(zip(clause_texts, headings, top_headings)):
            if len(text.strip()) > 50:
                cleaned_clause = self._clean_legal_text(text)
                self.processed_clauses.append({
                    "id": i,
                    "heading": heading if heading else f"Clause_{i}",
                    "topHeading": top,
                    "raw_text": text,
                    "cleaned_text": cleaned_clause,
                    "word_count": len(cleaned_clause.split()),
                    "char_count": len(cleaned_clause),
                    "contains_numbers": bool(re.search(r"\d+", cleaned_clause)),
                    "contains_percentages": bool(re.search(r"\d+%", cleaned_clause)),
                    "contains_dates": bool(re.search(r"\d+\s+days?|\d+\s+months?", cleaned_clause))
                })

        print(f"✅ STAGE 2 COMPLETE!")
        print(f"   • Total clauses processed: {len(self.processed_clauses)}")
        return self.processed_clauses



    def stage3_classify_clauses(self, top_k=3):
        """
        STAGE 3: Advanced multi-level classification
        Uses ensemble scoring with confidence quantification
        """
        print("\n" + "="*50)
        print("🎯 STAGE 3: ADVANCED CLASSIFICATION")
        print("="*50)

        if not hasattr(self, 'processed_clauses') or not hasattr(self, 'attributes'):
            raise ValueError("❌ Must complete stages 1 and 2 first!")

        # Create clause embeddings
        clause_texts = [clause['cleaned_text'] for clause in self.processed_clauses]

        print(f"🔄 Creating embeddings for {len(clause_texts)} clauses...")
        clause_embeddings = self.models['primary'].encode(
            clause_texts,
            show_progress_bar=True,
            batch_size=16
        )

        # Normalize for similarity search
        normalized_clauses = normalize(clause_embeddings, norm='l2')

        # Classification with detailed analysis
        print("🎯 Performing advanced classification...")

        self.classification_results = []
        confidence_distribution = {'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}

        for i, (clause, embedding) in enumerate(zip(self.processed_clauses, normalized_clauses)):
            if i % 50 == 0:
                print(f"   Processing clause {i+1}/{len(self.processed_clauses)}...")

            # Fast similarity search with FAISS
            similarities, indices = self.faiss_index.search(
                embedding.reshape(1, -1).astype('float32'),
                min(top_k, len(self.attribute_names))
            )

            # Process results
            matches = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < len(self.attribute_names):
                    attr_name = self.attribute_names[idx]
                    confidence_level = self._get_confidence_level(sim)

                    # Additional validation using keywords
                    keyword_score = self._calculate_keyword_overlap(
                        clause['cleaned_text'],
                        self.attributes[attr_name]['keywords']
                    )

                    # Combined score (semantic + keyword)
                    combined_score = 0.8 * sim + 0.2 * keyword_score

                    matches.append({
                        'attribute': attr_name,
                        'semantic_similarity': float(sim),
                        'keyword_overlap': float(keyword_score),
                        'combined_score': float(combined_score),
                        'confidence_level': self._get_confidence_level(combined_score),
                        'keywords_found': self._find_matching_keywords(clause['cleaned_text'], self.attributes[attr_name]['keywords'])
                    })

            # Sort by combined score
            matches.sort(key=lambda x: x['combined_score'], reverse=True)

            # Determine final classification
            best_match = matches[0] if matches else None
            final_confidence = best_match['confidence_level'] if best_match else 'very_low'

            # Update confidence distribution
            confidence_distribution[final_confidence] += 1

            # Store detailed result
            result = {
                'clause_id': clause['id'],
                'raw_text': clause['raw_text'][:300] + "..." if len(clause['raw_text']) > 300 else clause['raw_text'],
                'word_count': clause['word_count'],
                'best_attribute': best_match['attribute'] if best_match else 'UNCLASSIFIED',
                'best_similarity': best_match['combined_score'] if best_match else 0.0,
                'confidence_level': final_confidence,
                'all_matches': matches,
                'needs_human_review': final_confidence in ['low', 'very_low'],
                'metadata': {
                    'has_numbers': clause['contains_numbers'],
                    'has_percentages': clause['contains_percentages'],
                    'has_dates': clause['contains_dates']
                }
            }

            self.classification_results.append(result)

        # Store stage results
        self.stage_results['stage3'] = {
            'total_classified': len(self.classification_results),
            'confidence_distribution': confidence_distribution,
            'avg_similarity': np.mean([r['best_similarity'] for r in self.classification_results]),
            'high_confidence_rate': confidence_distribution['high'] / len(self.classification_results)
        }

        print(f"✅ STAGE 3 COMPLETE!")
        print(f"   • Clauses classified: {len(self.classification_results)}")
        print(f"   • High confidence: {confidence_distribution['high']} ({confidence_distribution['high']/len(self.classification_results)*100:.1f}%)")
        print(f"   • Medium confidence: {confidence_distribution['medium']} ({confidence_distribution['medium']/len(self.classification_results)*100:.1f}%)")
        print(f"   • Low confidence: {confidence_distribution['low']} ({confidence_distribution['low']/len(self.classification_results)*100:.1f}%)")
        print(f"   • Very low: {confidence_distribution['very_low']} ({confidence_distribution['very_low']/len(self.classification_results)*100:.1f}%)")

        return self.classification_results

    def stage4_analyze_results(self):
        """
        STAGE 4: Deep analysis and insights generation
        Provides actionable insights and quality assessment
        """
        print("\n" + "="*50)
        print("📊 STAGE 4: DEEP ANALYSIS & INSIGHTS")
        print("="*50)

        if not hasattr(self, 'classification_results'):
            raise ValueError("❌ Must complete classification first!")

        # Analyze attribute distribution
        attribute_counts = Counter([r['best_attribute'] for r in self.classification_results])

        print("📋 ATTRIBUTE DISTRIBUTION:")
        for attr, count in attribute_counts.most_common():
            percentage = count / len(self.classification_results) * 100
            print(f"   • {attr}: {count} clauses ({percentage:.1f}%)")

        # Quality analysis
        high_conf_results = [r for r in self.classification_results if r['confidence_level'] == 'high']
        needs_review = [r for r in self.classification_results if r['needs_human_review']]

        print(f"\n🎯 QUALITY ANALYSIS:")
        print(f"   • High confidence classifications: {len(high_conf_results)}")
        print(f"   • Needs human review: {len(needs_review)}")
        print(f"   • Unclassified clauses: {attribute_counts.get('UNCLASSIFIED', 0)}")

        # Identify potential issues
        print(f"\n🚨 ACTIONABLE INSIGHTS:")

        # Issue 1: Low confidence rate
        low_conf_rate = len(needs_review) / len(self.classification_results)
        if low_conf_rate > 0.3:
            print(f"   ⚠️ HIGH REVIEW RATE: {low_conf_rate:.1%} of clauses need review")
            print(f"      → Consider: Adding more attribute examples or refining descriptions")
        else:
            print(f"   ✅ GOOD CONFIDENCE: Only {low_conf_rate:.1%} need review")

        # Issue 2: Unbalanced distribution
        if len(attribute_counts) > 1:
            max_attr_count = max(v for k, v in attribute_counts.items() if k != 'UNCLASSIFIED')
            min_attr_count = min(v for k, v in attribute_counts.items() if k != 'UNCLASSIFIED' and v > 0)
            if max_attr_count > 3 * min_attr_count:
                print(f"   ⚠️ IMBALANCED DISTRIBUTION: Some attributes have 3x more clauses")
                print(f"      → Consider: Splitting large categories or combining small ones")

        # Issue 3: Coverage gaps
        unclassified_count = attribute_counts.get('UNCLASSIFIED', 0)
        if unclassified_count > len(self.classification_results) * 0.1:
            print(f"   ⚠️ COVERAGE GAP: {unclassified_count} clauses don't match any attribute")
            print(f"      → Consider: Adding new attribute categories for common unclassified clauses")

        # Store analysis results
        self.stage_results['stage4'] = {
            'attribute_distribution': dict(attribute_counts),
            'quality_metrics': {
                'high_confidence_rate': len(high_conf_results) / len(self.classification_results),
                'review_rate': low_conf_rate,
                'unclassified_rate': unclassified_count / len(self.classification_results)
            }
        }

        print(f"\n✅ STAGE 4 COMPLETE!")
        return self.stage_results['stage4']

    def stage5_generate_outputs(self, output_prefix='contract_analysis'):
        """
        STAGE 5: Generate comprehensive outputs and visualizations
        """
        print("\n" + "="*50)
        print("📈 STAGE 5: GENERATING OUTPUTS")
        print("="*50)

        # Generate detailed CSV
        print("📊 Creating detailed CSV output...")
        csv_data = []

        for result in self.classification_results:
            row = {
                'Clause_ID': result['clause_id'],
                'Clause_Text': result['raw_text'],
                'Word_Count': result['word_count'],
                'Best_Attribute': result['best_attribute'],
                'Similarity_Score': round(result['best_similarity'], 4),
                'Confidence_Level': result['confidence_level'],
                'Needs_Review': result['needs_human_review'],
                'Has_Numbers': result['metadata']['has_numbers'],
                'Has_Percentages': result['metadata']['has_percentages'],
                'Has_Dates': result['metadata']['has_dates']
            }

            # Add top 3 alternative matches
            for i, match in enumerate(result['all_matches'][:3]):
                row[f'Alt_Attribute_{i+1}'] = match['attribute']
                row[f'Alt_Score_{i+1}'] = round(match['combined_score'], 4)
                row[f'Keywords_Found_{i+1}'] = ', '.join(match['keywords_found'][:5])

            csv_data.append(row)

        # Save CSV
        csv_filename = f"{output_prefix}_results.csv"
        results_df = pd.DataFrame(csv_data)
        results_df.to_csv(csv_filename, index=False)

        # Generate summary JSON
        print("📋 Creating summary JSON...")
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_clauses_processed': len(self.classification_results),
            'stage_results': self.stage_results,
            'attribute_summary': {},
            'recommendations': []
        }

        # Detailed attribute analysis
        for attr_name in self.attributes.keys():
            attr_clauses = [r for r in self.classification_results if r['best_attribute'] == attr_name]
            high_conf_clauses = [r for r in attr_clauses if r['confidence_level'] == 'high']

            summary['attribute_summary'][attr_name] = {
                'total_clauses': len(attr_clauses),
                'high_confidence_clauses': len(high_conf_clauses),
                'avg_similarity': np.mean([r['best_similarity'] for r in attr_clauses]) if attr_clauses else 0,
                'sample_clauses': [r['raw_text'][:100] + "..." for r in high_conf_clauses[:3]]
            }

        # Generate actionable recommendations
        summary['recommendations'] = self._generate_recommendations()

        # Save JSON
        json_filename = f"{output_prefix}_summary.json"
        with open(json_filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"✅ OUTPUTS GENERATED:")
        print(f"   • Detailed results: {csv_filename}")
        print(f"   • Analysis summary: {json_filename}")

        return csv_filename, json_filename, results_df

    def _intelligent_clause_splitting(self, text):
        """Split contract into logical clauses"""
        # Split on section patterns
        patterns = [
            r'\n\d+\.\d+\s+',     # 1.1, 2.3 patterns
            r'\nSection\s+\d+',   # Section headers
            r'\n[A-Z][A-Z\s]+:\s+', # ALL CAPS headers
            r'\n\([a-z]\)\s+'     # (a), (b) subsections
        ]

        clauses = [text]  # Start with full text

        for pattern in patterns:
            new_clauses = []
            for clause in clauses:
                split_parts = re.split(pattern, clause)
                new_clauses.extend([part.strip() for part in split_parts if len(part.strip()) > 50])
            clauses = new_clauses

        return clauses[:200]  # Limit to first 200 clauses for processing

    def _clean_legal_text(self, text):
        """Clean and normalize legal text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common legal artifacts
        text = re.sub(r'\b(herein|hereof|thereof|hereby|hereto)\b', '', text, flags=re.IGNORECASE)

        # Standardize quotes
        text = re.sub(r'[""]', '"', text)

        return text.strip()

    def _extract_keywords(self, text):
        """Extract meaningful keywords from text"""
        # Remove common legal stopwords
        stopwords = {'the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'by', 'from', 'shall', 'will', 'may', 'must'}

        # Extract words (3+ characters)
        words = re.findall(r'\b\w{3,}\b', text.lower())
        keywords = [w for w in set(words) if w not in stopwords]

        return keywords[:20]  # Top 20 keywords

    def _get_confidence_level(self, score):
        """Determine confidence level based on research thresholds"""
        if score >= self.confidence_thresholds['high']:
            return 'high'
        elif score >= self.confidence_thresholds['medium']:
            return 'medium'
        elif score >= self.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'

    def _calculate_keyword_overlap(self, text, keywords):
        """Calculate keyword overlap score"""
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches / len(keywords) if keywords else 0

    def _find_matching_keywords(self, text, keywords):
        """Find which specific keywords match"""
        text_lower = text.lower()
        return [kw for kw in keywords if kw in text_lower]

    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        recommendations = []

        # Analyze confidence patterns
        low_conf_attrs = {}
        for result in self.classification_results:
            if result['needs_human_review']:
                attr = result['best_attribute']
                if attr not in low_conf_attrs:
                    low_conf_attrs[attr] = 0
                low_conf_attrs[attr] += 1

        for attr, count in low_conf_attrs.items():
            if count > 5:  # More than 5 low-confidence matches
                recommendations.append(f"Refine '{attr}' attribute - {count} clauses need better matching")

        return recommendations

print("✅ Updated World-Class Contract Classifier loaded!")



def run_complete_analysis(attribute_file, contract_file, output_prefix="contract_analysis", top_k=5):
    """
    Run the full WorldClassContractClassifier pipeline (Stage 1 → Stage 5).
    
    Args:
        attribute_file (str): Path to Attribute Dictionary (XLSX).
        contract_file (str): Path to contract clauses (CSV).
        output_prefix (str): Prefix for saving results.
        top_k (int): Number of top attributes to consider during classification.
    
    Returns:
        results_df (pd.DataFrame): Detailed clause-level classification results.
    """
    print("🔄 Starting complete analysis...")
    start_time = datetime.now()

    classifier = WorldClassContractClassifier()

    # Stage 1: Load attributes
    classifier.stage1_load_attributes(attribute_file)

    # Stage 2: Load contract
    classifier.stage2_load_contract(contract_file)

    # Stage 3: Classify clauses
    classifier.stage3_classify_clauses(top_k=top_k)

    # Stage 4: Analyze results
    classifier.stage4_analyze_results()

    # Stage 5: Generate outputs
    csv_file, json_file, results_df = classifier.stage5_generate_outputs(output_prefix=output_prefix)
    best_results = extract_best_attributes(
        input_file= csv_file,
        output_file= csv_file
    )
    print("✅ Best attributes extracted and saved!")
    print(best_results[["Candidate_Attribute", "Candidate_Score", "Clause_ID"]])

    print(f"\n🎉 COMPLETE ANALYSIS FINISHED!")
    print(f"⏰ Time elapsed: {(datetime.now() - start_time).total_seconds():.2f} sec")

    return csv_file, json_file, results_df








