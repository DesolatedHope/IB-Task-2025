import pymupdf 
import easyocr
from PIL import Image
import io
import cv2
import numpy as np
import os, argparse
import re
import pandas as pd
from collections import Counter
import json

from variables import *

import pandas as pd

def flat_entries_to_df(entries: list) -> pd.DataFrame:
    """
    Convert extractor flat JSON entries into a normalized DataFrame
    for world_classifier / clause_classifier.
    """
    df = pd.DataFrame(entries)

    # Normalize column names
    rename_map = {
        "clauseHeading": "ClauseHeading",
        "content": "Content",
        "topHeading": "TopHeading"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure ClauseHeading exists
    if "ClauseHeading" not in df.columns:
        df["ClauseHeading"] = [f"Clause_{i}" for i in range(len(df))]

    # Ensure Content exists
    if "Content" not in df.columns:
        raise ValueError("Extractor entries missing 'content' field")

    # Reorder columns for consistency
    cols = ["ClauseHeading", "Content"]
    for extra in ["TopHeading", "page", "order", "type", "level"]:
        if extra in df.columns:
            cols.append(extra)

    return df[cols]

class TextExtractor:
    def __init__(self, languages=['en']):
        self.reader = easyocr.Reader(languages, gpu=True)

        # Patterns
        self.caps_pattern = re.compile(
            r'^(?P<heading>[A-Z][A-Z0-9\-\s\:\,\(\)\/&]{2,})\s*$',
            re.MULTILINE
        )
        self.numbered_pattern = re.compile(
            r'^\s*(?P<number>\d+(?:\.\d+)*)\s*\.?\s*(?P<title>.*\S.*)?$',
            re.MULTILINE
        )

        self.header_ratio = 0.05
        self.footer_ratio = 0.1

    def image_to_text(self, doc_image):
        mat = pymupdf.Matrix(3.0, 3.0)
        pix = doc_image.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        h, w, _ = img.shape
        top = int(h * self.header_ratio)
        bottom = int(h * (1 - self.footer_ratio))
        img = img[top:bottom, :]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self._preprocess_image(gray)

        results = self.reader.readtext(gray, paragraph=True, width_ths=0.8, detail=1)

        text_blocks = []
        for item in results:
            if len(item) == 3:
                _, text, prob = item
                if prob > 0.5:
                    text_blocks.append(text.strip())
            elif len(item) == 2:
                _, text = item
                text_blocks.append(text.strip())
            elif isinstance(item, str):
                text_blocks.append(item.strip())

        return "\n".join(text_blocks)

    def _preprocess_image(self, gray):
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(cnt)
            if area > 500 and 0.5 < aspect_ratio < 5:
                roi = gray[y:y+h, x:x+w]
                if np.mean(roi) < 50:
                    cv2.rectangle(thresh, (x, y), (x+w, y+h), (255, 255, 255), -1)

        return thresh

    def extract_text_without_headers_footers(self, page):
        words = page.get_text("words")
        page_height = page.rect.height
        header_cutoff = page_height * self.header_ratio
        footer_cutoff = page_height * (1 - self.footer_ratio)

        filtered_words = [
            w[4] for w in words
            if header_cutoff < w[1] < footer_cutoff
        ]
        return " ".join(filtered_words)

    def extract_hierarchical_structure(self, pages_text):
        document_structure = {
            "document": {
                "sections": [],
                "metadata": {
                    "total_pages": len(pages_text),
                    "extraction_date": pd.Timestamp.now().isoformat()
                }
            }
        }
        
        order = 0
        
        for page_idx, page_text in enumerate(pages_text):
            cleaned = self._clean_text(page_text)
            top_sections = self._split_by_caps_headings(cleaned)
            
            for sec in top_sections:
                top_heading = sec['heading']
                top_content = sec['content']
                clauses = self._split_by_numbered_clauses(top_content)
                
                section_data = {
                    "heading": top_heading,
                    "type": "section",
                    "page": page_idx + 1,
                    "order": order,
                    "content": None,
                    "clauses": []
                }
                order += 1
                
                if clauses:
                    section_data["clauses"] = self._build_clause_hierarchy(clauses, page_idx + 1, order)
                    order += self._count_all_clauses(clauses)
                else:
                    section_data["content"] = top_content
                
                document_structure["document"]["sections"].append(section_data)
        
        return document_structure
    
    def _build_clause_hierarchy(self, clauses, page, start_order):
        result = []
        order_counter = start_order
        
        for clause in clauses:
            clause_data = {
                "heading": clause['Heading'],
                "content": clause['Content'],
                "type": "clause",
                "level": clause['level'],
                "page": page,
                "order": order_counter,
                "subclauses": []
            }
            order_counter += 1
            
            if clause.get("Subclauses"):
                clause_data["subclauses"] = self._build_clause_hierarchy(
                    clause["Subclauses"], page, order_counter
                )
                order_counter += (self._count_all_clauses(clause["Subclauses"]))
            
            result.append(clause_data)
        
        return result
    
    def _count_all_clauses(self, clauses):
        count = len(clauses)
        for clause in clauses:
            if clause.get("Subclauses"):
                count += self._count_all_clauses(clause["Subclauses"])
        return count

    def extract_flat_structure(self, pages_text):
        rows = []
        order = 0

        def flatten_clauses(top_heading, clause, page_idx, parent=None):
            nonlocal order
            rows.append({
                "topHeading": top_heading,
                "clauseHeading": clause['Heading'],
                "content": clause['Content'],
                "type": "clause",
                "level": clause['level'],
                "parent": parent,
                "page": page_idx + 1,
                "order": order
            })
            order += 1

            for sub in clause.get("Subclauses", []):
                flatten_clauses(top_heading, sub, page_idx, parent=clause['Heading'])

        for page_idx, page_text in enumerate(pages_text):
            cleaned = self._clean_text(page_text)
            top_sections = self._split_by_caps_headings(cleaned)

            for sec in top_sections:
                top_heading = sec['heading']
                top_content = sec['content']
                clauses = self._split_by_numbered_clauses(top_content)

                if clauses:
                    for clause in clauses:
                        flatten_clauses(top_heading, clause, page_idx)
                else:
                    rows.append({
                        "topHeading": top_heading,
                        "clauseHeading": None,
                        "content": top_content,
                        "type": "section",
                        "level": 1,
                        "parent": None,
                        "page": page_idx + 1,
                        "order": order
                    })
                    order += 1

        return rows

    def _clean_text(self, text):
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\r\n?', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        return text.strip()

    def _split_by_caps_headings(self, text):
        matches = list(self.caps_pattern.finditer(text))
        if not matches:
            return [{'heading': 'Document Content', 'content': text}]

        sections = []
        first = matches[0]
        if first.start() > 0:
            pre = text[:first.start()].strip()
            if pre:
                sections.append({'heading': 'Document Content', 'content': pre})
        for i, m in enumerate(matches):
            heading = m.group('heading').strip()
            start_content = m.end()
            end_content = matches[i+1].start() if i+1 < len(matches) else len(text)
            content = text[start_content:end_content].strip()
            sections.append({'heading': heading, 'content': content})
        return sections

    def _split_by_numbered_clauses(self, text):
        pattern = re.compile(r'\n?(\d+(?:\.\d+)+)\s') 
        matches = list(pattern.finditer(text))
        clauses = []

        clause_map = {}

        for i, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.end()
            end = matches[i+1].start() if (i + 1) < len(matches) else len(text)
            content = text[start:end].strip()

            clause_data = {
                "Heading": heading,
                "Content": content,
                "level": heading.count(".") + 1,
                "Subclauses": []
            }

            clause_map[heading] = clause_data

            if "." in heading:
                parent = ".".join(heading.split(".")[:-1])
                if parent in clause_map:
                    clause_map[parent]["Subclauses"].append(clause_data)
                else:
                    clauses.append(clause_data)
            else:
                clauses.append(clause_data)

        return clauses

    def pdf_to_json(self, pdf_name, output_format='flat'):
        doc = pymupdf.open(pdf_name + ".pdf")
        text = []
        raw_pages = []

        print(f"Processing {len(doc)} pages...")
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = self.extract_text_without_headers_footers(page)

            if page_text.strip():
                text.append(page_text)
                raw_pages.append({
                    "page": page_num + 1,
                    "content": page_text,
                    "method": "text_extraction"
                })
            else:
                print(f"OCR on page {page_num+1} ...")
                ocr_text = self.image_to_text(doc[page_num])
                text.append(ocr_text)
                raw_pages.append({
                    "page": page_num + 1,
                    "content": ocr_text,
                    "method": "ocr"
                })

        os.makedirs('./result', exist_ok=True)
        
        raw_json_path = f'./result/{pdf_name}_raw.json'
        with open(raw_json_path, 'w', encoding='utf-8') as f:
            json.dump({"pages": raw_pages}, f, indent=2, ensure_ascii=False)
        print(f"Raw text JSON saved to {raw_json_path}")

        if output_format in ['hierarchical', 'both']:
            hierarchical_structure = self.extract_hierarchical_structure(text)
            hierarchical_json_path = f'./result/{pdf_name}_hierarchical.json'
            with open(hierarchical_json_path, 'w', encoding='utf-8') as f:
                json.dump(hierarchical_structure, f, indent=2, ensure_ascii=False)
            print(f"Hierarchical JSON saved to {hierarchical_json_path}")

        if output_format in ['flat', 'both']:
            flat_structure = self.extract_flat_structure(text)
            flat_json_path = f'./result/{pdf_name}_flat.json'
            with open(flat_json_path, 'w', encoding='utf-8') as f:
                json.dump({"entries": flat_structure}, f, indent=2, ensure_ascii=False)
                print(f"Flat JSON saved to {flat_json_path}")

        if output_format == 'both':
            combined_json_path = f'./result/{pdf_name}_combined.json'
            combined_data = {
                "metadata": {
                    "filename": pdf_name + ".pdf",
                    "total_pages": len(doc),
                    "extraction_date": pd.Timestamp.now().isoformat(),
                    "extraction_method": "PyMuPDF + EasyOCR"
                },
                "raw_pages": raw_pages,
                "hierarchical_structure": hierarchical_structure["document"],
                "flat_structure": flat_structure
            }
            with open(combined_json_path, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            print(f"Combined JSON saved to {combined_json_path}")

        doc.close()
        
        if output_format == 'hierarchical':
            return hierarchical_structure
        elif output_format == 'flat':
            return {"entries": flat_structure}
        else:  # both
            return combined_data
    
    def _write_hierarchical_section(self, file, sections, indent_level=0):
        """Write hierarchical sections to text file"""
        indent = "  " * indent_level
        
        for section in sections:
            file.write(f"{indent}{section['heading']}\n")
            file.write(f"{indent}{'=' * len(section['heading'])}\n")
            
            if section.get('content'):
                file.write(f"{indent}{section['content']}\n\n")
            
            if section.get('clauses'):
                self._write_clauses(file, section['clauses'], indent_level + 1)
            
            file.write("\n")
    
    def _write_clauses(self, file, clauses, indent_level):
        """Write clauses to text file"""
        indent = "  " * indent_level
        
        for clause in clauses:
            file.write(f"{indent}{clause['heading']} - {clause['content'][:100]}...\n")
            
            if clause.get('subclauses'):
                self._write_clauses(file, clause['subclauses'], indent_level + 1)


# if __name__ == "__main__":
#     te = TextExtractor(languages=['en'])

#     flat_result = te.pdf_to_json(TN_STANDARD_TEMPLATE, output_format='flat')