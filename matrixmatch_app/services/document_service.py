import json
import os
import io
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import docx

from matrixmatch_app.repositories import documents, document_logs


def list_all_documents() -> List[Dict]:
    return documents.list_all_documents()


def get_document(document_id: int) -> Optional[Dict]:
    return documents.get_document_by_id(document_id)


def add_document(title: str, program: str, abstract: str, modified_by: str) -> Tuple[bool, str]:
    title = (title or "").strip()
    program = (program or "").strip()
    abstract = (abstract or "").strip()

    if not title:
        return False, "Title is required."
    
    if not abstract:
        return False, "Abstract is required."
        
    if documents.check_duplicate_document(title, abstract):
        return False, "A document with the exact same title or abstract already exists."
        
    doc_id = documents.add_document(title, program, abstract)
    if doc_id:
        document_logs.add_log(doc_id, title, "Added", modified_by)
        return True, "Document added successfully."
    return False, "Failed to add document."


def update_document(document_id: int, title: str, program: str, abstract: str, modified_by: str) -> Tuple[bool, str]:
    title = (title or "").strip()
    program = (program or "").strip()
    abstract = (abstract or "").strip()

    if not title:
        return False, "Title is required."
    
    if not abstract:
        return False, "Abstract is required."
        
    if documents.check_duplicate_document(title, abstract, exclude_id=document_id):
        return False, "Another document with the exact same title or abstract already exists."
        
    updated = documents.update_document(document_id, title, program, abstract)
    if updated:
        document_logs.add_log(document_id, title, "Edited", modified_by)
        return True, "Document updated successfully."
    return False, "Failed to update document or document not found."


def delete_document(document_id: int, modified_by: str) -> bool:
    doc = documents.get_document_by_id(document_id)
    if not doc:
        return False
    title = doc.get("title", f"Document #{document_id}")
    
    deleted = documents.delete_document(document_id)
    if deleted:
        document_logs.add_log(document_id, title, "Deleted", modified_by)
        return True
    return False


def extract_document_info(file_data: bytes, filename: str) -> Tuple[bool, str, Dict[str, str]]:
    """
    Attempts to extract the title and abstract from an uploaded file.
    Returns (success, message, data_dict).
    """
    ext = os.path.splitext(filename)[1].lower()
    
    title = ""
    abstract = ""
    
    try:
        if ext == '.json':
            data = json.loads(file_data.decode('utf-8'))
            title = data.get('title', '')
            abstract = data.get('abstract', '')
            
        elif ext == '.txt':
            text = file_data.decode('utf-8')
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
                title = lines[0]
                abstract = "\n".join(lines[1:])
                
        elif ext in ['.docx', '.doc']:
            # docx can sometimes handle .doc but typically it's for .docx
            file_stream = io.BytesIO(file_data)
            doc = docx.Document(file_stream)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            if paragraphs:
                title = paragraphs[0]
                abstract = "\n".join(paragraphs[1:])
                
        elif ext == '.pdf':
            file_stream = io.BytesIO(file_data)
            doc = fitz.open(stream=file_stream, filetype="pdf")
            
            # Very basic heuristic: extract first few chunks of text for title
            # and the rest of the first page for the abstract.
            if len(doc) > 0:
                first_page = doc[0]
                text = first_page.get_text()
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                if lines:
                    title = lines[0]
                    # If "Abstract" is explicitly there, try to split
                    if "Abstract" in text or "ABSTRACT" in text:
                        try:
                            # Try case-insensitive split
                            import re
                            parts = re.split(r'(?i)abstract', text, maxsplit=1)
                            if len(parts) > 1:
                                abstract = parts[1].strip()
                            else:
                                abstract = "\n".join(lines[1:])
                        except Exception:
                            abstract = "\n".join(lines[1:])
                    else:
                        abstract = "\n".join(lines[1:])
            doc.close()
            
        else:
            return False, f"Unsupported file extension: {ext}", {}
            
    except Exception as e:
        return False, f"Error processing file: {str(e)}", {}
        
    return True, "Extraction successful", {
        "title": title.strip(),
        "abstract": abstract.strip()
    }
