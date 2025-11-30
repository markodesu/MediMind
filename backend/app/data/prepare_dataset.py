"""
Dataset preparation script for MediMind.
Loads, filters, and transforms the mediqa_qa dataset into a safe, student-friendly format.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# Keywords that indicate unsafe content to filter out
# Only filter truly dangerous content - let confidence scoring handle the rest
UNSAFE_KEYWORDS = [
    # Specific prescription instructions with dosages
    "prescribe", "prescription", "dosage of", "dose of",
    "take [0-9]+ mg", "take [0-9]+ ml", "take [0-9]+ tablet", "take [0-9]+ pill",
    "milligram", "milliliter",
]


def contains_unsafe_content(text: str) -> bool:
    """
    Check if text contains unsafe medical content.
    Very lenient - only filters specific prescription dosages.
    Trust confidence scoring and disclaimers for safety.
    
    Args:
        text: Text to check
        
    Returns:
        True if unsafe content is detected
    """
    if not text:
        return True
    
    text_lower = text.lower()
    
    # Only filter specific prescription dosage patterns
    # Pattern: "take X mg/ml" or "prescribe X mg"
    if re.search(r'(take|prescribe|prescription|dosage|dose)\s+\d+\s*(mg|ml|milligram|milliliter|tablet|pill|capsule)', text_lower):
        return True
    
    # Filter explicit prescription instructions
    if re.search(r'prescribe.*?\d+\s*(mg|ml)', text_lower):
        return True
    
    return False


def simplify_text(text: str) -> str:
    """
    Simplify text to student-friendly language.
    
    Args:
        text: Original text
        
    Returns:
        Simplified text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove technical jargon patterns (basic cleanup)
    # Replace common medical abbreviations with full words where safe
    replacements = {
        r'\bDr\.': 'doctor',
        r'\bvs\.': 'versus',
        r'\betc\.': 'and so on',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


def add_safety_disclaimer(answer: str) -> str:
    """
    Add safety disclaimer to answer if not already present.
    
    Args:
        answer: Original answer
        
    Returns:
        Answer with safety disclaimer
    """
    disclaimer = "If your symptoms worsen or feel concerning, seek help from a healthcare professional."
    
    if disclaimer.lower() not in answer.lower():
        if answer.endswith('.'):
            return f"{answer} {disclaimer}"
        else:
            return f"{answer}. {disclaimer}"
    
    return answer


def is_safe_entry(question: str, answer: str) -> bool:
    """
    Determine if an entry is safe for student use.
    Very lenient - only blocks specific prescription dosages.
    Trust confidence scoring and disclaimers for safety regulation.
    
    Args:
        question: User question
        answer: Answer text
        
    Returns:
        True if entry is safe
    """
    # Check for unsafe content (only specific prescription dosages)
    if contains_unsafe_content(question) or contains_unsafe_content(answer):
        return False
    
    # Check if answer is too short (likely incomplete or error)
    if len(answer.strip()) < 10:
        return False
    
    # Allow longer answers (up to 5000 chars for comprehensive information)
    if len(answer.strip()) > 5000:
        return False
    
    # Only filter extremely dangerous prescription patterns
    answer_lower = answer.lower()
    
    # Filter only explicit prescription dosages
    dangerous_patterns = [
        r'prescribe\s+\d+\s*(mg|ml)', 
        r'dosage.*?\d+\s*(mg|ml)',
        r'take\s+\d+\s*(mg|ml)\s+(of|every|daily)',
    ]
    
    has_dangerous_pattern = any(re.search(pattern, answer_lower) for pattern in dangerous_patterns)
    
    # Allow all general medical information - confidence scoring will regulate
    return not has_dangerous_pattern


def transform_entry(entry: Dict[str, Any]) -> Dict[str, str]:
    """
    Transform a dataset entry into MediMind format.
    
    Args:
        entry: Original dataset entry
        
    Returns:
        Transformed entry in MediMind format
    """
    # Extract question and answer from different possible field names
    question = ""
    answer = ""
    
    # Handle medical_questions_pairs dataset structure (question pairs)
    if "question_1" in entry and "question_2" in entry:
        # Use question_1 as the question, question_2 as context/answer
        question = str(entry["question_1"])
        # For question pairs, we'll use question_2 as a reference answer
        # but only if it's a proper answer (not another question)
        question_2 = str(entry["question_2"])
        if "?" not in question_2[-10:]:  # If question_2 doesn't end with ?
            answer = question_2
        else:
            # If both are questions, skip this entry
            return None
    
    # Try common field names
    elif "question" in entry:
        question = str(entry["question"])
    elif "input" in entry:
        question = str(entry["input"])
    elif "instruction" in entry:
        question = str(entry["instruction"])
    elif "text" in entry:
        question = str(entry["text"])
    
    if not answer:
        if "answer" in entry:
            answer = str(entry["answer"])
        elif "output" in entry:
            answer = str(entry["output"])
        elif "response" in entry:
            answer = str(entry["response"])
        elif "label" in entry and str(entry["label"]) not in ["0", "1"]:  # Not binary label
            answer = str(entry["label"])
    
    # If we have nested structures, try to extract
    if not question or not answer:
        # Try to get from nested structures
        for key, value in entry.items():
            if isinstance(value, dict):
                if "question" in value and not question:
                    question = str(value["question"])
                if "answer" in value and not answer:
                    answer = str(value["answer"])
    
    # Skip if we don't have both question and answer
    if not question or not answer:
        return None
    
    # Check if entry is safe
    if not is_safe_entry(question, answer):
        return None
    
    # Simplify text
    question = simplify_text(question)
    answer = simplify_text(answer)
    
    # Add safety disclaimer
    answer = add_safety_disclaimer(answer)
    
    return {
        "instruction": question,
        "response": answer,
        "source": "mediqa_qa",
        "safety_level": "safe"
    }


def parse_mediqa_xml(filepath: Path) -> List[Dict[str, Any]]:
    """
    Parse MEDIQA XML files and extract Q&A pairs.
    
    Args:
        filepath: Path to XML file
        
    Returns:
        List of entries with question and answer
    """
    entries = []
    
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Find all Question elements
        for question_elem in root.findall('.//Question'):
            question_text = question_elem.find('QuestionText')
            if question_text is None or not question_text.text:
                continue
            
            question = question_text.text.strip()
            
            # Get all answers for this question
            answer_list = question_elem.find('AnswerList')
            if answer_list is not None:
                for answer_elem in answer_list.findall('Answer'):
                    answer_text_elem = answer_elem.find('AnswerText')
                    if answer_text_elem is not None and answer_text_elem.text:
                        answer = answer_text_elem.text.strip()
                        # Use the highest ranked answer (lowest ReferenceRank)
                        reference_rank = int(answer_elem.get('ReferenceRank', '999'))
                        
                        entries.append({
                            'question': question,
                            'answer': answer,
                            'reference_rank': reference_rank,
                            'source_file': str(filepath.name)
                        })
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
    
    return entries


def prepare_and_save():
    """
    Main function to download, filter, transform, and save the dataset.
    """
    script_dir = Path(__file__).parent
    raw_data_dir = script_dir / "raw_datasets"
    mediqa_dir = raw_data_dir / "MEDIQA2019-master" / "MEDIQA_Task3_QA"
    
    all_entries = []
    
    # First, try to load from local MEDIQA XML files
    if mediqa_dir.exists():
        print("Found local MEDIQA dataset files!")
        xml_files = list(mediqa_dir.glob("*.xml"))
        
        if xml_files:
            print(f"Processing {len(xml_files)} XML files...")
            
            for xml_file in xml_files:
                print(f"\nProcessing: {xml_file.name}")
                raw_entries = parse_mediqa_xml(xml_file)
                print(f"  Found {len(raw_entries)} Q&A pairs")
                
                safe_count = 0
                unsafe_count = 0
                
                # Group by question and collect top 3 answers (for more training data)
                questions_dict = {}
                for entry in raw_entries:
                    q = entry['question']
                    if q not in questions_dict:
                        questions_dict[q] = []
                    questions_dict[q].append(entry)
                
                # Sort answers by reference rank and take top 3 per question
                for question, answers in questions_dict.items():
                    # Sort by reference rank (lower is better)
                    sorted_answers = sorted(answers, key=lambda x: x['reference_rank'])[:3]
                    
                    # Transform entries (use top 3 answers to increase dataset size)
                    for entry in sorted_answers:
                        transformed = transform_entry({
                            'question': entry['question'],
                            'answer': entry['answer']
                        })
                        
                        if transformed:
                            all_entries.append(transformed)
                            safe_count += 1
                        else:
                            unsafe_count += 1
                
                print(f"  Safe entries: {safe_count}")
                print(f"  Filtered out: {unsafe_count}")
    
    # If no local files, try datasets library
    if not all_entries and HAS_DATASETS:
        print("\nNo local MEDIQA files found. Trying datasets library...")
        dataset_sources = [
            "medical_questions_pairs",
            "bigbio/mediqa_qa",
        ]
        
        ds = None
        dataset_name = None
        
        for source in dataset_sources:
            print(f"Attempting to load dataset: {source}")
            try:
                ds = load_dataset(source)
                dataset_name = source
                print(f"✓ Dataset '{source}' loaded successfully. Splits: {list(ds.keys())}")
                break
            except Exception as e:
                print(f"✗ Failed to load '{source}': {e}")
                continue
        
        if ds:
            for split_name, split_data in ds.items():
                print(f"\nProcessing split: {split_name} ({len(split_data)} entries)")
                
                safe_count = 0
                unsafe_count = 0
                
                for entry in split_data:
                    transformed = transform_entry(entry)
                    
                    if transformed:
                        all_entries.append(transformed)
                        safe_count += 1
                    else:
                        unsafe_count += 1
                
                print(f"  Safe entries: {safe_count}")
                print(f"  Filtered out: {unsafe_count}")
    
    if not all_entries:
        print("\n❌ Error: Could not load any dataset.")
        print("Please ensure MEDIQA data is downloaded or check your internet connection.")
        return
    
    print(f"\nTotal safe entries collected: {len(all_entries)}")
    
    # Save to JSONL
    output_path = script_dir / "dataset.jsonl"
    print(f"\nSaving to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Dataset saved successfully! Total entries: {len(all_entries)}")
    print(f"File location: {output_path}")


if __name__ == "__main__":
    prepare_and_save()

