"""
Utility functions for DoctorFollow Medical Search Demo
Optimized for Turkish medical literature and practitioners
"""
import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Citation:
    """Structured citation with source tracking"""
    source_id: int
    chunk_text: str
    page_number: Optional[int] = None
    quote: Optional[str] = None


def clean_text(text: str) -> str:
    """
    Clean and normalize text from PDF documents.
    Preserves Turkish characters (ğüşıöçĞÜŞİÖÇ).

    Args:
        text: Raw text from PDF

    Returns:
        Cleaned text string
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but KEEP Turkish characters
    # Also keep common medical symbols: %, μ, °, etc.
    text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ.,!?()%°μ/-]', '', text)

    # Fix common Turkish OCR errors in PDFs
    text = text.replace('ı̇', 'i')  # Common PDF encoding issue

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.
    Optimized for Turkish sentence structures.

    Args:
        text: Text to chunk
        size: Target chunk size in characters (500 works well for Turkish)
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Get chunk
        end = start + size
        chunk = text[start:end]

        # Try to break at sentence boundary if possible
        if end < text_length:
            # Look for sentence ending punctuation (. ! ? common in Turkish)
            last_period = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
            if last_period > size // 2:  # Only break if we're past halfway
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1

        chunks.append(chunk.strip())

        # Move start forward with overlap
        start = end - overlap

        # Prevent infinite loop
        if start + size >= text_length and start < text_length:
            chunks.append(text[start:].strip())
            break

    return [c for c in chunks if c]  # Remove empty chunks


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (0-1)
    """
    if len(a.shape) == 1:
        a = a.reshape(1, -1)
    if len(b.shape) == 1:
        b = b.reshape(1, -1)

    dot_product = np.dot(a, b.T)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def rrf_fusion(bm25_scores: List[Tuple[int, float]],
               semantic_scores: List[Tuple[int, float]],
               k: int = 60) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion for combining BM25 and semantic search results.

    RRF is language-agnostic and works excellently for Turkish medical texts.

    Args:
        bm25_scores: List of (index, score) tuples from BM25
        semantic_scores: List of (index, score) tuples from semantic search
        k: RRF constant (default 60)

    Returns:
        Fused and sorted list of (index, score) tuples
    """
    rrf_scores = {}

    # Add BM25 scores with RRF formula: 1 / (k + rank)
    for rank, (idx, score) in enumerate(bm25_scores):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

    # Add semantic scores
    for rank, (idx, score) in enumerate(semantic_scores):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

    # Sort by RRF score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_results


def extract_citations(answer: str) -> List[int]:
    """
    Extract citation IDs from answer text.
    Supports Turkish medical citation formats:
    - [1], [2], [3] (Vancouver/NLM style - standard in Turkish medical journals)
    - [Kaynak 1], [Kaynak 2] (Turkish format)

    Turkish medical journals typically use Vancouver style (numeric citations).

    Args:
        answer: Generated answer text

    Returns:
        List of citation IDs found in the answer
    """
    # Match patterns: [1], [2], [Kaynak 1], etc.
    citation_pattern = r'\[(?:Kaynak|kaynak)?\s*(\d+)\]'
    matches = re.findall(citation_pattern, answer, re.IGNORECASE)

    # Convert to integers and remove duplicates while preserving order
    citation_ids = []
    seen = set()
    for match in matches:
        cid = int(match)
        if cid not in seen:
            citation_ids.append(cid)
            seen.add(cid)

    return citation_ids


def validate_citations(answer: str, num_sources: int) -> Dict[str, any]:
    """
    Validate citations in answer for completeness and correctness.

    Follows Turkish medical literature best practices (Vancouver style).

    Args:
        answer: Generated answer text
        num_sources: Number of source documents provided

    Returns:
        Dictionary with validation results
    """
    citation_ids = extract_citations(answer)

    result = {
        'is_valid': False,
        'has_citations': len(citation_ids) > 0,
        'citation_ids': citation_ids,
        'invalid_ids': [],
        'message': ''
    }

    # Check if citations exist
    if not citation_ids:
        result['message'] = 'Kaynaklarda atıf bulunamadı (No citations found)'
        return result

    # Validate citation IDs are within valid range
    invalid_ids = [cid for cid in citation_ids if cid < 1 or cid > num_sources]
    result['invalid_ids'] = invalid_ids

    if invalid_ids:
        result['message'] = f'Geçersiz kaynak numaraları: {invalid_ids}'
        return result

    # All checks passed
    result['is_valid'] = True
    result['message'] = f'Geçerli: {len(citation_ids)} kaynak referans verildi'

    return result


def format_sources_with_citations(chunks: List[str],
                                  citation_ids: Optional[List[int]] = None,
                                  max_display: int = 5) -> str:
    """
    Format source chunks with Vancouver-style numbering.
    Used by Turkish medical journals (similar to PubMed/MEDLINE format).

    Args:
        chunks: List of source text chunks
        citation_ids: Optional list of citation IDs that were actually used
        max_display: Maximum number of sources to display

    Returns:
        Formatted sources string (in Turkish)
    """
    if not chunks:
        return "\n\n📚 **Kaynaklar:** Kaynak bulunamadı."

    formatted = "\n\n📚 **Kaynaklar:**\n\n"

    # Display sources with numbering
    display_chunks = chunks[:max_display]

    for i, chunk in enumerate(display_chunks, 1):
        # Truncate long chunks for readability
        display_chunk = chunk[:250] + "..." if len(chunk) > 250 else chunk

        # Add checkmark if this source was cited
        used_indicator = "✓ " if citation_ids and i in citation_ids else ""

        formatted += f"**[{i}]** {used_indicator}{display_chunk}\n\n"

    if len(chunks) > max_display:
        formatted += f"_...ve {len(chunks) - max_display} kaynak daha_\n"

    return formatted


def create_citation_prompt_instruction() -> str:
    """
    Generate instruction for LLM to include proper citations.
    Optimized for Turkish medical context and practitioners.

    Returns:
        Prompt instruction string (in Turkish for better LLM understanding)
    """
    return """
Sen bir tıbbi asistansın. Tüm iddialarını kaynaklarla desteklemelisin.

KAYNAK KULLANIM KURALLARI:
- Her iddianın hemen ardından [1], [2], [3] formatında kaynak belirt
- Örnek: "Parasetamol ateş düşürücü etkiye sahiptir [1]."
- Birden fazla kaynak: "Bu bulgu desteklenmektedir [1][2]."
- Sağlanan kaynaklarda bilgi yoksa: "Bu konuda sağlanan kaynaklarda bilgi bulunmamaktadır."
- Kaynak vermeden iddia belirtme
- Sadece verilen kaynaklardaki bilgileri kullan

Turkish Medical Context:
- Use Istanbul decimal comma (,) for decimals if applicable: "2,5 mg"
- Common Turkish medical abbreviations are acceptable
- Maintain professional medical terminology
"""


def verify_answer_grounding(answer: str, source_chunks: List[str],
                           language: str = 'tr') -> Dict[str, any]:
    """
    Verify that claims in the answer can be found in source chunks.
    Optimized for Turkish medical text with Turkish-specific word processing.

    Args:
        answer: Generated answer text
        source_chunks: List of source text chunks
        language: Language code ('tr' for Turkish)

    Returns:
        Dictionary with grounding verification results
    """
    # Remove citation markers for analysis
    clean_answer = re.sub(r'\[(?:Kaynak|kaynak)?\s*\d+\]', '', answer, flags=re.IGNORECASE)

    # Split into sentences (Turkish uses . ! ? like English)
    sentences = re.split(r'[.!?]+', clean_answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Turkish stop words (common words to ignore in overlap calculation)
    turkish_stop_words = {
        've', 'veya', 'ile', 'bir', 'bu', 'şu', 'o', 'için', 'da', 'de',
        'mi', 'mu', 'mü', 'ki', 'ne', 'kadar', 'gibi', 'daha', 'çok', 'az'
    }

    grounded_sentences = 0
    total_sentences = len(sentences)

    for sentence in sentences:
        # Tokenize and clean
        words = set(w.lower() for w in sentence.split() if len(w) > 2)
        # Remove stop words
        words = words - turkish_stop_words

        if len(words) < 3:  # Skip very short sentences
            continue

        # Check if significant overlap with any source
        for chunk in source_chunks:
            chunk_words = set(w.lower() for w in chunk.split() if len(w) > 2)
            chunk_words = chunk_words - turkish_stop_words

            overlap = len(words & chunk_words)
            # Lower threshold for Turkish due to agglutination
            if overlap >= len(words) * 0.25:  # 25% overlap threshold
                grounded_sentences += 1
                break

    grounding_ratio = grounded_sentences / total_sentences if total_sentences > 0 else 0

    return {
        'grounding_ratio': grounding_ratio,
        'grounded_sentences': grounded_sentences,
        'total_sentences': total_sentences,
        'is_well_grounded': grounding_ratio >= 0.65  # 65% threshold for Turkish
    }
