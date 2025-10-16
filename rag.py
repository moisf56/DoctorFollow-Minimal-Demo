"""
Medical RAG System for DoctorFollow Demo
Hybrid search with BM25 + Semantic search + AWS Bedrock
"""
import os
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import boto3
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import PyPDF2
import numpy as np

from utils import (
    clean_text,
    chunk_text,
    cosine_similarity,
    rrf_fusion,
    extract_citations,
    validate_citations,
    format_sources_with_citations,
    create_citation_prompt_instruction,
)


class MedicalRAG:
    """
    Medical RAG system optimized for Turkish medical literature.

    Features:
    - Hybrid search (BM25 + Semantic)
    - Conversational memory
    - Citation tracking
    - AWS Bedrock LLM integration
    """

    def __init__(self):
        """Initialize the RAG system with embeddings model and AWS Bedrock."""
        print("🔧 Initializing Medical RAG System...")

        # Load sentence transformer for embeddings
        # e5-small-v2 works well for Turkish and is lightweight
        print("📦 Loading embedding model (e5-small-v2)...")
        self.embeddings_model = SentenceTransformer('intfloat/e5-small-v2')
        print("✓ Embedding model loaded")

        # Initialize AWS Bedrock client
        print("☁️ Connecting to AWS Bedrock...")
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            )
            print("✓ AWS Bedrock connected")
        except Exception as e:
            print(f"⚠️ AWS Bedrock connection warning: {e}")
            self.bedrock_client = None

        # Document storage
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.document_name: str = ""

        # Stats
        self.stats = {
            'total_queries': 0,
            'total_chunks': 0,
            'avg_response_time': 0.0
        }

        print("✓ Medical RAG System initialized\n")

    def ingest_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from PDF, chunk it, create embeddings, and index.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with ingestion statistics
        """
        print(f"📄 Processing PDF: {pdf_path}")

        # Extract text from PDF
        try:
            text = self._extract_pdf_text(pdf_path)
            print(f"✓ Extracted {len(text)} characters")
        except Exception as e:
            return {'error': f'PDF extraction failed: {str(e)}'}

        # Clean text
        text = clean_text(text)
        print(f"✓ Cleaned text")

        # Chunk text
        self.chunks = chunk_text(text, size=500, overlap=50)
        self.stats['total_chunks'] = len(self.chunks)
        print(f"✓ Created {len(self.chunks)} chunks")

        if not self.chunks:
            return {'error': 'No text chunks created from PDF'}

        # Create embeddings
        print(f"🧮 Creating embeddings for {len(self.chunks)} chunks...")
        try:
            # e5 models need "query: " or "passage: " prefix
            passages = [f"passage: {chunk}" for chunk in self.chunks]
            self.embeddings = self.embeddings_model.encode(
                passages,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            print(f"✓ Embeddings created: {self.embeddings.shape}")
        except Exception as e:
            return {'error': f'Embedding creation failed: {str(e)}'}

        # Create BM25 index
        print("🔍 Creating BM25 index...")
        tokenized_chunks = [chunk.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        print("✓ BM25 index created")

        # Store document name
        self.document_name = Path(pdf_path).name

        return {
            'success': True,
            'document_name': self.document_name,
            'total_chunks': len(self.chunks),
            'total_characters': len(text),
            'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0
        }

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"⚠️ Warning: Could not extract page {page_num}: {e}")
                    continue
        return text

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Tuple[int, str]]:
        """
        Perform hybrid search using BM25 + Semantic search with RRF fusion.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (index, chunk) tuples
        """
        if not self.chunks or self.embeddings is None or self.bm25 is None:
            return []

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        bm25_results = [(idx, float(bm25_scores[idx])) for idx in bm25_top_indices]

        # Semantic search
        query_embedding = self.embeddings_model.encode(
            f"query: {query}",
            convert_to_numpy=True
        )

        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        semantic_top_indices = np.argsort(similarities)[::-1][:top_k * 2]
        semantic_results = [(idx, float(similarities[idx])) for idx in semantic_top_indices]

        # RRF Fusion
        fused_results = rrf_fusion(bm25_results, semantic_results)

        # Return top_k results with chunks
        results = []
        for idx, score in fused_results[:top_k]:
            results.append((idx, self.chunks[idx]))

        return results

    def generate_answer(self,
                       query: str,
                       context_chunks: List[str],
                       conversation_history: List[Dict] = None) -> str:
        """
        Generate answer using AWS Bedrock with citations.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            conversation_history: Previous conversation turns

        Returns:
            Generated answer with citations
        """
        if not self.bedrock_client:
            return "⚠️ AWS Bedrock bağlantısı mevcut değil. Lütfen AWS kimlik bilgilerinizi kontrol edin."

        # Prepare context with source numbers
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"[Kaynak {i}]\n{chunk}\n\n"

        # Build conversation history
        history_text = ""
        if conversation_history:
            for turn in conversation_history[-3:]:  # Last 3 turns for context
                history_text += f"Kullanıcı: {turn['user']}\nAsistan: {turn['assistant']}\n\n"

        # Create prompt with citation instructions
        citation_instruction = create_citation_prompt_instruction()

        prompt = f"""Sen DoctorFollow tıbbi asistan sistemisin. Sağlık profesyonellerine Türkçe tıbbi dokümanlarda arama yaparken yardımcı oluyorsun.

{citation_instruction}

Önceki Konuşma:
{history_text if history_text else "Yok"}

Kaynak Dökümanlar:
{context_text}

Kullanıcı Sorusu: {query}

Cevap (kaynaklarla):"""

        # Call AWS Bedrock (Llama 3.1 8B)
        try:
            request_body = {
                "prompt": prompt,
                "max_gen_len": 512,
                "temperature": 0.3,  # Lower temperature for factual medical info
                "top_p": 0.9,
            }

            response = self.bedrock_client.invoke_model(
                modelId="meta.llama3-1-8b-instruct-v1:0",  # Free tier
                body=json.dumps(request_body)
            )

            response_body = json.loads(response['body'].read())
            answer = response_body.get('generation', '').strip()

            return answer

        except Exception as e:
            return f"⚠️ LLM hatası: {str(e)}"

    def ask(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, any]:
        """
        Main RAG pipeline: search + generate with citations.

        Args:
            query: User query
            conversation_history: Previous conversation turns

        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.chunks:
            return {
                'answer': 'Lütfen önce bir PDF yükleyin.',
                'sources': [],
                'citations_valid': False
            }

        # Increment query count
        self.stats['total_queries'] += 1

        # Hybrid search
        search_results = self.hybrid_search(query, top_k=5)
        context_chunks = [chunk for idx, chunk in search_results]

        if not context_chunks:
            return {
                'answer': 'Bu konuda kaynaklarda ilgili bilgi bulunamadı.',
                'sources': [],
                'citations_valid': False
            }

        # Generate answer
        answer = self.generate_answer(query, context_chunks, conversation_history)

        # Extract and validate citations
        citation_ids = extract_citations(answer)
        validation = validate_citations(answer, len(context_chunks))

        # Format sources
        sources_formatted = format_sources_with_citations(
            context_chunks,
            citation_ids=citation_ids,
            max_display=5
        )

        return {
            'answer': answer,
            'sources': context_chunks,
            'sources_formatted': sources_formatted,
            'citation_ids': citation_ids,
            'citations_valid': validation['is_valid'],
            'validation_message': validation['message']
        }

    def get_stats(self) -> Dict[str, any]:
        """Return system statistics."""
        return {
            'document_name': self.document_name,
            'total_chunks': self.stats['total_chunks'],
            'total_queries': self.stats['total_queries'],
            'indexed': len(self.chunks) > 0
        }


if __name__ == "__main__":
    # Test the RAG system
    print("=== Medical RAG System Test ===\n")

    # Initialize
    rag = MedicalRAG()

    # Test would require actual PDF and AWS credentials
    print("System ready for PDF ingestion and queries.")
    print(f"Stats: {rag.get_stats()}")
