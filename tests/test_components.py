"""
test_components.py - Unit tests for individual system components
Tests DocumentProcessor, TextChunker, EmbeddingService, and LLMService classes
"""

import pytest
import asyncio
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import requests
from io import BytesIO

# Import components to test
from main import (
    DocumentProcessor, TextChunker, EmbeddingService, LLMService,
    DocumentChunk, QueryRetrievalSystem, Config
)

class TestDocumentProcessor:
    """Test DocumentProcessor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = DocumentProcessor()
    
    @pytest.mark.asyncio
    async def test_download_document_success(self):
        """Test successful document download"""
        mock_response = Mock()
        mock_response.content = b"PDF content here"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = Mock()
        
        with patch('requests.get', return_value=mock_response):
            content, format_ext = await self.processor.download_document("https://example.com/test.pdf")
            
            assert content == b"PDF content here"
            assert format_ext == ".pdf"
    
    @pytest.mark.asyncio
    async def test_download_document_failure(self):
        """Test document download failure"""
        with patch('requests.get', side_effect=requests.RequestException("Network error")):
            with pytest.raises(Exception):  # Should raise HTTPException, but we test the underlying exception
                await self.processor.download_document("https://invalid-url.com/test.pdf")
    
    def test_detect_format_pdf(self):
        """Test PDF format detection"""
        # Test with URL
        format_ext = self.processor._detect_format("https://example.com/document.pdf", "")
        assert format_ext == ".pdf"
        
        # Test with content-type
        format_ext = self.processor._detect_format("https://example.com/document", "application/pdf")
        assert format_ext == ".pdf"
    
    def test_detect_format_docx(self):
        """Test DOCX format detection"""
        format_ext = self.processor._detect_format("https://example.com/document.docx", "")
        assert format_ext == ".docx"
        
        format_ext = self.processor._detect_format("https://example.com/document", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert format_ext == ".docx"
    
    def test_detect_format_default(self):
        """Test default format detection"""
        format_ext = self.processor._detect_format("https://example.com/unknown", "application/octet-stream")
        assert format_ext == ".pdf"  # Default assumption
    
    @pytest.mark.asyncio
    async def test_extract_pdf_content(self):
        """Test PDF content extraction"""
        # Create a minimal PDF content for testing
        sample_pdf = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000174 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
267
%%EOF"""
        
        # Mock fitz.open to avoid actual PDF processing in tests
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_page = Mock()
            mock_page.get_text.return_value = "Hello World"
            mock_doc.page_count = 1
            mock_doc.__getitem__.return_value = mock_page
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            content = await self.processor._extract_pdf_content(sample_pdf)
            assert "Hello World" in content
    
    @pytest.mark.asyncio
    async def test_extract_docx_content(self):
        """Test DOCX content extraction"""
        with patch('docx.Document') as mock_docx:
            mock_doc = Mock()
            mock_paragraph = Mock()
            mock_paragraph.text = "This is a paragraph."
            mock_doc.paragraphs = [mock_paragraph]
            mock_docx.return_value = mock_doc
            
            # Create dummy DOCX content
            docx_content = b"PK\x03\x04"  # ZIP file signature (DOCX is a ZIP)
            
            content = await self.processor._extract_docx_content(docx_content)
            assert content == "This is a paragraph."
    
    @pytest.mark.asyncio
    async def test_extract_text_content(self):
        """Test plain text extraction"""
        text_content = b"This is plain text content."
        content = await self.processor.extract_content(text_content, ".txt")
        assert content == "This is plain text content."
    
    @pytest.mark.asyncio
    async def test_extract_email_content(self):
        """Test email content extraction"""
        email_content = b"""From: sender@example.com
To: recipient@example.com
Subject: Test Email

This is the email body content."""
        
        content = await self.processor._extract_email_content(email_content)
        assert "This is the email body content." in content

class TestTextChunker:
    """Test TextChunker class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.chunker = TextChunker(chunk_size=100, overlap=20)
    
    def test_chunk_text_single_chunk(self):
        """Test chunking text that fits in one chunk"""
        text = "This is a short text that should fit in one chunk."
        chunks = self.chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_id is not None
        assert len(chunks[0].chunk_id) == 8  # MD5 hash truncated to 8 chars
    
    def test_chunk_text_multiple_chunks(self):
        """Test chunking text that requires multiple chunks"""
        # Create text longer than chunk_size
        sentences = ["This is sentence number {}.".format(i) for i in range(1, 20)]
        text = " ".join(sentences)
        
        chunks = self.chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk.content) <= 120 for chunk in chunks)  # chunk_size + some buffer
        assert all(chunk.chunk_id is not None for chunk in chunks)
        assert all(len(chunk.chunk_id) == 8 for chunk in chunks)
    
    def test_chunk_metadata(self):
        """Test chunk metadata handling"""
        text = "This is a test text."
        metadata = {"document_url": "https://example.com/test.pdf", "page": 1}
        
        chunks = self.chunker.chunk_text(text, metadata)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.metadata["document_url"] == "https://example.com/test.pdf"
        assert chunk.metadata["page"] == 1
        assert "chunk_id" in chunk.metadata

class TestEmbeddingService:
    """Test EmbeddingService class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Mock SentenceTransformer to avoid loading actual model
        with patch('main.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(2, 384)  # Mock embeddings
            mock_st.return_value = mock_model
            
            self.embedding_service = EmbeddingService()
    
    @pytest.mark.asyncio
    async def test_embed_chunks(self):
        """Test chunk embedding generation"""
        chunks = [
            DocumentChunk("First chunk content", {}),
            DocumentChunk("Second chunk content", {})
        ]
        
        # Mock the model.encode method
        with patch.object(self.embedding_service.model, 'encode') as mock_encode:
            mock_encode.return_value = np.random.rand(2, 384)
            
            embedded_chunks = await self.embedding_service.embed_chunks(chunks)
            
            assert len(embedded_chunks) == 2
            assert all(chunk.embedding is not None for chunk in embedded_chunks)
            assert all(chunk.embedding.shape == (384,) for chunk in embedded_chunks)
    
    @pytest.mark.asyncio
    async def test_store_chunks_faiss(self):
        """Test storing chunks in FAISS"""
        # Ensure we're using FAISS
        self.embedding_service.use_faiss = True
        self.embedding_service.faiss_index = Mock()
        self.embedding_service.faiss_index.ntotal = 0
        self.embedding_service.faiss_index.add = Mock()
        
        chunks = [
            DocumentChunk("Test content", {}, embedding=np.random.rand(384)),
        ]
        
        with patch('faiss.normalize_L2') as mock_normalize:
            await self.embedding_service.store_chunks(chunks, "test-doc-id")
            
            mock_normalize.assert_called_once()
            self.embedding_service.faiss_index.add.assert_called_once()
            assert 0 in self.embedding_service.chunk_store
    
    @pytest.mark.asyncio
    async def test_search_similar_faiss(self):
        """Test similarity search with FAISS"""
        # Setup FAISS index mock
        self.embedding_service.use_faiss = True
        self.embedding_service.faiss_index = Mock()
        self.embedding_service.faiss_index.search.return_value = (
            np.array([[0.9, 0.8]]),  # scores
            np.array([[0, 1]])       # indices
        )
        
        # Setup chunk store
        self.embedding_service.chunk_store = {
            0: {"content": "First chunk", "document_id": "doc1"},
            1: {"content": "Second chunk", "document_id": "doc1"}
        }
        
        with patch.object(self.embedding_service.model, 'encode') as mock_encode, \
             patch('faiss.normalize_L2') as mock_normalize:
            
            mock_encode.return_value = np.random.rand(1, 384)
            
            results = await self.embedding_service.search_similar("test query", k=2)
            
            assert len(results) == 2
            assert results[0]["content"] == "First chunk"
            assert results[0]["score"] == 0.9
            assert results[1]["content"] == "Second chunk"
            assert results[1]["score"] == 0.8

class TestLLMService:
    """Test LLMService class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('main.OpenAI') as mock_openai:
            self.llm_service = LLMService()
            self.mock_client = Mock()
            mock_openai.return_value = self.mock_client
    
    @pytest.mark.asyncio
    async def test_generate_answer_success(self):
        """Test successful answer generation"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is the answer to your question."
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        query = "What is covered?"
        context_chunks = [
            {"content": "Coverage includes medical expenses.", "metadata": {"chunk_id": "chunk1"}},
            {"content": "Exclusions apply to pre-existing conditions.", "metadata": {"chunk_id": "chunk2"}}
        ]
        
        result = await self.llm_service.generate_answer(query, context_chunks)
        
        assert result["answer"] == "This is the answer to your question."
        assert "reasoning" in result
        assert "sources" in result
        assert "confidence" in result
        assert len(result["sources"]) <= 3  # Limited to top 3 sources
    
    @pytest.mark.asyncio
    async def test_generate_answer_openai_error(self):
        """Test answer generation with OpenAI error"""
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        query = "What is covered?"
        context_chunks = [{"content": "Test content", "metadata": {"chunk_id": "chunk1"}}]
        
        result = await self.llm_service.generate_answer(query, context_chunks)
        
        assert "error processing your query" in result["answer"].lower()
        assert result["confidence"] == 0.0
        assert result["sources"] == []
    
    def test_create_answer_prompt(self):
        """Test prompt creation"""
        query = "What is the waiting period?"
        context = "[Context 1]: The waiting period is 30 days."
        
        prompt = self.llm_service._create_answer_prompt(query, context)
        
        assert query in prompt
        assert context in prompt
        assert "Based on the following context" in prompt
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        # High confidence scenario
        high_conf_chunks = [
            {"score": 0.95, "content": "relevant content"},
            {"score": 0.90, "content": "also relevant"},
        ]
        confidence = self.llm_service._calculate_confidence(high_conf_chunks)
        assert confidence > 0.8
        
        # Low confidence scenario
        low_conf_chunks = [
            {"score": 0.3, "content": "barely relevant"},
            {"score": 0.2, "content": "not very relevant"},
        ]
        confidence = self.llm_service._calculate_confidence(low_conf_chunks)
        assert confidence < 0.5
        
        # Empty chunks
        confidence = self.llm_service._calculate_confidence([])
        assert confidence == 0.0

class TestQueryRetrievalSystem:
    """Test the main QueryRetrievalSystem orchestrator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('main.create_engine'), \
             patch('main.Base'), \
             patch('main.sessionmaker'):
            
            self.system = QueryRetrievalSystem()
    
    @pytest.mark.asyncio
    async def test_process_document_new(self):
        """Test processing a new document"""
        document_url = "https://example.com/test.pdf"
        
        # Mock all the dependencies
        with patch.object(self.system.doc_processor, 'download_document') as mock_download, \
             patch.object(self.system.doc_processor, 'extract_content') as mock_extract, \
             patch.object(self.system.chunker, 'chunk_text') as mock_chunk, \
             patch.object(self.system.embedding_service, 'embed_chunks') as mock_embed, \
             patch.object(self.system.embedding_service, 'store_chunks') as mock_store, \
             patch.object(self.system, 'SessionLocal') as mock_session:
            
            # Setup mocks
            mock_download.return_value = (b"PDF content", ".pdf")
            mock_extract.return_value = "Extracted text content"
            mock_chunk.return_value = [DocumentChunk("chunk1", {}), DocumentChunk("chunk2", {})]
            mock_embed.return_value = [DocumentChunk("chunk1", {}, np.random.rand(384))]
            
            # Mock database session
            mock_db = Mock()
            mock_db.query.return_value.filter.return_value.first.return_value = None  # No existing doc
            mock_session.return_value.__enter__.return_value = mock_db
            
            doc_id = await self.system.process_document(document_url)
            
            assert doc_id is not None
            assert len(doc_id) == 36  # UUID length
            
            # Verify method calls
            mock_download.assert_called_once_with(document_url)
            mock_extract.assert_called_once()
            mock_chunk.assert_called_once()
            mock_embed.assert_called_once()
            mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_document_existing(self):
        """Test processing an existing document"""
        document_url = "https://example.com/test.pdf"
        
        with patch.object(self.system, 'SessionLocal') as mock_session:
            # Mock existing document
            mock_doc = Mock()
            mock_doc.id = "existing-doc-id"
            mock_doc.processed = "completed"
            
            mock_db = Mock()
            mock_db.query.return_value.filter.return_value.first.return_value = mock_doc
            mock_session.return_value.__enter__.return_value = mock_db
            
            doc_id = await self.system.process_document(document_url)
            
            assert doc_id == "existing-doc-id"
    
    @pytest.mark.asyncio
    async def test_process_document_failure(self):
        """Test document processing failure"""
        document_url = "https://invalid-url.com/test.pdf"
        
        with patch.object(self.system.doc_processor, 'download_document') as mock_download, \
             patch.object(self.system, 'SessionLocal') as mock_session:
            
            mock_download.side_effect = Exception("Download failed")
            
            # Mock database session
            mock_db = Mock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            mock_session.return_value.__enter__.return_value = mock_db
            
            with pytest.raises(Exception):
                await self.system.process_document(document_url)
    
    @pytest.mark.asyncio
    async def test_process_queries_success(self):
        """Test successful query processing"""
        document_id = "test-doc-id"
        questions = ["What is covered?", "What are the limits?"]
        
        with patch.object(self.system.embedding_service, 'search_similar') as mock_search, \
             patch.object(self.system.llm_service, 'generate_answer') as mock_generate, \
             patch.object(self.system, 'SessionLocal') as mock_session:
            
            # Mock search results
            mock_search.return_value = [
                {"content": "Coverage details", "metadata": {"chunk_id": "chunk1"}},
                {"content": "More coverage info", "metadata": {"chunk_id": "chunk2"}}
            ]
            
            # Mock LLM responses
            mock_generate.side_effect = [
                {"answer": "Coverage includes medical expenses."},
                {"answer": "Limits are $100,000 per year."}
            ]
            
            # Mock database session
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            answers = await self.system.process_queries(document_id, questions)
            
            assert len(answers) == 2
            assert answers[0] == "Coverage includes medical expenses."
            assert answers[1] == "Limits are $100,000 per year."
            
            # Verify search was called for each question
            assert mock_search.call_count == 2
            assert mock_generate.call_count == 2
    
    @pytest.mark.asyncio 
    async def test_process_queries_partial_failure(self):
        """Test query processing with some failures"""
        document_id = "test-doc-id"
        questions = ["What is covered?", "Invalid question?"]
        
        with patch.object(self.system.embedding_service, 'search_similar') as mock_search, \
             patch.object(self.system.llm_service, 'generate_answer') as mock_generate, \
             patch.object(self.system, 'SessionLocal') as mock_session:
            
            # Mock search results
            mock_search.return_value = [{"content": "Test content", "metadata": {"chunk_id": "chunk1"}}]
            
            # Mock LLM responses - first succeeds, second fails
            mock_generate.side_effect = [
                {"answer": "Coverage includes medical expenses."},
                Exception("LLM processing failed")
            ]
            
            # Mock database session
            mock_db = Mock()
            mock_session.return_value.__enter__.return_value = mock_db
            
            answers = await self.system.process_queries(document_id, questions)
            
            assert len(answers) == 2
            assert answers[0] == "Coverage includes medical expenses."
            assert "Error processing query" in answers[1]

class TestDocumentChunk:
    """Test DocumentChunk dataclass"""
    
    def test_document_chunk_creation(self):
        """Test DocumentChunk creation and attributes"""
        content = "This is test content"
        metadata = {"page": 1, "section": "introduction"}
        embedding = np.random.rand(384)
        chunk_id = "test-chunk-id"
        
        chunk = DocumentChunk(
            content=content,
            metadata=metadata,
            embedding=embedding,
            chunk_id=chunk_id
        )
        
        assert chunk.content == content
        assert chunk.metadata == metadata
        assert np.array_equal(chunk.embedding, embedding)
        assert chunk.chunk_id == chunk_id
    
    def test_document_chunk_defaults(self):
        """Test DocumentChunk with default values"""
        chunk = DocumentChunk(content="Test content", metadata={})
        
        assert chunk.content == "Test content"
        assert chunk.metadata == {}
        assert chunk.embedding is None
        assert chunk.chunk_id == ""

class TestIntegrationScenarios:
    """Test integration scenarios between components"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_pdf_processing(self):
        """Test full pipeline from PDF to embeddings"""
        # This is a more comprehensive integration test
        with patch('main.SentenceTransformer') as mock_st, \
             patch('main.OpenAI') as mock_openai, \
             patch('main.create_engine'), \
             patch('main.Base'), \
             patch('main.sessionmaker'):
            
            # Setup mocks
            mock_st.return_value.encode.return_value = np.random.rand(2, 384)
            
            system = QueryRetrievalSystem()
            
            # Mock document processing chain
            with patch.object(system.doc_processor, 'download_document') as mock_download, \
                 patch.object(system.doc_processor, 'extract_content') as mock_extract:
                
                mock_download.return_value = (b"PDF content", ".pdf")
                mock_extract.return_value = "This is extracted PDF content with multiple sentences. It should be chunked properly."
                
                # Test the chunking and embedding pipeline
                content = await system.doc_processor.extract_content(b"PDF content", ".pdf")
                chunks = system.chunker.chunk_text(content)
                embedded_chunks = await system.embedding_service.embed_chunks(chunks)
                
                assert len(chunks) >= 1
                assert all(chunk.embedding is not None for chunk in embedded_chunks)
                assert all(chunk.embedding.shape == (384,) for chunk in embedded_chunks)
    
    @pytest.mark.asyncio
    async def test_end_to_end_query_flow(self):
        """Test end-to-end query processing flow"""
        with patch('main.SentenceTransformer') as mock_st, \
             patch('main.OpenAI') as mock_openai, \
             patch('main.create_engine'), \
             patch('main.Base'), \
             patch('main.sessionmaker'):
            
            # Setup embedding service mock
            mock_st.return_value.encode.return_value = np.random.rand(1, 384)
            
            # Setup LLM service mock
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Based on the policy, coverage includes medical expenses."
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            system = QueryRetrievalSystem()
            
            # Mock the search to return relevant chunks
            with patch.object(system.embedding_service, 'search_similar') as mock_search:
                mock_search.return_value = [
                    {
                        "content": "The policy covers medical expenses up to $100,000.",
                        "score": 0.9,
                        "metadata": {"chunk_id": "chunk1", "document_id": "doc1"}
                    }
                ]
                
                # Process a single query
                query = "What medical expenses are covered?"
                
                # Test similarity search
                similar_chunks = await system.embedding_service.search_similar(query)
                assert len(similar_chunks) == 1
                assert similar_chunks[0]["score"] == 0.9
                
                # Test answer generation
                result = await system.llm_service.generate_answer(query, similar_chunks)
                assert "coverage includes medical expenses" in result["answer"].lower()
                assert result["confidence"] > 0.0
                assert len(result["sources"]) == 1

class TestErrorHandling:
    """Test error handling scenarios"""
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """Test handling of network timeouts"""
        processor = DocumentProcessor()
        
        with patch('requests.get', side_effect=requests.Timeout("Request timed out")):
            with pytest.raises(Exception):
                await processor.download_document("https://slow-server.com/document.pdf")
    
    @pytest.mark.asyncio
    async def test_invalid_pdf_handling(self):
        """Test handling of corrupted PDF files"""
        processor = DocumentProcessor()
        
        # Invalid PDF content
        invalid_pdf = b"This is not a PDF file"
        
        with patch('fitz.open', side_effect=Exception("Invalid PDF")):
            with pytest.raises(Exception):
                await processor._extract_pdf_content(invalid_pdf)
    
    @pytest.mark.asyncio
    async def test_embedding_service_failure(self):
        """Test embedding service failure handling"""
        with patch('main.SentenceTransformer') as mock_st:
            mock_st.return_value.encode.side_effect = Exception("Model loading failed")
            
            embedding_service = EmbeddingService()
            chunks = [DocumentChunk("Test content", {})]
            
            with pytest.raises(Exception):
                await embedding_service.embed_chunks(chunks)
    
    @pytest.mark.asyncio
    async def test_database_connection_failure(self):
        """Test database connection failure handling"""
        with patch('main.create_engine', side_effect=Exception("Database connection failed")):
            with pytest.raises(Exception):
                QueryRetrievalSystem()

class TestPerformanceOptimizations:
    """Test performance-related functionality"""
    
    @pytest.mark.asyncio
    async def test_batch_embedding_efficiency(self):
        """Test that embeddings are generated in batches efficiently"""
        with patch('main.SentenceTransformer') as mock_st:
            # Mock to verify batch processing
            mock_st.return_value.encode.return_value = np.random.rand(100, 384)
            
            embedding_service = EmbeddingService()
            
            # Create 100 chunks
            chunks = [DocumentChunk(f"Content {i}", {}) for i in range(100)]
            
            embedded_chunks = await embedding_service.embed_chunks(chunks)
            
            # Verify all chunks got embeddings
            assert len(embedded_chunks) == 100
            assert all(chunk.embedding is not None for chunk in embedded_chunks)
            
            # Verify encode was called once (batch processing)
            mock_st.return_value.encode.assert_called_once()
    
    def test_chunk_size_optimization(self):
        """Test that chunk sizes are optimized for token efficiency"""
        chunker = TextChunker(chunk_size=512, overlap=50)
        
        # Create long text
        long_text = " ".join([f"This is sentence number {i}." for i in range(1, 200)])
        
        chunks = chunker.chunk_text(long_text)
        
        # Verify chunk sizes are within limits
        for chunk in chunks:
            assert len(chunk.content) <= 600  # chunk_size + reasonable buffer
        
        # Verify there's meaningful overlap (not tested directly but chunks should have some overlap)
        assert len(chunks) > 1  # Long text should be split
    
    @pytest.mark.asyncio
    async def test_vector_search_performance(self):
        """Test vector search performance characteristics"""
        with patch('main.SentenceTransformer') as mock_st:
            mock_st.return_value.encode.return_value = np.random.rand(1, 384)
            
            embedding_service = EmbeddingService()
            
            # Mock large vector index
            embedding_service.faiss_index = Mock()
            embedding_service.faiss_index.search.return_value = (
                np.array([[0.9, 0.8, 0.7, 0.6, 0.5]]),  # scores
                np.array([[0, 1, 2, 3, 4]])             # indices
            )
            
            # Setup chunk store with many items
            embedding_service.chunk_store = {
                i: {"content": f"Chunk {i}", "document_id": "doc1"}
                for i in range(1000)
            }
            
            # Search should return top-k results efficiently
            results = await embedding_service.search_similar("test query", k=5)
            
            assert len(results) == 5
            assert results[0]["score"] == 0.9  # Highest score first
            assert results[4]["score"] == 0.5  # Lowest score last

# Test fixtures and utilities
@pytest.fixture
def sample_document_chunks():
    """Create sample document chunks for testing"""
    return [
        DocumentChunk(
            content="This policy covers medical expenses for hospitalization.",
            metadata={"page": 1, "section": "coverage"},
            chunk_id="chunk1"
        ),
        DocumentChunk(
            content="The waiting period for pre-existing conditions is 36 months.",
            metadata={"page": 2, "section": "conditions"},
            chunk_id="chunk2"
        ),
        DocumentChunk(
            content="Premium payment grace period is 30 days.",
            metadata={"page": 3, "section": "payment"},
            chunk_id="chunk3"
        )
    ]

@pytest.fixture
def mock_embedding_vectors():
    """Create mock embedding vectors"""
    return np.random.rand(3, 384)

@pytest.fixture
def mock_llm_responses():
    """Create mock LLM responses"""
    return [
        "The policy covers medical expenses for hospitalization and outpatient care.",
        "There is a 36-month waiting period for pre-existing conditions.",
        "The grace period for premium payment is 30 days from the due date."
    ]

# Performance benchmarks (optional - for development)
class TestPerformanceBenchmarks:
    """Performance benchmark tests (run separately)"""
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_document_processing_speed(self):
        """Benchmark document processing speed"""
        import time
        
        with patch('main.SentenceTransformer') as mock_st, \
             patch('main.OpenAI'), \
             patch('main.create_engine'), \
             patch('main.Base'), \
             patch('main.sessionmaker'):
            
            mock_st.return_value.encode.return_value = np.random.rand(10, 384)
            
            system = QueryRetrievalSystem()
            
            # Mock document content (simulate medium-sized document)
            test_content = " ".join([f"This is paragraph {i} with relevant information." for i in range(100)])
            
            with patch.object(system.doc_processor, 'download_document', return_value=(b"content", ".txt")), \
                 patch.object(system.doc_processor, 'extract_content', return_value=test_content):
                
                start_time = time.time()
                
                # Process document
                chunks = system.chunker.chunk_text(test_content)
                embedded_chunks = await system.embedding_service.embed_chunks(chunks)
                
                processing_time = time.time() - start_time
                
                # Performance assertions (adjust based on requirements)
                assert processing_time < 5.0  # Should process in under 5 seconds
                assert len(embedded_chunks) > 0
                assert all(chunk.embedding is not None for chunk in embedded_chunks)
    
    @pytest.mark.slow  
    @pytest.mark.asyncio
    async def test_query_response_speed(self):
        """Benchmark query response speed"""
        import time
        
        with patch('main.SentenceTransformer') as mock_st, \
             patch('main.OpenAI') as mock_openai, \
             patch('main.create_engine'), \
             patch('main.Base'), \
             patch('main.sessionmaker'):
            
            # Setup fast mocks
            mock_st.return_value.encode.return_value = np.random.rand(1, 384)
            
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Quick response"
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            system = QueryRetrievalSystem()
            
            # Mock search results
            with patch.object(system.embedding_service, 'search_similar') as mock_search:
                mock_search.return_value = [
                    {"content": "Relevant content", "score": 0.9, "metadata": {"chunk_id": "chunk1"}}
                ]
                
                start_time = time.time()
                
                # Process query
                result = await system.llm_service.generate_answer("What is covered?", mock_search.return_value)
                
                response_time = time.time() - start_time
                
                # Performance assertions
                assert response_time < 2.0  # Should respond in under 2 seconds
                assert result["answer"] == "Quick response"

# Run tests
if __name__ == "__main__":
    # Run all tests except slow ones by default
    pytest.main([__file__, "-v", "-m", "not slow"])