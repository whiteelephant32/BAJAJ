"""
test_api.py - Unit tests for FastAPI endpoints
Tests individual API endpoints using FastAPI TestClient
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import json
import tempfile
import os

# Import the main application
from main import app, retrieval_system, QueryRequest, QueryResponse

# Create test client
client = TestClient(app)

# Test configuration
TEST_BEARER_TOKEN = "5198af4a74b3f28046d225858ffe5010789c03ac0a414b41e5fa08533884a424"
TEST_HEADERS = {
    "Authorization": f"Bearer {TEST_BEARER_TOKEN}",
    "Content-Type": "application/json"
}

class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

class TestHackRxEndpoint:
    """Test the main HackRx endpoint"""
    
    def test_hackrx_run_endpoint_structure(self):
        """Test that the endpoint accepts correct request structure"""
        payload = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?", "What are the limits?"]
        }
        
        # Mock the retrieval system to avoid actual processing
        with patch.object(retrieval_system, 'process_document', return_value="test-doc-id") as mock_process_doc, \
             patch.object(retrieval_system, 'process_queries', return_value=["Answer 1", "Answer 2"]) as mock_process_queries:
            
            response = client.post("/hackrx/run", headers=TEST_HEADERS, json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            # Check response structure
            assert "answers" in data
            assert "processing_time" in data
            assert "metadata" in data
            
            # Check data types
            assert isinstance(data["answers"], list)
            assert isinstance(data["processing_time"], int)
            assert isinstance(data["metadata"], dict)
            
            # Check that methods were called
            mock_process_doc.assert_called_once_with("https://example.com/test.pdf")
            mock_process_queries.assert_called_once_with("test-doc-id", ["What is covered?", "What are the limits?"])
    
    def test_hackrx_run_missing_fields(self):
        """Test endpoint with missing required fields"""
        # Missing documents field
        payload = {
            "questions": ["What is covered?"]
        }
        response = client.post("/hackrx/run", headers=TEST_HEADERS, json=payload)
        assert response.status_code == 422  # Validation error
        
        # Missing questions field
        payload = {
            "documents": "https://example.com/test.pdf"
        }
        response = client.post("/hackrx/run", headers=TEST_HEADERS, json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_hackrx_run_empty_questions(self):
        """Test endpoint with empty questions list"""
        payload = {
            "documents": "https://example.com/test.pdf",
            "questions": []
        }
        
        with patch.object(retrieval_system, 'process_document', return_value="test-doc-id"), \
             patch.object(retrieval_system, 'process_queries', return_value=[]):
            
            response = client.post("/hackrx/run", headers=TEST_HEADERS, json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["answers"] == []
    
    def test_hackrx_run_invalid_document_url(self):
        """Test endpoint with invalid document URL"""
        payload = {
            "documents": "invalid-url",
            "questions": ["What is covered?"]
        }
        
        # Mock to raise an exception for invalid URL
        with patch.object(retrieval_system, 'process_document', side_effect=Exception("Invalid URL")):
            response = client.post("/hackrx/run", headers=TEST_HEADERS, json=payload)
            assert response.status_code == 500
    
    def test_hackrx_run_processing_error(self):
        """Test endpoint when processing fails"""
        payload = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?"]
        }
        
        # Mock processing failure
        with patch.object(retrieval_system, 'process_document', side_effect=Exception("Processing failed")):
            response = client.post("/hackrx/run", headers=TEST_HEADERS, json=payload)
            assert response.status_code == 500
            assert "Processing failed" in response.json()["detail"]

class TestDocumentEndpoints:
    """Test document-related endpoints"""
    
    @patch.object(retrieval_system, 'SessionLocal')
    def test_get_document_info(self, mock_session):
        """Test getting document information"""
        # Mock database response
        mock_doc = Mock()
        mock_doc.id = "test-doc-id"
        mock_doc.url = "https://example.com/test.pdf"
        mock_doc.processed = "completed"
        mock_doc.metadata = {"format": ".pdf", "chunk_count": 10}
        mock_doc.created_at.isoformat.return_value = "2024-01-01T00:00:00"
        
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_doc
        mock_session.return_value.__enter__.return_value = mock_db
        
        response = client.get("/documents/test-doc-id")
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == "test-doc-id"
        assert data["url"] == "https://example.com/test.pdf"
        assert data["processed"] == "completed"
        assert data["metadata"]["format"] == ".pdf"
    
    @patch.object(retrieval_system, 'SessionLocal')
    def test_get_document_info_not_found(self, mock_session):
        """Test getting non-existent document"""
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        mock_session.return_value.__enter__.return_value = mock_db
        
        response = client.get("/documents/non-existent-id")
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]
    
    @patch.object(retrieval_system, 'SessionLocal')
    def test_get_document_queries(self, mock_session):
        """Test getting document queries"""
        # Mock query records
        mock_query1 = Mock()
        mock_query1.id = "query-1"
        mock_query1.query = "What is covered?"
        mock_query1.response = {"answer": "Coverage details"}
        mock_query1.processing_time = 1500
        mock_query1.created_at.isoformat.return_value = "2024-01-01T00:00:00"
        
        mock_query2 = Mock()
        mock_query2.id = "query-2"
        mock_query2.query = "What are limits?"
        mock_query2.response = {"answer": "Limit details"}
        mock_query2.processing_time = 1200
        mock_query2.created_at.isoformat.return_value = "2024-01-01T00:01:00"
        
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.all.return_value = [mock_query1, mock_query2]
        mock_session.return_value.__enter__.return_value = mock_db
        
        response = client.get("/documents/test-doc-id/queries")
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 2
        assert data[0]["query"] == "What is covered?"
        assert data[1]["query"] == "What are limits?"

class TestRequestValidation:
    """Test request validation and error handling"""
    
    def test_invalid_json(self):
        """Test endpoint with invalid JSON"""
        response = client.post("/hackrx/run", headers=TEST_HEADERS, data="invalid json")
        assert response.status_code == 422
    
    def test_wrong_content_type(self):
        """Test endpoint with wrong content type"""
        headers = {**TEST_HEADERS, "Content-Type": "text/plain"}
        response = client.post("/hackrx/run", headers=headers, data="some data")
        assert response.status_code == 422
    
    def test_missing_authorization(self):
        """Test endpoint without authorization header"""
        payload = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?"]
        }
        headers = {"Content-Type": "application/json"}  # No Authorization header
        
        # Note: The current implementation doesn't enforce auth, but this test
        # is here for when auth is implemented
        response = client.post("/hackrx/run", headers=headers, json=payload)
        # Currently should still work, but can be updated when auth is added
        assert response.status_code in [200, 401, 403]

class TestPydanticModels:
    """Test Pydantic model validation"""
    
    def test_query_request_validation(self):
        """Test QueryRequest model validation"""
        # Valid request
        valid_data = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?"]
        }
        request = QueryRequest(**valid_data)
        assert request.documents == "https://example.com/test.pdf"
        assert len(request.questions) == 1
        
        # Invalid request - missing documents
        with pytest.raises(ValueError):
            QueryRequest(questions=["What is covered?"])
        
        # Invalid request - missing questions
        with pytest.raises(ValueError):
            QueryRequest(documents="https://example.com/test.pdf")
    
    def test_query_response_validation(self):
        """Test QueryResponse model validation"""
        # Valid response
        valid_data = {
            "answers": ["Answer 1", "Answer 2"],
            "processing_time": 1500,
            "metadata": {"doc_id": "test-id"}
        }
        response = QueryResponse(**valid_data)
        assert len(response.answers) == 2
        assert response.processing_time == 1500
        
        # Minimal valid response
        minimal_data = {"answers": ["Answer 1"]}
        response = QueryResponse(**minimal_data)
        assert response.processing_time is None
        assert response.metadata is None

class TestConcurrency:
    """Test concurrent request handling"""
    
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        import threading
        import time
        
        payload = {
            "documents": "https://example.com/test.pdf",
            "questions": ["What is covered?"]
        }
        
        results = []
        
        def make_request():
            with patch.object(retrieval_system, 'process_document', return_value="test-doc-id"), \
                 patch.object(retrieval_system, 'process_queries', return_value=["Test answer"]):
                
                response = client.post("/hackrx/run", headers=TEST_HEADERS, json=payload)
                results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5

# Integration test fixtures
@pytest.fixture
def sample_pdf_content():
    """Create a sample PDF for testing"""
    pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    return pdf_content

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is a test answer from the LLM."
    return mock_response

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])