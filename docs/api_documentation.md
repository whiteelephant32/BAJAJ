# HackRx 6.0 LLM-Powered Query-Retrieval System API Documentation

## Overview
This API provides intelligent document processing and query-answering capabilities for insurance, legal, HR, and compliance documents using LLM technology.

## Base URL
```
http://localhost:8000
```

## Authentication
All endpoints require Bearer token authentication:
```
Authorization: Bearer 5198af4a74b3f28046d225858ffe5010789c03ac0a414b41e5fa08533884a424
```

## Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-31T10:30:00.000Z",
  "version": "1.0.0"
}
```

### 2. Root Endpoint
**GET** `/`

Basic system information.

**Response:**
```json
{
  "message": "LLM-Powered Intelligent Query-Retrieval System",
  "status": "running"
}
```

### 3. Main HackRx Endpoint
**POST** `/hackrx/run`

Process a document and answer questions about it.

**Request Body:**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What are the waiting periods for coverage?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The grace period for premium payment is 30 days from the due date.",
    "The waiting period for pre-existing diseases is 48 months."
  ],
  "processing_time": 1250,
  "metadata": {
    "document_id": "uuid-here",
    "question_count": 2,
    "system_version": "1.0.0"
  }
}
```

### 4. Document Information
**GET** `/documents/{document_id}`

Get information about a processed document.

**Parameters:**
- `document_id` (path): UUID of the processed document

**Response:**
```json
{
  "id": "uuid-here",
  "url": "https://example.com/document.pdf",
  "processed": "completed",
  "metadata": {
    "format": ".pdf",
    "chunk_count": 45,
    "content_length": 15420
  },
  "created_at": "2025-07-31T10:30:00.000Z"
}
```

### 5. Document Queries
**GET** `/documents/{document_id}/queries`

Get all queries processed for a specific document.

**Parameters:**
- `document_id` (path): UUID of the processed document

**Response:**
```json
[
  {
    "id": "query-uuid",
    "query": "What is the grace period?",
    "response": {
      "answer": "The grace period is 30 days.",
      "reasoning": "Found in section 3.2 of the policy document",
      "sources": ["chunk-123", "chunk-456"],
      "confidence": 0.92
    },
    "processing_time": 450,
    "created_at": "2025-07-31T10:30:00.000Z"
  }
]
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Failed to download document: Invalid URL"
}
```

### 404 Not Found
```json
{
  "detail": "Document not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Processing failed: OpenAI API key not configured"
}
```

## Sample Usage

### Using curl
```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Process document with questions
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 5198af4a74b3f28046d225858ffe5010789c03ac0a414b41e5fa08533884a424" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "What are the waiting periods for coverage?"
    ]
  }'
```

### Using Python requests
```python
import requests

headers = {
    "Authorization": "Bearer 5198af4a74b3f28046d225858ffe5010789c03ac0a414b41e5fa08533884a424",
    "Content-Type": "application/json"
}

payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What are the waiting periods for coverage?"
    ]
}

response = requests.post(
    "http://localhost:8000/hackrx/run",
    headers=headers,
    json=payload
)

print(response.json())
```

## Supported Document Formats
- PDF (.pdf)
- Microsoft Word (.docx)
- Plain text (.txt)
- Email (.eml)

## Performance Considerations
- Document processing time depends on document size and complexity
- Larger documents are automatically chunked for optimal processing
- Vector embeddings are cached for repeated queries
- Average processing time: 200-500ms per question

## Rate Limits
- No explicit rate limits currently implemented
- Recommended: Max 10 concurrent requests
- Document processing is CPU-intensive

## Environment Variables Required
```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here  # Optional, FAISS used as fallback
PINECONE_ENV=us-east-1-aws  # Optional
DATABASE_URL=sqlite:///hackrx.db  # Optional, defaults to SQLite
```

## Testing
Use the provided test script to validate your installation:

```bash
# Run comprehensive test
python test_hackrx.py

# Quick health check
python test_hackrx.py health

# Quick test with 2 questions
python test_hackrx.py quick
```
