# API Documentation

## Overview

Perplexa is a Retrieval Augmented Generation (RAG) application that combines web search, document retrieval, and multiple LLM APIs to provide intelligent answers to user queries. This document provides comprehensive API documentation, performance metrics, and operational details.

---

## Architecture Overview (30-Second Sketch)

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY INPUT                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA RETRIEVAL LAYER                          │
│  ┌────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  SerpAPI       │  │  BeautifulSoup   │  │  Content        │ │
│  │  Google Search │→ │  Web Scraping    │→ │  Extraction     │ │
│  └────────────────┘  └──────────────────┘  └─────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE/EMBEDDING LAYER                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  SentenceTransformer (all-MiniLM-L6-v2)                    │ │
│  │  • Query Embedding                                         │ │
│  │  • Document Embeddings                                     │ │
│  │  • Cosine Similarity Calculation                           │ │
│  │  • Top-K Document Selection (k=3)                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GENERATION LAYER                              │
│  ┌────────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  Gemini    │  │ Mistral  │  │ Command R+   │  │OpenRouter │ │
│  │    API     │  │   API    │  │  (Cohere)    │  │    API    │ │
│  └────────────┘  └──────────┘  └──────────────┘  └───────────┘ │
│       │              │               │                  │        │
│       └──────────────┴───────────────┴──────────────────┘        │
│                           │                                      │
│                      RAG Prompt                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SERVING LAYER                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Streamlit Web Application                                 │ │
│  │  • User Authentication (Auth0)                             │ │
│  │  • Chat Interface                                          │ │
│  │  • MongoDB (User & Chat Storage)                           │ │
│  │  • Session Management                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
                  ┌───────────────────┐
                  │  USER RESPONSE    │
                  └───────────────────┘
```

---

## Performance Metrics

### Latency Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **P50 Latency** | ~8-12 seconds | Typical query processing time |
| **P95 Latency** | ~20-25 seconds | Including web scraping & API calls |
| **P99 Latency** | ~30-35 seconds | Complex queries with slow web sources |
| **Embedding Generation** | ~100-200ms | Cached model, batch processing |
| **Document Retrieval** | ~3-8 seconds | SerpAPI + web scraping (5 URLs) |
| **LLM API Call** | ~2-10 seconds | Varies by model and prompt size |

### Cost Per Request

| Component | Cost | Details |
|-----------|------|---------|
| **SerpAPI** | $0.002-0.005 | Per search query (5 results) |
| **Gemini API** | $0.001-0.003 | Per 1K tokens (~2-3K tokens/query) |
| **Mistral API** | $0.002-0.006 | Per 1K tokens |
| **Command R+ (Cohere)** | $0.003-0.008 | Per 1K tokens |
| **OpenRouter** | $0.000-0.005 | Varies by model (free tier available) |
| **MongoDB Atlas** | $0.08/hr | Shared cluster (~$60/month) |
| **Streamlit Cloud** | Free | Community deployment |
| **Total Est. Cost/Request** | **$0.005-0.015** | Average across all components |

### Quality/Evaluation Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Answer Relevance** | >85% | Manual evaluation of top-K document selection |
| **Citation Accuracy** | >90% | Verification of source URLs in responses |
| **Model Uptime** | >99% | API availability monitoring |
| **User Satisfaction** | >4.0/5.0 | User feedback (when implemented) |
| **Context Retrieval Precision** | >80% | Cosine similarity threshold (>0.5) |

---

## Cloud Infrastructure & Tools

### Hosting & Deployment
- **Application Host**: Streamlit Cloud (Community Tier)
- **Deployment Method**: GitHub webhook-triggered auto-deployment
- **SSL/TLS**: Automatic HTTPS via Streamlit Cloud

### Database
- **Database**: MongoDB Atlas (Cloud)
- **Collections**: 
  - `users`: User profiles and authentication data
  - `chats`: Chat history and session data
- **Connection**: TLS-enabled with connection string in environment variables

### External APIs & Services

#### 1. SerpAPI (Search)
- **Purpose**: Google search results retrieval
- **Rate Limit**: 100 searches/month (free), scalable plans
- **Endpoint**: `https://serpapi.com/search`

#### 2. LLM APIs

**Gemini API**
- **Provider**: Google
- **Model**: gemini-1.5-flash-latest
- **Endpoint**: `https://generativelanguage.googleapis.com/v1beta/models/`
- **Rate Limit**: 60 requests/minute

**Mistral API**
- **Provider**: Mistral AI
- **Model**: mistral-small-latest
- **Endpoint**: `https://api.mistral.ai/v1/chat/completions`
- **Rate Limit**: 100 requests/minute

**Cohere (Command R+)**
- **Provider**: Cohere
- **Model**: command-r-plus
- **SDK**: Cohere Python SDK v2
- **Rate Limit**: 100 requests/minute

**OpenRouter**
- **Provider**: OpenRouter (Multi-model gateway)
- **Models**: 
  - deepseek/deepseek-r1-zero:free
  - microsoft/phi-3-medium-128k-instruct:free
  - nvidia/llama-3.1-nemotron-70b-instruct:free
  - meta-llama/llama-3.3-70b-instruct:free
  - qwen/qwq-32b:free
- **Endpoint**: `https://openrouter.ai/api/v1`
- **Rate Limit**: Varies by model

#### 3. Hugging Face
- **Purpose**: SentenceTransformer model hosting
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Token Required**: For private models (optional for public)

#### 4. Authentication
- **Provider**: Auth0 (via Streamlit)
- **Method**: OAuth 2.0
- **Session Management**: Streamlit session state

### Dependencies & Libraries

```
Core Framework:
- streamlit==1.42.0          # Web application framework
- Authlib==1.5.1             # Authentication library

Database:
- pymongo[srv]               # MongoDB driver

NLP & ML:
- sentence-transformers==2.7.0  # Embedding generation
- transformers==4.49.0          # Base transformers library
- scikit-learn                  # Cosine similarity calculation
- numpy                         # Numerical operations

Web Scraping:
- requests                      # HTTP client
- beautifulsoup4                # HTML parsing
- google-search-results         # SerpAPI client
- serpapi                       # Alternative SerpAPI client

LLM APIs:
- openai                        # OpenAI-compatible API client
- cohere                        # Cohere API client

Configuration:
- python-dotenv                 # Environment variable management
```

---

## CI/CD & MLOps

### Current CI/CD Setup

**Version Control**: GitHub
- Repository: https://github.com/sayan112207/Perplexa
- Branching Strategy: Main branch with feature branches

**Continuous Integration**
- **Dependency Review**: GitHub Actions workflow (`.github/workflows/dependency-review.yml`)
  - Scans for vulnerable dependencies on pull requests
  - Automated security checks

**Continuous Deployment**
- **Trigger**: Git push to main branch
- **Process**: 
  1. Developer pushes code to GitHub
  2. GitHub webhook notifies Streamlit Cloud
  3. Streamlit Cloud pulls latest commit
  4. Automatic build and deployment
  5. HTTPS-enabled app goes live
- **Rollback**: Manual revert via GitHub + redeploy

### MLOps Considerations

#### Model Management
- **Embedding Model**: Cached via `@st.cache_resource` decorator
  - Loaded once per application instance
  - Persists across user sessions
  - Reduces latency and memory usage

#### Monitoring & Observability
- **Application Logs**: Streamlit Cloud logs
- **Error Tracking**: Try-catch blocks with Streamlit error displays
- **API Monitoring**: Manual monitoring of API responses

#### Future MLOps Enhancements
- [ ] Implement structured logging (e.g., Python `logging` module)
- [ ] Add API latency tracking and metrics dashboard
- [ ] Implement A/B testing for different models
- [ ] Set up automated model version updates
- [ ] Add user feedback loop for answer quality
- [ ] Implement caching layer for repeated queries
- [ ] Set up alerts for API failures or high latency

### Testing Infrastructure

**Load Testing**
- Tool: Locust (defined in `tests/lucustfile.py`)
- Usage: `locust -f tests/locust_test.py --host=http://localhost:5500`
- Purpose: Simulate concurrent users and measure performance

**Unit Testing**
- Test files: `tests/test_app.py`, `tests/test_ui.py`
- Framework: (To be determined based on test file contents)

---

## API Endpoints

### Internal Application Functions

While Perplexa is primarily a Streamlit application without exposed REST endpoints, the following internal functions serve as the application's "API":

#### 1. **Document Retrieval**

```python
def aggregate_documents(query: str) -> tuple[list[dict], list[str]]
```

**Purpose**: Fetch and aggregate documents from web search results

**Parameters**:
- `query` (string): User search query

**Returns**:
- Tuple containing:
  - `documents`: List of dicts with `url` and `content` keys
  - `fetched_urls`: List of fetched URLs for citation

**Process**:
1. Calls SerpAPI with query
2. Extracts top 5 URLs
3. Scrapes content from each URL using BeautifulSoup
4. Falls back to snippet if scraping fails

**Example Flow**:
```python
query = "What is machine learning?"
documents, urls = aggregate_documents(query)
# documents: [{"url": "...", "content": "..."}, ...]
# urls: ["https://...", "https://...", ...]
```

---

#### 2. **Embedding & Similarity Calculation**

```python
def get_top_k_documents(query: str, documents: list[dict], k: int = 3) -> dict
```

**Purpose**: Select top-K most relevant documents using semantic similarity

**Parameters**:
- `query` (string): User query
- `documents` (list): List of document dicts
- `k` (int): Number of top documents to return (default: 3)

**Returns**:
- Dict with `context` key containing concatenated top documents

**Process**:
1. Generate embeddings for all documents
2. Generate embedding for query
3. Calculate cosine similarity
4. Sort and select top K documents
5. Concatenate content

**Example**:
```python
top_docs = get_top_k_documents(query, documents, k=3)
# top_docs: {"context": "Document 1 content\n\nDocument 2 content\n\n..."}
```

---

#### 3. **RAG Prompt Construction**

```python
def build_rag_prompt(query: str, context: str, references: list[str]) -> str
```

**Purpose**: Build structured prompt for LLM with context and citations

**Parameters**:
- `query` (string): User query
- `context` (string): Concatenated top documents
- `references` (list): List of source URLs

**Returns**:
- Formatted prompt string

**Prompt Structure**:
```
You are an expert assistant. Using the following retrieved context, 
provide a detailed explanation for the query: '{query}'.

Context:
{context}

Your response should include:
1. An introduction defining the topic.
2. Key points in bullet format.
3. A references section listing the source URLs.

References:
- {url1}
- {url2}
- {url3}

Make sure your answer is clear, concise, and accessible to beginners.
```

---

#### 4. **LLM API Calls**

**4a. Gemini API**

```python
def call_gemini_api(prompt: str) -> str
```

**API Details**:
- **Endpoint**: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent`
- **Method**: POST
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "contents": [
    {
      "parts": [
        {"text": "prompt_text"}
      ]
    }
  ]
}
```
- **Response**: JSON with nested structure
- **Error Handling**: Returns error message on failure

---

**4b. Mistral API**

```python
def call_mistral_api(prompt: str) -> str
```

**API Details**:
- **Endpoint**: `https://api.mistral.ai/v1/chat/completions`
- **Method**: POST
- **Headers**: 
  - `Authorization: Bearer {MISTRAL_API_KEY}`
  - `Content-Type: application/json`
- **Request Body**:
```json
{
  "model": "mistral-small-latest",
  "messages": [
    {"role": "user", "content": "prompt_text"}
  ]
}
```
- **Response**: Standard OpenAI-compatible format

---

**4c. Command R+ (Cohere)**

```python
def call_cmd_r_plus(prompt: str) -> str
```

**API Details**:
- Uses Cohere Python SDK v2
- **Model**: command-r-plus
- **Method**: `client.chat()`
- **Parameters**:
```python
{
  "model": "command-r-plus",
  "messages": [{"role": "user", "content": prompt}]
}
```

---

**4d. OpenRouter (Multi-Model)**

```python
def call_openrouter_api(prompt: str, model_choice: str) -> str
```

**API Details**:
- **Endpoint**: `https://openrouter.ai/api/v1`
- **Method**: Uses OpenAI-compatible client
- **Model Mapping**:
  - Deepseek R1 → `deepseek/deepseek-r1-zero:free`
  - Phi 3 → `microsoft/phi-3-medium-128k-instruct:free`
  - Nemotron → `nvidia/llama-3.1-nemotron-70b-instruct:free`
  - Meta Llama → `meta-llama/llama-3.3-70b-instruct:free`
  - Qwen 32B → `qwen/qwq-32b:free`

---

#### 5. **Database Operations**

**5a. User Management**

```python
def save_user_to_mongo(user: dict) -> None
```

**Purpose**: Save or update user profile in MongoDB

**Parameters**:
- `user`: Dict with keys: `name`, `email`, `picture`

**Database Operation**: `upsert` (update or insert)

---

**5b. Chat History**

```python
def load_chats_from_mongo(user_email: str) -> list[dict]
```

**Purpose**: Load all chat sessions for a user

**Returns**: List of chat documents

---

```python
def save_chat_to_mongo(user_email: str, chat_id: str, 
                       chat_title: str, messages: list) -> str
```

**Purpose**: Save or update chat session

**Returns**: Chat ID (string)

---

```python
def delete_chat_from_mongo(chat_id: str) -> None
```

**Purpose**: Delete a specific chat session

---

#### 6. **People Also Ask**

```python
def generate_people_also_ask(query: str) -> list[str]
```

**Purpose**: Fetch related questions from Google search results

**Returns**: List of related question strings

**Source**: SerpAPI's `related_questions` field

---

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the project root with the following:

```bash
# Database
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/database

# Search API
SERPAPI_API_KEY=your_serpapi_key_here

# LLM APIs
GEMINI_API_KEY=your_gemini_api_key
MISTRAL_API_KEY=your_mistral_api_key
COMMAND_R_PLUS=your_cohere_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional: Hugging Face (for private models)
HF_API_KEY=your_huggingface_token
```

### Streamlit Cloud Configuration

For deployment on Streamlit Cloud:

1. Go to app settings → Secrets
2. Add all environment variables in TOML format:

```toml
MONGO_URI = "mongodb+srv://..."
SERPAPI_API_KEY = "..."
GEMINI_API_KEY = "..."
MISTRAL_API_KEY = "..."
COMMAND_R_PLUS = "..."
OPENROUTER_API_KEY = "..."
HF_API_KEY = "..."
```

---

## Postman & API Testing Alternatives

While Perplexa doesn't expose traditional REST APIs, here are testing approaches:

### 1. **Locust Load Testing** (Implemented)

**Location**: `tests/lucustfile.py`

**Usage**:
```bash
# Start Streamlit app
streamlit run bot.py

# Run Locust in another terminal
locust -f tests/lucustfile.py --host=http://localhost:8501
```

**What it tests**:
- Concurrent user simulation
- Response time measurement
- Throughput analysis

---

### 2. **Unit Testing with pytest**

**Recommended Structure**:
```python
# tests/test_api_functions.py
import pytest
from app import (
    get_top_k_documents,
    build_rag_prompt,
    compute_embeddings
)

def test_top_k_documents():
    documents = [
        {"url": "url1", "content": "Machine learning content"},
        {"url": "url2", "content": "Deep learning content"},
        {"url": "url3", "content": "Unrelated content"}
    ]
    query = "What is machine learning?"
    result = get_top_k_documents(query, documents, k=2)
    assert "context" in result
    assert len(result["context"]) > 0

def test_build_rag_prompt():
    query = "Test query"
    context = "Test context"
    refs = ["http://url1.com", "http://url2.com"]
    prompt = build_rag_prompt(query, context, refs)
    assert query in prompt
    assert context in prompt
    assert refs[0] in prompt
```

---

### 3. **Manual API Testing**

**Test External APIs Directly**:

```bash
# Test Gemini API
curl -X POST \
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{
      "parts": [{"text": "Explain machine learning"}]
    }]
  }'

# Test Mistral API
curl -X POST \
  "https://api.mistral.ai/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-small-latest",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Test SerpAPI
curl "https://serpapi.com/search?q=machine+learning&api_key=YOUR_KEY"
```

---

### 4. **Streamlit Testing**

Use Streamlit's built-in testing utilities:

```python
# tests/test_ui.py
from streamlit.testing.v1 import AppTest

def test_app_loads():
    at = AppTest.from_file("bot.py")
    at.run()
    assert not at.exception
```

---

## Postmortem: Common Issues & Fixes

### Issue 1: Model Loading Timeout

**Symptom**: Application hangs on startup, embedding model fails to load

**Root Cause**: Hugging Face model download timeout or memory limits

**Fix Applied**:
1. Implemented `@st.cache_resource` decorator for model caching
2. Added timeout handling in model loading
3. Set `token=HF_API_KEY` for authenticated downloads

**Prevention**: 
- Monitor Streamlit Cloud memory usage
- Consider using smaller embedding models for constrained environments

---

### Issue 2: API Rate Limiting

**Symptom**: Intermittent errors when calling LLM APIs during high traffic

**Root Cause**: Exceeded free tier rate limits on various APIs

**Fix Applied**:
1. Added try-catch blocks around all API calls
2. Display user-friendly error messages
3. Implemented fallback to alternative models

**Prevention**:
- Implement request queuing
- Add exponential backoff retry logic
- Monitor API usage dashboards

---

### Issue 3: MongoDB Connection Failures

**Symptom**: `Connection failed` errors, users unable to save chats

**Root Cause**: 
- TLS certificate validation issues
- Connection string misconfiguration

**Fix Applied**:
```python
client = MongoClient(
    MONGO_URI, 
    tls=True, 
    tlsAllowInvalidCertificates=True  # For self-signed certs
)
```

**Prevention**:
- Use proper SSL certificates in production
- Implement connection pooling
- Add health check endpoint for database

---

### Issue 4: Web Scraping Timeouts

**Symptom**: `aggregate_documents()` takes too long, some URLs timeout

**Root Cause**: Slow or unresponsive websites in search results

**Fix Applied**:
```python
response = requests.get(url, timeout=5)  # 5-second timeout
```

**Fallback**: Use snippet from SerpAPI if scraping fails

**Prevention**:
- Implement async scraping with concurrent requests
- Add URL blacklist for known slow sites
- Increase timeout for slow networks

---

### Issue 5: Torch Runtime Error

**Symptom**: `RuntimeError: Tried to instantiate class '__path__._path'`

**Root Cause**: PyTorch internal issue with certain operations

**Fix Applied**:
```python
import torch
torch.classes.__path__ = []  # Workaround for torch bug
```

**Status**: Temporary fix, monitor PyTorch updates for permanent solution

---

### Issue 6: Streamlit Session State Conflicts

**Symptom**: Chat history gets corrupted, messages duplicate

**Root Cause**: Multiple reruns causing race conditions in session state

**Fix Applied**:
1. Implemented `is_same_chat()` comparison function
2. Structured message processing to avoid duplicate appends
3. Used `st.rerun()` strategically

**Prevention**:
- Minimize reruns
- Use session state flags to track processing status

---

## Troubleshooting Guide

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Commands

```bash
# Check Streamlit version
streamlit --version

# Run with verbose output
streamlit run bot.py --logger.level=debug

# Clear Streamlit cache
streamlit cache clear

# Test MongoDB connection
python -c "from pymongo import MongoClient; print(MongoClient('YOUR_URI').server_info())"

# Test API keys
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Keys loaded:', bool(os.getenv('GEMINI_API_KEY')))"
```

---

## Security Considerations

### API Key Management
- ✅ Keys stored in `.env` file (not committed to Git)
- ✅ Streamlit Cloud secrets for production
- ⚠️ Consider using key rotation policies
- ⚠️ Implement rate limiting per user

### Authentication
- ✅ Auth0 integration via Streamlit
- ✅ User data stored securely in MongoDB
- ⚠️ Implement session timeout
- ⚠️ Add CSRF protection for future API endpoints

### Data Privacy
- ✅ User queries not logged externally
- ✅ MongoDB connection encrypted (TLS)
- ⚠️ Add data retention policies
- ⚠️ Implement GDPR compliance measures

---

## Future Enhancements

### Performance Optimization
- [ ] Implement Redis caching for repeated queries
- [ ] Add CDN for static assets
- [ ] Use connection pooling for database
- [ ] Implement async API calls

### Features
- [ ] Export chat history to PDF
- [ ] Voice input support
- [ ] Multi-language support
- [ ] Custom embedding model fine-tuning

### Monitoring & Ops
- [ ] Set up Prometheus metrics
- [ ] Add Grafana dashboards
- [ ] Implement ELK stack for log aggregation
- [ ] Set up PagerDuty alerts

---

## Support & Contact

For issues, questions, or contributions:
- **GitHub**: https://github.com/sayan112207/Perplexa
- **Issues**: https://github.com/sayan112207/Perplexa/issues

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**Maintained By**: Perplexa Development Team
