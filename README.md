# Perplexa <img src="https://github.com/sayan112207/Perplexa/blob/main/perplexa_logo.png?raw=true" alt="Perplexa Logo" width="20"/>

Perplexa is an AI-powered search and chat application built using Streamlit, designed to intelligently answer user queries by leveraging retrieval-augmented generation (RAG). The app integrates multiple external APIs, performs real-time web scraping, calculates document-query similarity via sentence embeddings, and stores user data and chat history in a MongoDB database.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture & Performance](#architecture--performance)
3. [Workflow and Functionality](#workflow-and-functionality)
   - [User Authentication](#user-authentication)
   - [Database Integration](#database-integration)
   - [Embedding Model & Caching](#embedding-model--caching)
   - [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
   - [Document Retrieval & top_k_documents](#document-retrieval--top_k_documents)
   - [Model Selection and Supported Models](#model-selection-and-supported-models)
4. [Deployment Process](#deployment-process)
5. [Installation and Environment Variables](#installation-and-environment-variables)
6. [How It Works: Detailed Code Walkthrough](#how-it-works-detailed-code-walkthrough)
7. [API Documentation](#api-documentation)
8. [Contributing and License](#contributing-and-license)

---

## Overview

Perplexa uses a combination of natural language processing (NLP), web scraping, and cloud-based generative APIs to answer queries from users. It leverages a retrieval-augmented generation (RAG) approach – using web search results as context – to provide highly relevant responses and includes options to choose between several generative models.

---

## Architecture & Performance

### Quick Architecture Overview (30 Seconds)

```
USER QUERY → [Search (SerpAPI) + Web Scraping] → [Embedding + Top-K Selection] 
→ [LLM Generation (Gemini/Mistral/etc.)] → [Streamlit UI + MongoDB Storage] → RESPONSE
```

**Data Flow:**
1. **Retrieval**: User query triggers Google search via SerpAPI → Fetch top 5 URLs → Extract content with BeautifulSoup
2. **Inference**: Generate embeddings using SentenceTransformer (all-MiniLM-L6-v2) → Calculate cosine similarity → Select top 3 most relevant documents
3. **Serving**: Build RAG prompt with context → Call selected LLM API (Gemini/Mistral/Command R+/OpenRouter) → Display response in Streamlit UI

### Key Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **P95 Latency** | ~20-25s | Full query processing including web scraping |
| **Cost/Request** | $0.005-0.015 | SerpAPI + LLM API + MongoDB costs |
| **Context Retrieval** | Top-3 of 5 docs | Cosine similarity-based selection |
| **Embedding Speed** | ~100-200ms | Cached SentenceTransformer model |

### Infrastructure & Tools

**Cloud Services:**
- **Hosting**: Streamlit Cloud (webhook-based auto-deployment)
- **Database**: MongoDB Atlas (TLS-enabled)
- **Authentication**: Auth0 (via Streamlit)
- **APIs**: SerpAPI, Google Gemini, Mistral AI, Cohere, OpenRouter, Hugging Face

**CI/CD & MLOps:**
- **Version Control**: GitHub with automatic deployment on push
- **Dependency Scanning**: GitHub Actions (dependency-review.yml)
- **Model Caching**: `@st.cache_resource` for SentenceTransformer
- **Load Testing**: Locust (see `tests/lucustfile.py`)
- **Monitoring**: Streamlit Cloud logs + try-catch error handling

### Common Issues & Fixes (Postmortem Notes)

**Issue**: Model loading timeout on startup  
**Fix**: Implemented `@st.cache_resource` caching and HF token authentication

**Issue**: API rate limiting during high traffic  
**Fix**: Added try-catch blocks with user-friendly error messages and model fallbacks

**Issue**: MongoDB TLS certificate errors  
**Fix**: Set `tlsAllowInvalidCertificates=True` for self-signed certificates

**Issue**: Web scraping timeouts  
**Fix**: Added 5-second timeout with fallback to SerpAPI snippets

**Issue**: PyTorch `__path__._path` runtime error  
**Fix**: Added `torch.classes.__path__ = []` workaround

For comprehensive API documentation, performance analysis, and troubleshooting, see **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)**.

---

## Workflow and Functionality

### User Authentication

- **Login via Streamlit's Authentication:**  
  The application uses a login mechanism (integrated via `st.login("auth0")`) to authenticate users. When a user first accesses the app, they are presented with a custom login page complete with a logo, headings, and a styled login button.  
- **User Data Storage:**  
  Once authenticated, user details such as the name, email, and picture are saved into a MongoDB collection using the `save_user_to_mongo(user)` function for further tracking and personalization.

### Database Integration

- **MongoDB for Storing Users and Chats:**  
  Perplexa uses MongoDB as its backend database. The MongoDB connection is established using a connection string (`MONGO_URI`), which is stored in the `.env` file and loaded via Python’s `dotenv` package.  
- **Collections:**  
  Two primary collections are used:  
  - **users:** Stores user profile information.
  - **chats:** Stores chat sessions with fields for the user email, chat title, and the conversation messages.
- **Chat Operations:**  
  The code includes helper functions to save, update, fetch, and delete chat sessions. For example, `save_chat_to_mongo()` updates an existing chat or creates a new chat session.

### Embedding Model & Caching

- **Purpose of Sentence Transformer:**  
  The application uses the `SentenceTransformer` model from Hugging Face (specifically, the `all-MiniLM-L6-v2` model) to generate vector embeddings for both user queries and fetched document texts. These embeddings enable the calculation of cosine similarity to rank and select the most relevant documents.
- **Caching Strategy:**  
  The model is loaded via Streamlit’s `@st.cache_resource` decorator. This caching ensures that the heavy-weight transformer model is loaded only once, thereby reducing latency and resource consumption on subsequent queries.

### Retrieval Augmented Generation (RAG)

- **What is RAG?**  
  RAG stands for Retrieval Augmented Generation. It combines retrieval of relevant external documents with generative language models. In Perplexa, user queries are first used to retrieve context documents from the web.
- **Why Use RAG:**  
  The objective is to enrich the answer generation process by providing real-time, accurate context from external sources. The retrieved documents help the generative models produce more factually grounded and detailed responses.
- **Workflow in Code:**  
  1. The function `aggregate_documents(query)` calls `get_serpapi_results(query)` to perform a Google search.
  2. It then fetches and extracts content from each URL.
  3. The helper function `get_top_k_documents(query, documents, k=3)` computes embeddings for each document and compares them with the query embedding to select the top documents.
  4. These top documents are then combined into a context for the prompt built by `build_rag_prompt()` before being sent to a selected generative API.

### Document Retrieval & top_k_documents

- **Purpose of `top_k_documents`:**  
  This function is critical to filtering out the most relevant documents from the aggregated results. It computes cosine similarities between the query and document embeddings (generated by the SentenceTransformer) and selects the top *k* (default is 3) documents that best match the query.
- **Retrieval Process:**  
  1. Compute embeddings for all fetched documents.
  2. Compute the embedding for the user query.
  3. Calculate cosine similarities.
  4. Sort the documents based on similarity scores and concatenate the content from the top matches to form the context.

### Model Selection and Supported Models

- **User-Selectable Models:**  
  The application provides a sidebar selection for several generative models. Supported models include:
  - **Gemini:** The primary generative API for responses.
  - **Mistral:** Another option, using Mistral’s chat completions.
  - **Command R+:** Using Cohere’s API for enhanced retrieval.
  - **Deepseek R1, Phi 3, Nemotron, Meta Llama, Qwen 32B:** Options supported through the OpenRouter API.
- **Why Provide Multiple Models:**  
  Offering a range of models allows users to select based on performance, response style, cost, or specific capabilities. The app abstracts the API calls such that the correct endpoint and parameters are passed automatically based on the user’s choice.
- **API Calls:**  
  Each model has its own API call function (e.g., `call_gemini_api()`, `call_mistral_api()`, etc.) which standardizes the process of sending the final prompt with the combined context to the respective generative backend.

---

## Deployment Process

- **Webhook-Triggered Deployment:**  
  The Perplexa app is deployed on Streamlit Cloud. It is integrated with a GitHub App that uses SSL-secured webhooks to trigger deployments.  
- **SSL Security:**  
  Webhooks from GitHub are secured with SSL, ensuring that deployment triggers are transmitted securely between GitHub and Streamlit Cloud.
- **Deployment Flow:**  
  1. **Code Push:** Developers push new commits to the GitHub repository.
  2. **GitHub Webhook:** A webhook call is sent to Streamlit Cloud whenever changes are pushed.
  3. **Automatic Deployment:** Streamlit Cloud receives the webhook, pulls the latest commit, and deploys the updated app with a secure HTTPS connection.
- **Environment Configuration:**  
  Deployment also relies on environment variables (such as API keys and the MongoDB URI) being set correctly within the Streamlit Cloud environment for smooth integration with external services.

---

## Installation and Environment Variables

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sayan112207/Perplexa.git
   cd Perplexa
   ```

2. **Install Dependencies:**
   Ensure you have Python 3 installed. Then install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   Create a `.env` file in the project root with the following keys (replace placeholder values with your actual keys):
   ```env
   MONGO_URI=your_mongodb_uri_here
   SERPAPI_API_KEY=your_serpapi_key
   GEMINI_API_KEY=your_gemini_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   HF_API_KEY=your_huggingface_api_key
   COMMAND_R_PLUS=your_command_r_plus_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```
   - **MONGO_URI:** Connection string to your MongoDB database.
   - **SERPAPI_API_KEY:** API key for SerpApi to perform Google searches.
   - **GEMINI_API_KEY, MISTRAL_API_KEY, COMMAND_R_PLUS, OPENROUTER_API_KEY:** API keys for the respective generative models.
   - **HF_API_KEY:** API key for fetching Hugging Face resources like the sentence transformer.

4. **Start the Application:**
   Run the Streamlit app:
   ```bash
   streamlit run bot.py
   ```

---

## How It Works: Detailed Code Walkthrough

1. **Imports and Initialization:**
   - Various libraries are imported including Streamlit, requests, BeautifulSoup (for web scraping), and several API clients.
   - Tensor operations and caching mechanisms for the SentenceTransformer are set up, ensuring the embedding model is loaded only once.

2. **Page Configuration and Logo Rendering:**
   - The page title, icon, and layout are set using `st.set_page_config()`.
   - A helper function converts the logo image to a base64 string for efficient rendering in the UI.

3. **Database Setup:**
   - The MongoDB connection is initialized using credentials from the `.env` file.
   - Two collections, `users` and `chats`, are defined and helper functions are provided for CRUD operations.

4. **User Authentication:**
   - The login page is rendered with custom HTML and CSS styling.
   - Upon triggering the login (via query parameter), the authentication mechanism is activated.
   - Authenticated user data is stored to MongoDB, and session details are maintained.

5. **Embedding Model & Caching:**
   - The SentenceTransformer is loaded using the Hugging Face model `all-MiniLM-L6-v2` and is cached via `@st.cache_resource` to optimize performance.
   - This model converts textual documents and queries into vector embeddings.

6. **RAG and Document Processing:**
   - **Document Aggregation:**  
     The function `aggregate_documents(query)` uses SerpApi to perform a Google search and then retrieves article snippets and page contents using BeautifulSoup.
   - **Similarity Calculation:**  
     Function `get_top_k_documents(query, documents, k=3)` computes cosine similarity between query embeddings and document embeddings. It retrieves the top *k* documents to be used as context.
   - **Prompt Building:**  
     The final prompt is built by combining the user query, the top documents' context, and a reference list of URLs.
   
7. **Model Selection and API Calls:**
   - Based on the user’s sidebar selection, the final prompt is dispatched to one of several API endpoints:
     - **Gemini, Mistral, Command R+** use specific dedicated functions.
     - **Other models (Deepseek R1, Phi 3, Nemotron, Meta Llama, Qwen 32B)** are handled via the OpenRouter API.
   - The generative API returns an answer which is then appended to the session messages.

8. **Chat Interface and History Management:**
   - User messages are rendered in the chat window.
   - New chats are initiated, existing chats are loaded from MongoDB, and a chat history mechanism allows users to resume or delete previous sessions.
   - Changes to chat state automatically trigger page reruns to update the interface.

9. **People Also Ask:**
   - After an answer is generated, the app also fetches related queries (using SerpApi’s “related questions” field) to encourage further exploration.

10. **Custom Theming and Styling:**
    - The application dynamically applies dark/light themes using custom CSS, enhancing the user experience.

---

## API Documentation

For comprehensive API documentation including:
- **Detailed Architecture Diagrams**: Complete data flow from retrieval to serving
- **Performance Metrics**: Latency (P50/P95/P99), cost per request, quality metrics
- **API Endpoints**: All internal functions and external API integrations
- **Cloud Infrastructure**: Tools, services, and configuration details
- **CI/CD & MLOps**: Deployment pipeline, monitoring, and best practices
- **Postmortem Notes**: Common issues, root causes, and fixes applied
- **Testing Alternatives**: Locust, pytest, and manual API testing guides
- **Troubleshooting**: Debug mode, commands, and security considerations

Please refer to **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** for complete details.

### Quick API Reference

**Core Functions:**
- `aggregate_documents(query)` - Fetch and aggregate web search results
- `get_top_k_documents(query, documents, k=3)` - Semantic similarity-based document selection
- `build_rag_prompt(query, context, references)` - Construct LLM prompt with citations
- `call_gemini_api(prompt)` - Generate response using Google Gemini
- `call_mistral_api(prompt)` - Generate response using Mistral AI
- `call_cmd_r_plus(prompt)` - Generate response using Cohere Command R+
- `call_openrouter_api(prompt, model)` - Generate response using OpenRouter (multi-model)

**Database Functions:**
- `save_user_to_mongo(user)` - Save/update user profile
- `load_chats_from_mongo(user_email)` - Load chat history
- `save_chat_to_mongo(user_email, chat_id, title, messages)` - Save chat session
- `delete_chat_from_mongo(chat_id)` - Delete chat session

**Testing:**
```bash
# Load testing with Locust
locust -f tests/lucustfile.py --host=http://localhost:8501

# Run unit tests
pytest tests/
```

---

## Contributing and License

- **Contributing:**  
  Contributions are welcome. Developers may fork the repository, make changes, and submit pull requests. Ensure clear commit messages and follow the established code style.
- **License:**  
  The project is released under the MIT License. Please refer to the [LICENSE](https://github.com/sayan112207/Perplexa/blob/main/LICENSE) file for details.
