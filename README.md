# Perplexa

A smart web-searching AI with Retrieval Augmented Generation (RAG) system that provides AI-powered search capabilities. This project combines web scraping, semantic search, and language model generation to deliver comprehensive answers to user queries.

## Features

- **Web Search Integration**: Scrapes search results from DuckDuckGo to gather relevant information
- **Intelligent Query Processing**: Extracts optimal search keywords from user questions
- **Content Validation**: Evaluates and filters search results for relevance
- **Context-Aware Summarization**: Condenses web content into concise, useful information
- **AI-Powered Responses**: Generates comprehensive, referenced answers using the Qwen2 language model
- **Thai Language Support**: Processes and generates responses in Thai (configurable for other languages)

## Technology Stack

- **Ollama**: Local AI model hosting and inference
- **Qwen2 7B Instruct**: Compact yet powerful language model for generation tasks
- **BeautifulSoup**: HTML parsing for web scraping
- **Requests**: HTTP requests for data retrieval
- **JSON**: Data format for structured information exchange
- **Threading**: Concurrent operations for improved performance

## DataBase Used
- Mongo DB
  
## How It Works

1. **Query Processing**: Analyzes the user question to extract optimal search keywords
2. **Web Scraping**: Fetches relevant search results from DuckDuckGo
3. **Content Validation**: Evaluates the usefulness of each search result
4. **Summarization**: Extracts and condenses relevant information
5. **Response Generation**: Creates a comprehensive, well-structured answer with citations

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install requests bs4 ollama retry
   ```

2. Install Ollama:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

3. Pull the Qwen2 model:
   ```bash
   ollama pull qwen2:7b-instruct-q6_K
   ```

4. Set environment variables:
   ```bash
   export OLLAMA_HOST=0.0.0.0:11434
   export OLLAMA_ORIGINS=*
   ```

5. Start the Ollama server:
   ```bash
   ollama serve
   ```

## Usage

```python
# Import the necessary components
from perplexa import suplexity, display_answer

# Ask a question
question = "What is JavaScript"

# Get a comprehensive answer
response = suplexity(question)
answer = display_answer(response)
print(answer)
```
## Process Overview
<img src="https://github.com/sayan112207/Perplexa/blob/main/perplexa-process.jpg?raw=true"/>

## Architecture

This project implements a RAG (Retrieval Augmented Generation) architecture:

1. **Retrieval**: Searches the web for relevant information
2. **Augmentation**: Processes, filters, and contextualizes the retrieved information
3. **Generation**: Uses a language model to create a comprehensive response

## Workflow
```mermaid
graph TD;
    User-enters-query-->Scraped-Web-Results;
    Scraped-Web-Results-->Summarized-Results;
    Summarized-Results-->Converted-into-vector-embeddings;
    Converted-into-vector-embeddings-->Provide-Ollama-with-the-Vector-Document;
    Provide-Ollama-with-the-Vector-Document-->Generated-answer-with-web-results-in-context;
    Generated-answer-with-web-results-in-context-->Generated-Response-converted-into-standard-document;
    Generated-Response-converted-into-standard-document-->Results
```

# Perplexa Research

Perplexa Research Mode is an interactive research environment designed to streamline the process of fetching, processing, and querying academic content via [arxiv.org](http://arxiv.org/), it offers a modular, interactive, and highly extensible pipeline that leverages the latest advances in AI and Large Language Models (LLMs) to bring academic content to life.

## Perplexa Research Architectural Overview
1. **Data Ingestion via arXiv**
- **Purpose:** Retrieve academic papers and research content from the arXiv repository.
- **How It Works:**
  - The notebook leverages the [arxiv](https://pypi.org/project/arxiv/) package to search and download metadata and content of research papers.
  - This component acts as the data source for the entire pipeline, feeding raw text data into subsequent modules.

2. **Indexing and Embedding with Llama Index Framework**
- **Purpose:** Convert raw research documents into queryable data structures.
- **How It Works:**
  - Index Creation:
    - Uses the Llama Index framework to build indices from ingested research content.
    - These indices facilitate fast retrieval of relevant information when responding to queries.
  - Embedding Generation:
    - Extensions like llama-index-llms-mistralai and llama-index-embeddings-mistralai are employed to generate vector embeddings for document sections.
    - This enables the system to perform semantic similarity searches and contextual matching based on natural language queries.

3. **Query Processing and LLM Integration using [Mistral API](https://docs.mistral.ai/api/)**
- **Purpose:** Allow users to query the indexed research data in natural language.
- **How It Works:**
  - Agent-Based Querying:
    - The notebook sets up an agent layer that leverages LLMs to interpret and respond to user queries.
    - The integration with various LLM backends (*Mistralai* via dedicated modules) ensures that responses are generated with up-to-date language understanding.
  - Data Fusion:
    - By combining indexed data with live LLM processing, the system produces answers that are both contextually informed and semantically rich.

4. **Interactive UI with Gradio**
- **Purpose:** Provide an accessible, web-based interface for exploring research queries.
- **How It Works:**
  - Interface Building:
    - Using Gradio (version 3.39.0), the notebook launches a simple web interface.
    - Users can type in queries and view results in real time.
  - User Experience:
    - The UI is designed to be intuitive, lowering the barrier for end users to interact with the model.


## Future Improvements

- Add support for more search engines
- Add caching for frequently asked questions
- Support for multiple languages
- Add a web interface for easier access

## License

[MIT License](LICENSE)
