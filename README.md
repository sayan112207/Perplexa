# Perplexa
A smart web-searching AI that transforms your questions into accurate, reliable answers backed by real-time information.

1. Model: qwen2:7b-instruct-q6_K
2. Web-Search Engine: [Duck Duck Go](https://duckduckgo.com/)

## Process Overview
<img src="https://github.com/sayan112207/Perplexa/blob/main/perplexa-process.jpg?raw=true"/>

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
