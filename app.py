import os
import json
import streamlit as st
import requests
import numpy as np
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cohere
from openai import OpenAI


# -----------------
# Streamlit Layout
# -----------------
st.set_page_config(
        page_title="Perplexa"
    )
st.title("Perplexa")

# Load environment variables
load_dotenv()
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
HF_API_KEY = os.getenv('HF_API_KEY')
COMMAND_R_PLUS_API_KEY = os.getenv('COMMAND_R_PLUS')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Load the SentenceTransformer model (ensure correct model identifier)
@st.cache_resource
def load_embedding_model():
    # Use the public identifier "all-MiniLM-L6-v2"
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', token=HF_API_KEY)

embedding_model = load_embedding_model()

# SERPAPI query function
def get_serpapi_results(query, num_results=5):
    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "q": query,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "num": str(num_results)
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get("organic_results", [])

def extract_content(url, fallback=""):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join(p.get_text() for p in paragraphs).strip()
        return text if text else fallback
    except Exception as e:
        return fallback

def aggregate_documents(query):
    """
    Fetch and aggregate documents from SERPAPI results.
    Returns a tuple of:
      - documents: list of dicts with "url" and "content"
      - fetched_urls: list of fetched URLs (for display and references)
    """
    results = get_serpapi_results(query)
    documents = []
    fetched_urls = []
    for result in results:
        url = result.get("link")
        fetched_urls.append(url)
        snippet = result.get("snippet", "")
        st.write(f"Fetching content from: {url}")
        content = extract_content(url, fallback=snippet)
        if content:
            documents.append({"url": url, "content": content})
    return documents, fetched_urls

def compute_embeddings(texts):
    return embedding_model.encode(texts)

def get_top_k_documents(query, documents, k=3):
    if not documents:
        return None
    texts = [doc["content"] for doc in documents]
    doc_embeddings = compute_embeddings(texts)
    query_embedding = embedding_model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    top_docs = [documents[i]["content"] for i in top_indices]
    return {"context": "\n\n".join(top_docs)}

def build_rag_prompt(query, context, references):
    # Use the full list of fetched URLs as the references.
    refs_formatted = "\n".join(f"- {ref}" for ref in references)
    prompt = (
        f"You are an expert assistant. Using the following retrieved context, provide a detailed explanation for the query: '{query}'.\n\n"
        f"Context:\n{context}\n\n"
        "Your response should include:\n"
        "1. An introduction defining the topic.\n"
        "2. Key points in bullet format.\n"
        "3. A references section listing the source URLs.\n\n"
        f"References:\n{refs_formatted}\n\n"
        "Make sure your answer is clear, concise, and accessible to beginners."
    )
    return prompt

#Gemini
def call_gemini_api(prompt):
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))
        response.raise_for_status()
        answer = (response.json().get("candidates", [{}])[0]
                  .get("content", {})
                  .get("parts", [{}])[0]
                  .get("text"))
        return answer if answer else "No answer received from Gemini API."
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

#Mistral
def call_mistral_api(prompt):
    url = "https://api.mistral.ai/v1/chat/completions"
    model = "mistral-small-latest"
    headers = {
        'Authorization': f'Bearer {MISTRAL_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        return content
    except Exception as e:
        st.error(f"Error calling Mistral API: {e}")
        return None

#Command R+
def call_cmd_r_plus(prompt):
    co = cohere.ClientV2(COMMAND_R_PLUS_API_KEY)
    
    try:
        response = co.chat(
        model="command-r-plus", 
        messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content[0].text
    except Exception as e:
        st.error(f"Error calling Command-R-Plus API: {e}")
        return None

#DeepSeek R1, Phi 3, Nemotron, Meta Llama, Qwen 32B
def call_openrouter_api(prompt, model_choice):
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    )

    if model_choice.lower() == 'deepseek r1':
        final_model="deepseek/deepseek-r1-zero:free"
    elif model_choice.lower() == 'phi 3':
        final_model = "microsoft/phi-3-medium-128k-instruct:free"
    elif model_choice.lower() == 'nemotron':
        final_model = "nvidia/llama-3.1-nemotron-70b-instruct:free"
    elif model_choice.lower() == 'meta llama':
        final_model = "meta-llama/llama-3.3-70b-instruct:free"
    elif model_choice.lower() == 'qwen 32b':
        final_model = "qwen/qwq-32b:free"

    try:
        completion = client.chat.completions.create(
        extra_body={},
        
        model=final_model,

        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ]
        )
        return(completion.choices[0].message.content)
    except Exception as e:
        st.error(f"Error calling OpenRouter API: {e}")
        return None

# Streamlit UI

query = st.text_input("Enter your Query", "")
model_choice = st.selectbox("Select Model", ["Gemini", "Mistral", "Command R+", "Deepseek R1", "Phi 3", "Nemotron", "Meta Llama", "Qwen 32B"])

if st.button("Submit") and query:
    with st.spinner("Fetching documents..."):
        documents, fetched_urls = aggregate_documents(query)
        rag_results = get_top_k_documents(query, documents, k=3)
    
    if rag_results:
        context = rag_results["context"]
        # Use fetched_urls as references to ensure they are the same.
        final_prompt = build_rag_prompt(query, context, fetched_urls)
        
        with st.spinner("Generating answer..."):
            if model_choice.lower() == 'gemini':
                answer = call_gemini_api(final_prompt)
            elif model_choice.lower() == 'mistral':
                answer = call_mistral_api(final_prompt)
            elif model_choice.lower() == 'command r+':
                answer = call_cmd_r_plus(final_prompt)
            else:
                answer = call_openrouter_api(final_prompt, model_choice)
        
        st.subheader("Generated Answer")
        st.write(answer)
    else:
        st.error("No documents were retrieved for the query.")