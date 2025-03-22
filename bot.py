import os
import json
import streamlit as st
import requests
import numpy as np
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from serpapi.google_search import GoogleSearch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cohere
from openai import OpenAI
from PIL import Image

# -----------------
# Helper function for rerunning the app (Sayan's part)
# Ye code app ko rerun karne ke liye hai.
# -----------------
def rerun():
    if hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    elif hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.stop()

# -----------------
# Page Config (Sayan's part)
# Ye code page configuration set karta hai.
# -----------------
st.set_page_config(page_title="Perplexa Chat", layout="wide")

# -----------------
# Header with Logo and Title in Columns (Patel's work)
# Ye header design karta hai aur logo aur title ko aligned rakhta hai.
# -----------------
try:
    logo = Image.open("perplexa_logo.png")
except FileNotFoundError:
    logo = None

header_col1, header_col2 = st.columns([0.1, 0.9])
with header_col1:
    if logo:
        st.image(logo, width=50)
with header_col2:
    st.markdown(
        """
        <h1 class="app-title" style="margin-top: -25px; font-family: 'Poppins', sans-serif;">
            Perplexa Chat
        </h1>
        """,
        unsafe_allow_html=True
    )

# -----------------
# Authentication (Sayan's part)
# Ye code user ko authenticate karta hai.
# -----------------
def authenticate_user():
    user = getattr(st, "experimental_user", None)
    if not user or not getattr(user, "name", None):
        st.title("Authentication")  # Ye authentication page show karta hai.
        if st.button("Login @Perplexa"):
            st.login("auth0")
        st.warning("Please log in to access the chat.")
        st.stop()
    return {
        "name": user.name,
        "email": user.email if getattr(user, "email", None) else "No Email",
        "picture": user.picture if getattr(user, "picture", None) else None
    }

user = authenticate_user()

# -----------------
# Initialize Messages (Sayan's part)
# Ye initial chat messages set karta hai.
# -----------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, I'm Perplexa. How can I help you today?"}
    ]

# -----------------
# Sidebar Layout (Sayan's part)
# Ye sidebar me welcome message, model selector, etc. show karta hai.
# -----------------
st.sidebar.markdown(
    f"<h3 style='text-align: center; font-family: Poppins, sans-serif;'>Welcome,<br><strong>{user['name']}</strong></h3>", 
    unsafe_allow_html=True
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Gemini", "Mistral", "Command R+", "Deepseek R1", "Phi 3", "Nemotron", "Meta Llama", "Qwen 32B"],
    index=0
)

# Geek Mode Button with glow animation (Patel's work)
# Ye Geek Mode button ko glow effect deta hai.
# Geek Mode Button: Toggles between normal and geeky theme, and starts a new chat session
if st.sidebar.button("🤓 Geek Mode", key="reason_btn", help="Toggle Geek Mode"):
    st.session_state.geeky_theme = not st.session_state.get("geeky_theme", False)
    
    # Store current chat history before switching modes
    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        first_question = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), "Chat Session")
        st.session_state.chat_history[user["email"]].append({"title": first_question, "messages": st.session_state.messages.copy()})
    
    # Clear current messages and start a new chat
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Geek Mode! 🧠 What geeky stuff are we discussing today?"} 
        if st.session_state.geeky_theme 
        else {"role": "assistant", "content": "Hello, I'm Perplexa. How can I help you today?"}
    ]
    
    rerun()


# -----------------
# Chat History (Sayan's part)
# Ye chat history ko sidebar me show karta hai.
# -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
user_email = user["email"]
if user_email not in st.session_state.chat_history:
    st.session_state.chat_history[user_email] = []

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align: center; font-family: Poppins, sans-serif;'>Chat History</h4>", unsafe_allow_html=True)

if st.session_state.chat_history.get(user_email, []):
    for idx, session in enumerate(st.session_state.chat_history[user_email]):
        col1, col2 = st.sidebar.columns([4, 1])
        if col1.button(f"{idx+1}. {session['title']}", key=f"chat_{idx}", help="Load this chat"):
            st.session_state.messages = session["messages"]
            rerun()
        if col2.button("🗑️", key=f"delete_{idx}", help="Delete this chat"):
            st.session_state.chat_history[user_email].pop(idx)
            rerun()
else:
    st.sidebar.write("No chat history yet.")

# -----------------
# New Chat Button (Sayan's part)
# Ye naya chat session shuru karta hai.
# -----------------
with st.sidebar.container():
    st.markdown("<div id='new-chat-container'></div>", unsafe_allow_html=True)
    new_chat = st.button("➕", key="newchat", help="Start a new chat session")
    if new_chat:
        if "messages" in st.session_state and len(st.session_state.messages) > 1:
            first_question = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), "Chat Session")
            st.session_state.chat_history[user_email].append({"title": first_question, "messages": st.session_state.messages.copy()})
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello, I'm Perplexa. How can I help you today?"}
        ]
        rerun()

# -----------------
# My Profile (Sayan's part)
# Ye user profile details sidebar me show karta hai.
# -----------------
with st.sidebar.expander("My Profile", expanded=False):
    st.markdown(f"<p style='font-family: Poppins, sans-serif;'><strong>Name:</strong> {user['name']}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-family: Poppins, sans-serif;'><strong>Email:</strong> {user['email']}</p>", unsafe_allow_html=True)
    if user['picture']:
        st.image(user['picture'], width=120)
    if st.button("Logout", key="logout", help="Click to log out"):
        if hasattr(st, "logout"):
            st.logout()
        st.session_state.clear()
        st.stop()

# -----------------
# Custom CSS (Patel's work)
# Ye code design aur UI/UX ko improve karta hai.
# -----------------
if st.session_state.get("geeky_theme", False):
    # Dark theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    body {
        background: #010832;
        color: #eade8d;
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding: 0;
        transition: background 0.3s, color 0.3s;
    }
    .sidebar .sidebar-content {
        background-color: #010832;
    }
    .logo {
        margin-left: auto; 
        margin-right: auto;
    }
    .app-title {
        font-weight: 600;
        letter-spacing: 1px;
        font-size: 2rem;
        margin-top: 10px;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #eade8d, #010832);
        color: #010832;
        border: 2px solid #eade8d;
        padding: 0.7rem 1.4rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 14px rgba(0,0,0,0.4);
    }
    .stButton>button:active {
        transform: scale(0.98);
        box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    }
    /* Glow effect ONLY on the Geek Mode (Reason) button */
    .stButton>button#reason_btn {
        animation: glowPulse 1.5s infinite ease-in-out;
    }
    @keyframes glowPulse {
        0%   { box-shadow: 0 0 5px #eade8d; }
        50%  { box-shadow: 0 0 15px #eade8d; }
        100% { box-shadow: 0 0 5px #eade8d; }
    }
    /* Chat messages */
    .chat-message {
        background: rgba(234,222,141,0.15);
        border: 1px solid rgba(234,222,141,0.3);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        animation: fadeSlide 0.4s ease-out;
    }
    @keyframes fadeSlide {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)
else:
    # Light theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    body {
        background: #eade8d;
        color: #010832;
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding: 0;
        transition: background 0.3s, color 0.3s;
    }
    .sidebar .sidebar-content {
        background-color: #eade8d;
    }
    .logo {
        margin-left: auto; 
        margin-right: auto;
    }
    .app-title {
        font-weight: 600;
        letter-spacing: 1px;
        font-size: 2rem;
        margin-top: 10px;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #010832, #eade8d);
        color: #eade8d;
        border: 2px solid #010832;
        padding: 0.7rem 1.4rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 14px rgba(0,0,0,0.3);
    }
    .stButton>button:active {
        transform: scale(0.98);
        box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    }
    /* Glow effect ONLY on the Geek Mode (Reason) button */
    .stButton>button#reason_btn {
        animation: glowPulse 1.5s infinite ease-in-out;
    }
    @keyframes glowPulse {
        0%   { box-shadow: 0 0 5px #010832; }
        50%  { box-shadow: 0 0 15px #010832; }
        100% { box-shadow: 0 0 5px #010832; }
    }
    /* Chat messages */
    .chat-message {
        background: rgba(1,8,50,0.1);
        border: 1px solid rgba(1,8,50,0.2);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        animation: fadeSlide 0.4s ease-out;
    }
    @keyframes fadeSlide {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------
# Load Environment Variables (Sayan's part)
# Ye code environment variables load karta hai.
# -----------------
load_dotenv()
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
HF_API_KEY = os.getenv('HF_API_KEY')
COMMAND_R_PLUS_API_KEY = os.getenv('COMMAND_R_PLUS')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# -----------------
# Embedding Model Initialization (Sayan's part)
# Ye code embedding model load karta hai.
# -----------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', token=HF_API_KEY)

embedding_model = load_embedding_model()

# -----------------
# Helper Functions (Sayan's part)
# Ye code chat aur RAG ke liye helper functions define karta hai.
# -----------------
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
    except Exception:
        return fallback

def aggregate_documents(query):
    results = get_serpapi_results(query)
    documents = []
    fetched_urls = []
    for result in results:
        url = result.get("link")
        fetched_urls.append(url)
        snippet = result.get("snippet", "")
        st.write(f"Fetching content from: {url}")  # Ye URL se content fetch karta hai.
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

def call_gemini_api(prompt):
    data = {"contents": [{"parts": [{"text": prompt}]}]}
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

def call_mistral_api(prompt):
    url = "https://api.mistral.ai/v1/chat/completions"
    model = "mistral-small-latest"
    headers = {
        'Authorization': f'Bearer {MISTRAL_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        return content
    except Exception as e:
        st.error(f"Error calling Mistral API: {e}")
        return None

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

def call_openrouter_api(prompt, model_choice):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    if model_choice.lower() == 'deepseek r1':
        final_model = "deepseek/deepseek-r1-zero:free"
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
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling OpenRouter API: {e}")
        return None

def generate_people_also_ask(query):
    params = {
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "q": query,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en"
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        related = results.get("related_questions", [])
        if not related:
            return []
        questions = [item.get("question") for item in related if item.get("question")]
        return questions
    except Exception:
        return []
    
    # Function to check if the message is a greeting using the Gemini API
def is_greeting(message):
    prompt = f"Check if the following message is a greeting and always answer in True or False: '{message}'"
    response = call_gemini_api(prompt)
    return response.strip().lower() == "true"

# Function to generate a greeting response using the Gemini API
def generate_greeting_response():
    prompt = "Generate a friendly greeting message."
    return call_gemini_api(prompt)

# -----------------
# Chat Interface Logic (Sayan's part)
# Ye code user ke messages display karta hai aur input leta hai.
# -----------------
for msg in st.session_state.messages:
    st.markdown(f"<div class='chat-message'>{msg['content']}</div>", unsafe_allow_html=True)

user_input = st.chat_input("Type your message here...")  # Ye user input leta hai.
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    rerun()

if st.session_state.messages[-1]["role"] == "user":
    query = st.session_state.messages[-1]["content"]

    # Check if the message is a greeting
    if is_greeting(query):
        with st.spinner("Generating greeting..."):
            greeting_response = generate_greeting_response()
        st.session_state.messages.append({"role": "assistant", "content": greeting_response})
        st.rerun()

    with st.spinner("Fetching documents..."):  # Ye documents fetch karta hai.
        documents, fetched_urls = aggregate_documents(query)
        rag_results = get_top_k_documents(query, documents, k=3)
    if rag_results:
        context = rag_results["context"]
        final_prompt = build_rag_prompt(query, context, fetched_urls)
        with st.spinner("Generating answer..."):  # Ye answer generate karta hai.
            if model_choice.lower() == 'gemini':
                answer = call_gemini_api(final_prompt)
            elif model_choice.lower() == 'mistral':
                answer = call_mistral_api(final_prompt)
            elif model_choice.lower() == 'command r+':
                answer = call_cmd_r_plus(final_prompt)
            else:
                answer = call_openrouter_api(final_prompt, model_choice)
        if answer is None:
            answer = "Sorry, there was an error generating a response."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        rerun()
    else:
        st.error("No documents were retrieved for the query.")

if len(st.session_state.messages) >= 2 and st.session_state.messages[-1]["role"] == "assistant":
    last_user_query = st.session_state.messages[-2]["content"]
    if not is_greeting(last_user_query):
        related_questions = generate_people_also_ask(last_user_query)
        st.markdown("### People also ask")
        for q in related_questions:
            if st.button(q, key=q):
                st.session_state.messages.append({"role": "user", "content": q})
                rerun()
