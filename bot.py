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
import base64
from openai import OpenAI
from PIL import Image
from pymongo import MongoClient
from bson.objectid import ObjectId
#Fixing: RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
import torch
torch.classes.__path__ = []


# -----------------
# Helper Functions (Sayan)
# -----------------

def rerun():
    if hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()
    elif hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.stop()

def get_image_as_base64(image_path):     #for clearer logo image and faster render (Patel)
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert your logo to base64
logo_base64 = get_image_as_base64("perplexa_logo.png")
# -----------------
# Page Config
# -----------------
st.set_page_config(page_title="Perplexa Chat", 
                   layout="wide", 
                   page_icon="https://github.com/sayan112207/Perplexa/blob/main/perplexa_logo.png?raw=true")


# -----------------
# DataBase Setup and Functions Part (Satya's part)
# -----------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")  # Store your MongoDB URI in .env
try:
    client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
    # db = client.test  # Test the connection
    db = client.get_database("perplexa_chat")  # Specify your database name
    print("Connected Successfully!")
except Exception as e:
    print("Connection failed:", e)

if db is not None:
    users_collection = db["users"]
    chats_collection = db["chats"]
else:
    print("Database connection failed. Cannot proceed.")

# Function to save a new user
def save_user_to_mongo(user):
    users_collection.update_one({"email": user["email"]}, {"$set": user}, upsert=True)
    
# Function to fetch all user chats
def load_chats_from_mongo(user_email):
    return list(chats_collection.find({"email": user_email}))

# Function to save a new chat
def save_chat_to_mongo(user_email, chat_id,chat_title, messages):
    # chat_data = {"email": user_email, "title": chat_title, "messages": messages}
    # chats_collection.insert_one(chat_data)
    if chat_id:
        # If chat already exists update messages
        chats_collection.update_one(
            {"_id": ObjectId(chat_id)},
            {"$set": {"messages": messages}},
        )
    else:
        # If chat does not exist, create a new one
        chat_data = {"email": user_email, "title": chat_title, "messages": messages}
        chat_id = chats_collection.insert_one(chat_data).inserted_id
        return str(chat_id)  # Return the new chat ID

# Function to delete a specific chat
def delete_chat_from_mongo(chat_id):
    chats_collection.delete_one({"_id": ObjectId(chat_id)})


# -----------------
# User Authentication (Sayan & Patel)
# -----------------

def authenticate_user():
    user = getattr(st, "experimental_user", None)
    
    if not user or not getattr(user, "name", None):
        st.markdown(
            f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

            :root {{
                --primary-color: #1E1E2F;
                --secondary-color: #EAD78B;
                --button-color: linear-gradient(135deg, #EAD78B, #C9B368);
            }}

            .login-container {{
                text-align: center;
                color: var(--secondary-color);
                font-family: 'Poppins', sans-serif;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                width: 100vw;
                background: linear-gradient(to bottom, #0e1117, var(--primary-color));
                position: fixed;
                top: 0;
                left: 0;
            }}

            .login-logo {{
                width: 250px;
                margin-bottom: 20px;
                margin-right: 27px;
            }}

            .login-heading {{
                font-size: 2.5rem;
                font-weight: 600;
                margin-bottom: 10px;
            }}
            .login-tagline {{
                font-size: 1.5rem;
                letter-spacing: 5.5px;
            }}

            .login-btn {{
                background: var(--button-color);
                color: #1E1E2F !important;
                font-size: 1.2rem;
                font-weight: 600;
                padding: 14px 36px;
                margin-top: 20px;
                margin-right: 10px;
                border: none;
                border-radius: 30px;
                cursor: pointer;
                transition: 0.3s ease-in-out;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
                text-decoration: none !important;
            }}

            .login-btn:hover {{
                transform: scale(1.05);
                box-shadow: 0px 6px 14px rgba(0,0,0,0.4);
            }}
            </style>

            <div class="login-container">
                <img src="data:image/png;base64,{logo_base64}" class="login-logo"/>
                <h1 class="login-heading">Perplexa Chat</h1>
                <p class="login-tagline">A NEW WAY OF SEARCHING</p>
                <a href="?login=true" class="login-btn" target="_self">Login@Perplexa</a>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.query_params.get("login") == "true":
            st.query_params.clear()
            st.login("auth0")
        st.stop()
    
    user_data = {
        "name": user.name,
        "email": user.email if getattr(user, "email", None) else "No Email",
        "picture": user.picture if getattr(user, "picture", None) else None
    }
    save_user_to_mongo(user)
    return user_data

# Get user data after authentication
user = authenticate_user()
user_email = user["email"]

# -----------------
# Header with Logo and Title in Columns (Patel)
# (Only shown after authentication)
# -----------------

try:
    with open("perplexa_logo.png", "rb") as img_file:
        logo_base64 = base64.b64encode(img_file.read()).decode()
except FileNotFoundError:
    logo_base64 = None

# --- HEADER SECTION ---
st.markdown(
    """
    <style>
        .header-container {
            display: flex;
            align-items: center;  
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        .header-logo img {
            width: 55px;
            height: 55px; 
            display: block;
        }
        .header-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2rem;
            font-weight: 600;
            color: var(--secondary-color);
            line-height: 55px; /* Ensures text aligns with the image */
            margin: 0;
            display: flex;
            align-items: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Render Header with Logo ---

if logo_base64:
    st.markdown(
        f"""
        <div class="header-container">
            <div class="header-logo">
                <img src="data:image/png;base64,{logo_base64}" alt="Perplexa Logo">
            </div>
            <div class="header-title">Perplexa Chat</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="header-container">
            <div class="header-title">Perplexa Chat</div>
        </div>
        """,
        unsafe_allow_html=True
    )  

# --- INITIALIZE CHAT MESSAGES ---

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, I'm Perplexa. How can I help you today?"}
    ]

# --- SIDEBAR DESIGN ---

st.sidebar.markdown(
    f"""
    <div style="
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--secondary-color);
        padding: 15px 10px;
        margin-bottom: 10px;
        ">
        Welcome,<br><span style="font-weight: 500;">{user['name']}</span>
    </div>
    <hr style="border: none; height: 1px; background: rgba(0, 0, 0, 0.1); width: 80%; margin: auto;">
    """,
    unsafe_allow_html=True
)

# --- MODEL SELECTION ---

st.sidebar.markdown(
    "<p style='font-family: Poppins, sans-serif; font-size: 1.1rem; font-weight: 600; margin-top: 2px; margin-bottom: 2px;'>Select Model</p>",
    unsafe_allow_html=True
)

model_choice = st.sidebar.selectbox(
    "Select a model",
    ["Gemini", "Mistral", "Command R+", "Deepseek R1", "Phi 3", "Nemotron", "Meta Llama", "Qwen 32B"],
    index=0,
    label_visibility="collapsed"  # Hides the label visually but keeps it accessible
)



# -----------------
# Theme Mode Button with animation (Patel)
# -----------------

if "theme" not in st.session_state:
    st.session_state.theme = False

theme_mode_on = st.sidebar.toggle("Theme Mode", value=st.session_state.theme, help="Toggle Theme Mode") #toggle 

# Update session state based on the toggle state
if theme_mode_on != st.session_state.theme:
    st.session_state.theme = theme_mode_on

    # Store chat history if geek mode is toggled and messages exist
    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        first_question = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), "Chat Session")
        st.session_state.chat_history[user["email"]].append({"title": first_question, "messages": st.session_state.messages.copy()})

    # Set the assistant's initial response based on the toggle
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Light Mode! 😄 What stuff are we discussing today?"}
        if st.session_state.theme
        else {"role": "assistant", "content": "Hello, I'm Perplexa. How can I help you today?"}
    ]
    
    # Trigger page rerun after state change
    st.rerun() 

# Display the status of Theme Mode
if st.session_state.theme:
    st.sidebar.write("🌞 Light Mode is ON!")
else:
    st.sidebar.write("🌚 Dark Mode is ON!")

# -----------------
# Chat History (Sayan's part)
# Chat Database added (Satya's part)
# Ye chat history ko sidebar me show karta hai.
# -----------------

# Load user chats
user_chats = load_chats_from_mongo(user_email)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
# user_email = user["email"]

#.......(Satya's part) To store current chat id to update frequently
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if user_email not in st.session_state.chat_history:
    st.session_state.chat_history[user_email] = [{"_id": str(chat["_id"]),"title": chat["title"], "messages": chat["messages"]} for chat in user_chats]

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align: center; font-family: Poppins, sans-serif;'>Chat History</h4>", unsafe_allow_html=True)

if st.session_state.chat_history.get(user_email, []):
    for idx, session in enumerate(st.session_state.chat_history[user_email]):
        col1, col2 = st.sidebar.columns([4, 1])
        if col1.button(f"{idx+1}. {session['title']}", key=f"chat_{idx}", help="Load this chat"):
            st.session_state.messages = session["messages"]
            rerun()
        if col2.button("🗑️", key=f"delete_{idx}", help="Delete this chat"):
            if "_id" in session:
                delete_chat_from_mongo(session["_id"])  # Remove from MongoDB
            else:
                print("Error: '_id' not found in session")
            st.session_state.chat_history[user_email].pop(idx)
            rerun()
else:
    st.sidebar.write("No chat history yet.")

# -----------------
# New Chat Button (Sayan's part)
# Chat stored in Database (Satya's part)
# Ye naya chat session shuru karta hai.
# -----------------
with st.sidebar.container():
    st.markdown("<div id='new-chat-container'></div>", unsafe_allow_html=True)
    new_chat = st.button("➕", key="newchat", help="Start a new chat session")
    if new_chat:
        if "messages" in st.session_state and len(st.session_state.messages) > 1:
            first_question = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), "Chat Session")
            save_chat_to_mongo(user_email,None , first_question, st.session_state.messages.copy())
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
# Custom CSS (Patel)
# -----------------

if st.session_state.get("theme", False):
    # Dark Theme Styles
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    :root {
        --bg-color: #010832;
        --text-color: #eade8d;
        --accent-color: #eade8d;
        --button-bg: linear-gradient(135deg, #eade8d, #010832);
        --button-text: #010832;
        --button-border: #eade8d;
        --message-bg: rgba(234,222,141,0.15);
        --message-border: rgba(234,222,141,0.3);
        --shadow-color: rgba(0,0,0,0.3);
    }
    """, unsafe_allow_html=True)
else:
    # Light Theme Styles
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root {
        --bg-color: #f4f4f9;
        --text-color: #1b1b2f;
        --accent-color: #1b1b2f;
        --button-bg: linear-gradient(135deg, #1b1b2f, #eade8d);
        --button-text: #fff;
        --button-border: #1b1b2f;
        --message-bg: rgba(1,8,50,0.1);
        --message-border: rgba(1,8,50,0.2);
        --shadow-color: rgba(0,0,0,0.2);
    }
    """, unsafe_allow_html=True)

# Common Styles
st.markdown("""
<style>
/* Global Styles */
body {
    background: var(--bg-color);
    color: var(--text-color);
    font-family: 'Inter', sans-serif;
    margin: 0;
    padding: 0;
    transition: background 0.3s ease, color 0.3s ease;
}

/* Sidebar */
.sidebar .sidebar-content {
    background-color: var(--bg-color);
    padding: 10px;
}

/* App Title */
.app-title {
    font-weight: 700;
    letter-spacing: 1px;
    font-size: 2.5rem;
    margin-top: 10px;
    text-align: center;
    animation: fadeIn 1s ease-in-out;
}

/* Buttons */
.stButton > button {
    border: 2px solid var(--button-border);
    font-size: 16px;
    font-weight: 600;
    padding: 10px 20px;
    border-radius: 8px;
    transition: all 0.3s ease-in-out;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px var(--shadow-color);
}

.stButton > button:active {
    transform: translateY(1px);
    box-shadow: 0 3px 6px var(--shadow-color);
}

/* Special Animated Button */
.stButton > button#reason_btn {
    animation: glowPulse 1.5s infinite ease-in-out;
}

/* Chat Messages */
.chat-message {
    background: var(--message-bg);
    border: 1px solid var(--message-border);
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    font-size: 15px;
    line-height: 1.5;
    box-shadow: 2px 2px 10px var(--shadow-color);
    animation: fadeSlide 0.5s ease-out;
}

/* Hover effect on chat messages */
.chat-message:hover {
    transform: scale(1.02);
    transition: transform 0.3s ease-in-out;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes glowPulse {
    0%   { box-shadow: 0 0 5px var(--accent-color); }
    50%  { box-shadow: 0 0 15px var(--accent-color); }
    100% { box-shadow: 0 0 5px var(--accent-color); }
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
#@st.cache_resource
#def load_embedding_model():
    #return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', token=HF_API_KEY)
    #return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', token=HF_API_KEY)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None
    

embedding_model = load_embedding_model()

#Debugging Step
if embedding_model is None:
    st.error("Failed to load the embedding model. Please check your HF API key or model name.")
    st.stop()

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
        "A references section listing the source URLs.\n\n"
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

# .... Immediately Update Chat History (Satya's Part)
user_input = st.chat_input("Type your message here...")  # Ye user input leta hai.
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Save or update chat in MongoDB
    if not st.session_state.current_chat_id:
        chat_title = user_input[:50]  # Use first 50 characters as title
        chat_id = save_chat_to_mongo(user_email, None, chat_title, st.session_state.messages)

        if chat_id:
            st.session_state.current_chat_id = chat_id
            # Immediately update chat history
            if user_email in st.session_state.chat_history:
                st.session_state.chat_history[user_email].append({"_id": chat_id, "title": chat_title, "messages": st.session_state.messages.copy()})
            else:
                st.session_state.chat_history[user_email] = [{"_id": chat_id, "title": chat_title, "messages": st.session_state.messages.copy()}]
    else:
        save_chat_to_mongo(user_email, st.session_state.current_chat_id, None, st.session_state.messages)
    
    #Reload Chat History
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
