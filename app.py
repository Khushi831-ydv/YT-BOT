import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from datetime import datetime

def get_video_id(url):
    """Extracts video ID from YouTube URL."""
    video_id = None
    if "v=" in url:
        video_id = url.split("v=")[1][:11]
    elif "youtu.be" in url:
        video_id = url.split("/")[-1].split("?")[0]
    return video_id if video_id and len(video_id) == 11 else None

def get_transcript(video_id):
    """Fetches transcript for a given video ID."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([d['text'] for d in transcript_list])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

def get_text_chunks(text):
    """Splits text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key, model_name):
    """Creates a vector store from text chunks."""
    try:
        if model_name == "OpenAI":
            os.environ["OPENAI_API_KEY"] = api_key
            embeddings = OpenAIEmbeddings()
        elif model_name == "Gemini":
            os.environ["GOOGLE_API_KEY"] = api_key
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        else:
            st.error("Invalid model selected.")
            return None
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_summary(text, api_key, model_name):
    """Generates a summary of the given text."""
    try:
        if model_name == "OpenAI":
            llm = ChatOpenAI(api_key=api_key, temperature=0, model_name="gpt-3.5-turbo")
        elif model_name == "Gemini":
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, convert_system_message_to_human=True, temperature=0)
        else:
            return "Invalid model selected for summarization."

        docs = [Document(page_content=text)]
        chain = load_summarize_chain(llm, chain_type="stuff")
        summary = chain.run(docs)
        return summary
    except Exception as e:
        st.error(f"Error creating summary: {e}")
        return None

def get_conversation_chain(vector_store, api_key, model_name):
    try:
        if model_name == "OpenAI":
            llm = ChatOpenAI(api_key=api_key)
        elif model_name == "Gemini":
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, convert_system_message_to_human=True)
        else:
            return None
        
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

st.set_page_config(page_title="Chat with YouTube Videos", page_icon="🐬", layout="wide")
st.title("Chat with YouTube Videos 🐬")
st.info("Make sure the video has English subtitles and is a valid Youtube URL.")

# Initialize session state
if "api_key_loaded" not in st.session_state:
    st.session_state.api_key_loaded = False
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "processed" not in st.session_state:
    st.session_state.processed = False
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

with st.sidebar:
    st.markdown(
        """
        <a href="https://www.linkedin.com/in/khushi-yadav-937501291/" target="_blank" style="text-decoration: none;">
            <div style="
                padding: 10px 12px;
                background-color: #0077B5;
                color: white;
                text-align: center;
                border-radius: 5px;
                font-weight: bold;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            ">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-linkedin"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>
                <span>Connect on LinkedIn</span>
            </div>
        </a>
        <br>
        """,
        unsafe_allow_html=True
    )
    st.header("Configuration")
    model_choice = st.radio("Choose your model", ("OpenAI", "Gemini"), key="model_choice")
    api_key_input = st.text_input("Enter your API Key", type="password", key="api_key_input")
    if st.button("Load API Key"):
        if st.session_state.api_key_input:
            st.session_state.api_key = st.session_state.api_key_input
            st.session_state.api_key_loaded = True
            st.success("API Key loaded successfully!")
        else:
            st.error("Please enter an API Key.")

    if st.session_state.api_key_loaded:
        st.header("YouTube Video")
        youtube_url_input = st.text_input("Enter YouTube URL", key="youtube_url_input")
        if st.button("Load Video"):
            if st.session_state.youtube_url_input:
                video_id = get_video_id(st.session_state.youtube_url_input)
                if video_id:
                    st.session_state.video_id = video_id
                    st.session_state.youtube_url = st.session_state.youtube_url_input
                    st.session_state.video_loaded = True
                    st.session_state.processed = False # Reset processing state
                    st.session_state.chat_history = None
                    st.session_state.conversation = None
                    st.success("Video loaded successfully!")
                else:
                    st.error("Invalid YouTube URL.")
            else:
                st.error("Please enter a YouTube URL.")

    st.header("Feedback")
    feedback_name = st.text_input("Your Name", key="feedback_name")
    feedback_rating = st.slider("Rate your experience (1-5 Stars)", 1, 5, 3, key="feedback_rating")
    feedback_text = st.text_area("Your Feedback", key="feedback_text")
    if st.button("Submit Feedback"):
        if feedback_name and feedback_text:
            with open("feedback.log", "a") as f:
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Name: {feedback_name}\n")
                f.write(f"Rating: {feedback_rating} Stars\n")
                f.write(f"Feedback: {feedback_text}\n")
                f.write("-" * 20 + "\n")
            st.success("Feedback sent!")
        else:
            st.warning("Please provide your name and feedback.")

if st.session_state.video_loaded and not st.session_state.processed:
    st.video(st.session_state.youtube_url)
    with st.spinner("Processing video..."):
        transcript = get_transcript(st.session_state.video_id)
        if transcript:
            summary = get_summary(transcript, st.session_state.api_key, st.session_state.model_choice)
            st.session_state.summary = summary

            text_chunks = get_text_chunks(transcript)
            vector_store = get_vector_store(text_chunks, st.session_state.api_key, st.session_state.model_choice)
            if vector_store:
                st.session_state.conversation = get_conversation_chain(vector_store, st.session_state.api_key, st.session_state.model_choice)
                st.session_state.processed = True
            else:
                st.session_state.video_loaded = False
        else:
            st.session_state.video_loaded = False

if st.session_state.processed:
    st.video(st.session_state.youtube_url)
    st.subheader("Summary")
    st.write(st.session_state.summary)

    st.subheader("Chat with the Video")

    # Handle user input before displaying chat history
    user_question = st.chat_input("Ask a question about the video:")
    if user_question:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

    # Display chat history
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                with st.chat_message("assistant"):
                    st.write(message.content)
