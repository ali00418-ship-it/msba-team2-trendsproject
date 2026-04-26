# app.py
import streamlit as st
import duckdb
import os
import io
import tempfile
import plotly.io as pio
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from faster_whisper import WhisperModel
from openai import OpenAI
import file_manipulation as fm

from dotenv import load_dotenv
load_dotenv()

DATA_GLOB = "data/labeled/*.parquet"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@st.cache_resource
def load_whisper():
    """Load Whisper once and reuse across reruns."""
    return WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribe raw audio bytes from st.audio_input using Whisper."""
    model = load_whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        segments, _ = model.transcribe(tmp_path, beam_size=5)
        return " ".join([s.text for s in segments]).strip()
    finally:
        os.remove(tmp_path)


def text_to_speech(text: str) -> bytes:
    """Convert text to speech using OpenAI TTS. Returns MP3 bytes."""
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )
    return response.content


def run_agent(agent_executor, user_input: str):
    """Run the agent and return (output_text, figure_or_None)."""
    response = agent_executor.invoke({"input": user_input})
    output_text = response["output"]

    generated_fig = None
    if os.path.exists("plot.json"):
        with open("plot.json", "r") as f:
            generated_fig = pio.from_json(f.read())
        os.remove("plot.json")

    return output_text, generated_fig


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #

def configure_chatbot():

    if "app_initialized" not in st.session_state:
        if os.path.exists("plot.json"):
            os.remove("plot.json")
        st.session_state.app_initialized = True

    st.set_page_config(page_title="Data Chatbot", layout="wide")

    st.markdown("""
        <style>
        [data-testid="stHorizontalBlock"] {
            align-items: center;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("karen.png", width=700)
    with col2:
        st.title("Hi, I am Karen, your complaints data analyzer. How can I help? 📊")

    # --- Sidebar: voice toggle ---
    with st.sidebar:
        st.header("⚙️ Settings")
        voice_input_on  = st.toggle("🎙️ Voice input",  value=False)
        voice_output_on = st.toggle("🔊 Voice responses", value=False)

    # --- LLM + Agent ---
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    tools = [PythonREPLTool()]

    instructions = f"""
        You are an expert data analyst named Karen. You have access to a folder of Parquet files
        matched by the glob pattern '{DATA_GLOB}' containing consumer financial complaints.
        The data includes the original labeled dataset as well as newly transcribed and classified
        audio recordings that have been automatically assigned an Issue and priority_tier.

        Here is the schema of the dataset:
        - Date received (YYYY-MM-DD or TIMESTAMP — cast to TIMESTAMP when filtering)
        - Product
        - Sub-product
        - Issue (one of ~62 complaint categories, e.g. "Fraud or scam", "Managing an account")
        - Sub-issue
        - Consumer complaint narrative (the raw complaint text)
        - Company
        - State
        - Tags
        - Company response to consumer
        - Consumer disputed?
        - clean_narrative (cleaned version of the complaint text)
        - word_count (integer word count of the narrative)
        - lda_topic (LDA topic number)
        - lda_topic_label (human-readable LDA topic label)
        - bert_topic (BERT topic number)
        - bert_topic_label (human-readable BERT topic label)
        - category (broader category grouping)
        - year_month (e.g. "2024-01")
        - year (integer)
        - month (integer)
        - date_dt (DATE type)
        - volume_score (DOUBLE)
        - growth_score (DOUBLE)
        - is_unresolved (INTEGER, 0 or 1)
        - severity_score (DOUBLE)
        - recency_score (DOUBLE)
        - length_score (DOUBLE)
        - danger_boost (DOUBLE)
        - priority_score (DOUBLE, composite score)
        - priority_tier (one of: Critical, High, Medium, Low)

        IMPORTANT: Newly transcribed rows will have NULL for many columns (Product, Sub-product,
        topic labels, scores, etc.) but will always have: Date received, Consumer complaint narrative,
        clean_narrative, Issue, priority_tier, Company, State, and word_count.

        CRITICAL DUCKDB INSTRUCTIONS:
        1. NEVER use pandas to read the parquet files.
        2. ALWAYS create a fresh DuckDB connection for each query to ensure newly added files
           are visible. Use this pattern:
           ```
           import duckdb
           con = duckdb.connect()
           result_df = con.sql('SELECT "Company" FROM read_parquet("{DATA_GLOB}", filename=true) LIMIT 5').df()
           con.close()
           ```
        3. Because column names contain spaces, you MUST wrap them in double quotes in your SQL.
        4. Columns not present in transcribed rows will be NULL — handle them gracefully
           (use COALESCE or IS NOT NULL filters when needed).
        5. When filtering or sorting by "Date received", always cast it to TIMESTAMP first:
        ORDER BY CAST("Date received" AS TIMESTAMP) DESC

        CRITICAL OUTPUT INSTRUCTIONS:
        Before running ANY query, you MUST first run this setup code to prevent truncation:
        ```
        import pandas as pd
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 50)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 200)
        ```
        For large result sets, use `.to_string()` instead of just printing the dataframe:
        `print(result_df.to_string())`

        CRITICAL FORMATTING INSTRUCTIONS:
        You have access to the following tools:
        {{tools}}
        Available tool names: [{{tool_names}}]

        You must use the exact format below. Do not add extra words to the Action line.

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: Python_REPL
        Action Input: the exact python code to execute
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Instructions for Visualization:
        If the user asks for a chart, graph, or plot:
        1. Query the data using duckdb with the glob pattern.
        2. Create an interactive plot using `plotly.express`.
        3. You MUST save the figure as a JSON file named 'plot.json' using `fig.write_json('plot.json')`.
        4. You MUST include this exact print statement at the end of your python code: `print("CHART_SUCCESS")`
        5. CRITICAL: The moment you see "CHART_SUCCESS" in your Observation, you must IMMEDIATELY stop and output exactly this:

        Thought: I saw CHART_SUCCESS, the chart is ready.
        Final Answer: The chart has been generated successfully.

        Question: {{input}}
        {{agent_scratchpad}}
        """

    prompt = PromptTemplate.from_template(instructions)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,
        max_execution_time=120,
    )

    # --- Chat history ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_voice_input" not in st.session_state:
        st.session_state.pending_voice_input = None
    if "voice_key_counter" not in st.session_state:
        st.session_state.voice_key_counter = 0

    for message in st.session_state.messages:
        avatar_graphic = "karen.png" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar_graphic):
            st.markdown(message["content"])
            if "figure" in message and message["figure"] is not None:
                st.plotly_chart(message["figure"], use_container_width=True)
            if "audio" in message and message["audio"] is not None:
                st.audio(message["audio"], format="audio/mp3", autoplay=False)

    # --- Input: voice or text ---
    user_input = None

    if voice_input_on:
        # Check if there's a pending voice input from a previous rerun
        if st.session_state.pending_voice_input is not None:
            user_input = st.session_state.pending_voice_input
            st.session_state.pending_voice_input = None

        # Use a dynamic key so we can reset the widget by incrementing the counter
        audio_value = st.audio_input(
            "🎙️ Speak your question",
            key=f"voice_prompt_{st.session_state.voice_key_counter}"
        )
        if audio_value and user_input is None:
            with st.spinner("Transcribing your question..."):
                transcribed = transcribe_audio_bytes(audio_value.getvalue())
            st.info(f"📝 Heard: *{transcribed}*")
            # Store the transcription and bump the key counter to reset the widget
            st.session_state.pending_voice_input = transcribed
            st.session_state.voice_key_counter += 1
            st.rerun()
    else:
        user_input = st.chat_input("Ask about your data (e.g., 'Plot the top 10 categories')")

    # --- Process input ---
    if user_input:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant", avatar="karen.png"):
            with st.spinner("Analyzing data and generating code..."):
                try:
                    output_text, generated_fig = run_agent(agent_executor, user_input)
                    st.markdown(output_text)

                    if generated_fig:
                        st.plotly_chart(generated_fig, use_container_width=True)

                    audio_bytes = None
                    if voice_output_on:
                        with st.spinner("Generating voice response..."):
                            audio_bytes = text_to_speech(output_text)
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": output_text,
                        "figure": generated_fig,
                        "audio": audio_bytes,
                    })

                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # Force a clean rerun so the audio input widget re-renders fresh.
        # If voice output is active, wait for the audio to finish playing first.
        if voice_input_on:
            if audio_bytes:
                import time
                try:
                    from mutagen.mp3 import MP3
                    tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    tmp_mp3.write(audio_bytes)
                    tmp_mp3.close()
                    duration = MP3(tmp_mp3.name).info.length
                    os.remove(tmp_mp3.name)
                except Exception:
                    duration = len(audio_bytes) / 2000
                time.sleep(duration + 0.25)
            st.session_state.voice_key_counter += 1
            st.rerun()


def main():
    configure_chatbot()

main()