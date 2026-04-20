# app.py
import streamlit as st
import duckdb
import os
import plotly.io as pio
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import file_manipulation as fm

from dotenv import load_dotenv
load_dotenv()

def configure_chatbot():

    if "app_initialized" not in st.session_state:
        if os.path.exists("plot.json"):
            os.remove("plot.json")
        st.session_state.app_initialized = True

    st.set_page_config(page_title="Data Chatbot", layout="wide")

    col1, col2 = st.columns([1, 15])
    with col1:
        st.image("karen.png", width=200)
    with col2:
        st.title("Hi, I am Karen, your complaints data analyzer. How can I help? 📊")

    # --- Configuration ---
    DATA_GLOB = "data/*.parquet"   # picks up data.parquet + every recording_*.parquet
    llm = ChatOpenAI(temperature=0, model="gpt-4o")

    # --- Agent Setup ---
    tools = [PythonREPLTool()]

    instructions = f"""
        You are an expert data analyst. You have access to a folder of Parquet files
        matched by the glob pattern '{DATA_GLOB}' containing consumer financial complaints.
        Some rows come from the original dataset; others were transcribed from audio recordings.

        Here is the schema of the dataset:
        - Unnamed: 0 (Index, may be NULL in transcribed rows)
        - Date received (Format: YYYY-MM-DD). Some file might include time down to the second.
        - Product
        - Sub-product
        - Issue
        - Sub-issue
        - Consumer complaint narrative
        - Company public response
        - Company
        - State
        - ZIP code
        - Tags
        - Consumer consent provided?
        - Submitted via
        - Date sent to company
        - Company response to consumer
        - Timely response?
        - Consumer disputed?
        - Complaint ID

        CRITICAL DUCKDB INSTRUCTIONS:
        1. NEVER use pandas to read the parquet files.
        2. Query ALL files at once using a glob like this:
           `duckdb.sql('SELECT "Company" FROM "{DATA_GLOB}" LIMIT 5').df()`
        3. Because column names contain spaces, you MUST wrap them in double quotes in your SQL.
        4. Columns not present in transcribed rows will be NULL — handle them gracefully (e.g. use COALESCE or IS NOT NULL filters when needed).

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
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar_graphic = "karen.png" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar_graphic):
            st.markdown(message["content"])
            if "figure" in message and message["figure"] is not None:
                st.plotly_chart(message["figure"], use_container_width=True)

    user_input = st.chat_input("Ask about your data (e.g., 'Plot the top 10 categories')")

    if user_input:
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant", avatar="karen.png"):
            with st.spinner("Analyzing data and generating code..."):
                try:
                    response = agent_executor.invoke({"input": user_input})
                    output_text = response["output"]
                    st.markdown(output_text)

                    generated_fig = None
                    if os.path.exists("plot.json"):
                        with open("plot.json", "r") as f:
                            generated_fig = pio.from_json(f.read())
                        st.plotly_chart(generated_fig, use_container_width=True)
                        os.remove("plot.json")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": output_text,
                        "figure": generated_fig
                    })

                except Exception as e:
                    st.error(f"An error occurred: {e}")


def main():
    configure_chatbot()

main()
