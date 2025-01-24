import os
import asyncio
import streamlit as st
from contextlib import asynccontextmanager
import atexit
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser  # Assuming these are your custom classes

# --- Async Context Management ---
@asynccontextmanager
async def managed_browser():
    """Properly managed browser context"""
    browser = Browser()
    async with browser.new_context() as context:
        yield context

# --- Session State Management ---
def setup_session_state():
    """Initialize or reset session state"""
    if 'browser' not in st.session_state:
        st.session_state.browser = None
    if 'context' not in st.session_state:
        st.session_state.context = None
    if 'model' not in st.session_state:
        st.session_state.model = ChatOpenAI(model='gpt-4o')

# --- Resource Cleanup ---
async def async_cleanup():
    """Properly close browser context"""
    if st.session_state.context:
        await st.session_state.context.__aexit__(None, None, None)
        st.session_state.context = None
        st.session_state.browser = None

def register_cleanup():
    """Register cleanup with Streamlit's shutdown"""
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_cleanup())

# --- Main App ---
def main():
    st.set_page_config(
        page_title="Browser Automation Suite",
        page_icon="🌐",
        layout="wide"
    )
    setup_session_state()
    atexit.register(register_cleanup)  # Cleanup on app exit

    st.title("Multi-Agent Browser Automation")
    
    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        os.environ["OPENAI_API_KEY"] = api_key

        if st.button("Initialize Browser"):
            try:
                # Create new event loop for async operations
                loop = asyncio.new_event_loop()
                st.session_state.context = loop.run_until_complete(
                    managed_browser().__aenter__()
                )
                st.success("Browser initialized!")
            except Exception as e:
                st.error(f"Initialization failed: {str(e)}")

    # --- Main Interface ---
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Agent Control")
        task1 = st.text_input(
            "Agent 1 Task:",
            "Open 2 tabs with Wikipedia articles about the history of the meta and one random Wikipedia article"
        )
        task2 = st.text_input(
            "Agent 2 Task:",
            "Considering all open tabs give me the names of the Wikipedia articles"
        )

        if st.button("Run Agents"):
            if not st.session_state.context:
                st.error("Initialize browser first!")
                return

            async def execute_agents():
                agent1 = Agent(
                    task=task1,
                    llm=st.session_state.model,
                    browser_context=st.session_state.context
                )
                agent2 = Agent(
                    task=task2,
                    llm=st.session_state.model,
                    browser_context=st.session_state.context
                )
                return await asyncio.gather(
                    agent1.run(),
                    agent2.run()
                )

            try:
                # Execute in a new event loop
                loop = asyncio.new_event_loop()
                results = loop.run_until_complete(execute_agents())
                
                st.session_state.agent1_result = results[0]
                st.session_state.agent2_result = results[1]
                st.success("Agents completed successfully!")
                
            except Exception as e:
                st.error(f"Agent execution failed: {str(e)}")

    with col2:
        st.subheader("Execution Results")
        
        if 'agent1_result' in st.session_state:
            with st.expander("Agent 1 Details", expanded=True):
                st.json({
                    "Task": task1,
                    "Actions": st.session_state.agent1_result['actions'],
                    "Response": st.session_state.agent1_result['response']
                })

        if 'agent2_result' in st.session_state:
            with st.expander("Agent 2 Details", expanded=True):
                st.json({
                    "Task": task2,
                    "Actions": st.session_state.agent2_result['actions'],
                    "Response": st.session_state.agent2_result['response']
                })

if __name__ == "__main__":
    main()
