import os
import streamlit as st
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser
import asyncio

# Set page config
st.set_page_config(
    page_title="Browser Automation Suite",
    page_icon="🌐",
    layout="wide"
)

def setup_session_state():
    """Initialize session state variables"""
    if 'browser' not in st.session_state:
        st.session_state.browser = Browser()
    if 'context' not in st.session_state:
        st.session_state.context = None
    if 'model' not in st.session_state:
        st.session_state.model = ChatOpenAI(model='gpt-4o')

async def run_agent(task, context):
    """Run an agent with the given task"""
    agent = Agent(
        task=task,
        llm=st.session_state.model,
        browser_context=context
    )
    return await agent.run()

def main():
    setup_session_state()
    
    st.title("Multi-Agent Browser Automation")
    st.markdown("Control multiple AI agents to perform browser tasks simultaneously")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        os.environ["OPENAI_API_KEY"] = api_key
        
        if st.button("Initialize Browser Session"):
            st.session_state.context = asyncio.run(st.session_state.browser.new_context().__aenter__())
            st.success("Browser session initialized!")

    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Agent Control Panel")
        
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
                st.error("Please initialize browser session first!")
                return
                
            with st.spinner("Running agents..."):
                # Run agents asynchronously
                async def run_all_agents():
                    results = await asyncio.gather(
                        run_agent(task1, st.session_state.context),
                        run_agent(task2, st.session_state.context)
                    )
                    return results
                
                results = asyncio.run(run_all_agents())
                
                st.session_state.agent1_result = results[0]
                st.session_state.agent2_result = results[1]
                
            st.success("Agents completed their tasks!")

    with col2:
        st.subheader("Execution Results")
        
        if 'agent1_result' in st.session_state:
            with st.expander("Agent 1 Actions", expanded=True):
                st.write("**Task:**", task1)
                st.write("**Browser Actions:**")
                st.code(st.session_state.agent1_result['actions'], language="json")
                st.write("**Final Response:**")
                st.markdown(f"> {st.session_state.agent1_result['response']}")

        if 'agent2_result' in st.session_state:
            with st.expander("Agent 2 Actions", expanded=True):
                st.write("**Task:**", task2)
                st.write("**Browser Actions:**")
                st.code(st.session_state.agent2_result['actions'], language="json")
                st.write("**Final Response:**")
                st.markdown(f"> {st.session_state.agent2_result['response']}")

if __name__ == "__main__":
    main()
