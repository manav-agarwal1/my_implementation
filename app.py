import streamlit as st
from llm.model_handler import LLMHandler
from docs.docs_processor import DocsProcessor

def initialize_session_state():
    if "llm" not in st.session_state:
        st.session_state.llm = LLMHandler()
    if "docs_processor" not in st.session_state:
        with st.spinner("Loading documentation..."):
            try:
                st.session_state.docs_processor = DocsProcessor()
                st.session_state.docs_processor.process_docs()
            except Exception as e:
                st.error(f"Error loading documentation: {str(e)}")
                st.session_state.docs_processor = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.set_page_config(page_title="Crustdata API Q&A Bot", page_icon="🤖")
    
    st.title("Crustdata API Q&A Bot")
    st.write("Ask questions about Crustdata's APIs and get instant answers!")
    
    # Initialize components
    initialize_session_state()
    
    # Chat interface
    user_question = st.text_input("Your question:", key="user_input")
    
    # Add debug information in sidebar
    with st.sidebar:
        st.subheader("Debug Information")
        if st.session_state.docs_processor:
            st.write(f"Number of processed docs: {len(st.session_state.docs_processor.docs)}")
            if st.checkbox("Show sample docs"):
                st.write("Sample documentation chunks:")
                for i in range(min(3, len(st.session_state.docs_processor.docs))):
                    st.code(st.session_state.docs_processor.docs[i])
                    
    if st.button("Ask"):
        if user_question:
            with st.spinner("Thinking..."):
                # Find relevant context from docs
                context = st.session_state.docs_processor.find_relevant_context(user_question)
                
                # Generate response
                response = st.session_state.llm.generate_response(user_question, context)
                
                # Add to chat history
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response))
    
    # Display chat history
    st.subheader("Conversation History")
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.write(f"**You:** {message}")
        else:
            st.write(f"**Bot:** {message}")

if __name__ == "__main__":
    main()