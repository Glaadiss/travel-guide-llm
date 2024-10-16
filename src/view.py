import streamlit as st


def view(run_llm):
    st.title("✈️ 🧳 Vietnam Travel Guide 🇻🇳🏞️")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Add spinner while waiting for LLM response
        with st.spinner("Thinking..."):
            # Call LLM function
            result = run_llm(prompt, st.session_state.messages)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(result["output"])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result["output"]})
