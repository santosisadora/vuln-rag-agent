import streamlit as st
import requests

# 1. Configure the UI page
st.set_page_config(page_title="Vulnerability Agent", page_icon="🛡️")
st.title("🛡️ SecOps Vulnerability Agent")
st.caption("Powered by LangGraph, FastAPI, and AWS ECR")

# 2. Initialize chat history in the session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I am your SecOps assistant. What CVEs or security policies are we analyzing today?"}
    ]

# 3. Display previous chat messages when the app reruns
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. React to user input
if prompt := st.chat_input("Ask about a CVE or security policy..."):

    # Display the user's message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display an empty assistant message while waiting for the backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Analyzing vulnerabilities...*")

        try:
            # 1. Update the route AND enable streaming
            response = requests.post(
                "http://localhost:8000/triage/stream",  # <-- The correct FastAPI route!
                json={"query": prompt},
                stream=True  # <-- Tells Python to keep the connection open
            )
            response.raise_for_status()

            full_response = ""

            # 2. Loop through the incoming stream of Server-Sent Events
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    # Look for the "data: " prefix you defined in main.py
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        import json

                        data = json.loads(data_str)

                        # 3. If it's a token, add it to the UI instantly
                        if data["type"] == "token":
                            raw_content = data["content"]

                            # Handle standard string tokens
                            if isinstance(raw_content, str):
                                full_response += raw_content

                            # Handle complex LLM block tokens (lists)
                            elif isinstance(raw_content, list):
                                for item in raw_content:
                                    if isinstance(item, dict) and "text" in item:
                                        full_response += item["text"]
                                    elif isinstance(item, str):
                                        full_response += item

                            # The "▌" creates a cool blinking cursor effect!
                            message_placeholder.markdown(full_response + "▌")

                        # (Optional) You could also print data["type"] == "tool"
                        # using st.toast() or st.caption() to show what the agent is doing!

            # 4. Remove the cursor when finished
            message_placeholder.markdown(full_response)

            # Save the final answer to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Backend Connection Error: Make sure FastAPI is running! ({e})")