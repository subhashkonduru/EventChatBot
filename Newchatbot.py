import streamlit as st
import os
#Test
# Call set_page_config as the very first Streamlit command
st.set_page_config(page_title="AI Event Bot", page_icon="🤖", layout="wide")

# Now import other modules, including your own
import utils  # For initializing data files, agenda, etc.
import resume_processor
import qa_handler
import google.generativeai as genai # For API key configuration check

def main():
    # st.set_page_config() has been moved to the top of the script

    st.title("🤖 AI-Powered Event Bot")
    st.caption("Welcome to your AI assistant! I can help you with event information, resume analysis, and more.")

    # Ensure data directory and default files are created
    utils.initialize_data_files() # This also ensures DATA_DIR exists for temp PDF agenda

    # --- Initialize session state variables ---
    if "google_api_key" not in st.session_state:
        st.session_state.google_api_key = ""
    if "messages" not in st.session_state:
        welcome_message = """👋 Hello! I'm your AI Event Assistant. I can help you with:

• 📅 Event schedule and agenda
• 📍 Location information
• 📄 Resume analysis
• 💭 General questions

How can I assist you today?"""

        st.session_state.messages = [{
            "role": "assistant",
            "content": welcome_message
        }]
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = {} # Stores feedback for each message index {idx: 'up'/'down'}
    if "current_suggestions" not in st.session_state:
        st.session_state.current_suggestions = []
    if "suggestions_for_idx" not in st.session_state: # To know which message suggestions belong to
        st.session_state.suggestions_for_idx = -1

    # --- API Key Input and Management ---    
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key_input = st.text_input(
            "Enter Google API Key",
            type="password",
            value=st.session_state.google_api_key,
            help="Your Google API key is required to enable AI features. It will be stored securely in your session."
        )
        
        if api_key_input:
            if st.session_state.google_api_key != api_key_input:
                st.session_state.google_api_key = api_key_input
                try:
                    genai.configure(api_key=st.session_state.google_api_key)
                    st.success("Google API Key configured!")
                    # Attempt to load vector store if API key is newly provided or changed
                    if "vector_store" not in st.session_state and st.session_state.google_api_key:
                        with st.spinner("Loading existing resume data..."):
                            st.session_state.vector_store = resume_processor.get_existing_vector_store(st.session_state.google_api_key)
                except Exception as e:
                    st.error(f"Invalid Google API Key or configuration error: {e}")
                    st.session_state.google_api_key = "" # Reset if invalid
        
        if not st.session_state.google_api_key:
            st.warning("Please enter your Google API Key to enable all features.")

    # --- Display Agenda Source Info --- 
    with st.sidebar:
        st.markdown("---")
        st.subheader("📅 Agenda Status")
        if "agenda_source_message" not in st.session_state:
            _ = utils.get_agenda_string() 
        
        agenda_msg = st.session_state.get("agenda_source_message", "Checking agenda source...")
        if st.session_state.get("is_pdf_agenda_active", False):
            st.info(agenda_msg)
        elif "not found" in agenda_msg.lower() or "failed" in agenda_msg.lower():
            st.warning(agenda_msg)
        else:
            st.info(agenda_msg)
        st.caption("💡 Tip: Place your agenda PDF in the data directory to use a custom schedule.")

    with st.sidebar:
        st.markdown("---")
        st.subheader("🎯 Workshop Information")
        if os.path.exists(utils.WORKSHOPS_PDF_FILE):
            st.success("✅ Workshop details loaded from workshops.pdf")
        else:
            st.info("💡 Add workshops.pdf to the data directory for detailed workshop information")
            if os.path.exists(utils.AGENDA_FILE):
                st.caption("Currently using workshop information from the agenda")

    with st.sidebar:
        st.markdown("---")
        st.header("📄 Attendee Resumes")
        if st.session_state.google_api_key:
            if st.button("🔄 Process Resumes", help=f"Click to process resumes from the '{resume_processor.RESUMES_DIR_NAME}' folder"):
                with st.spinner("📝 Processing resumes..."):
                    vs = resume_processor.process_resumes_from_folder(st.session_state.google_api_key)
                    if vs:
                        st.session_state.vector_store = vs
                st.rerun() 
            
            if "vector_store" in st.session_state and st.session_state.vector_store:
                st.success("✅ Resume data loaded and ready for analysis!")
            else:
                st.info("ℹ️ No resumes processed yet. Click the button above to start.")
                st.caption("📁 Add your PDF or DOCX resumes to the 'resumes' folder first.")
                if not os.path.exists(resume_processor.RESUMES_DIR_NAME):
                    st.warning("⚠️ The 'resumes' directory doesn't exist yet. Create it to get started!")
        else:
            st.info("🔑 Enter your API key above to enable resume features")

    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Quick Resume Queries")
        if st.session_state.google_api_key and st.session_state.get("vector_store"):
            resume_prompts = {
                "📅 Recommended Sessions": "Based on my background and skills, which workshop sessions would be most beneficial for me to attend?",
                "👨‍💻 Technical Skills": "Find participants with specific technical skills like Python, JavaScript, or cloud technologies",
                "🎓 Education Background": "Find participants based on their educational background and qualifications",
                "💼 Work Experience": "Find participants with specific work experience or years of experience",
                "📊 Data & ML Experience": "Find participants with experience in data analysis, machine learning, or AI projects",
                "🌐 Full-Stack Development": "Find participants with full-stack development experience",
                "☁️ Cloud Technologies": "Find participants with experience in AWS, Docker, or cloud technologies",
                "📊 Data Engineer Profiles": "Identify participants whose resumes align with a Data Engineer role, look for skills like ETL, data pipelines, SQL, Python, Spark, and cloud data platforms."
            }
            for btn_label, prompt_text in resume_prompts.items():
                if st.button(btn_label, key=f"resume_prompt_{btn_label.replace(' ', '_')}"):
                    st.session_state.messages.append({"role": "user", "content": prompt_text})
                    with st.spinner("Analyzing resumes and sessions..."):
                        response = qa_handler.get_bot_response(
                            prompt_text,
                            api_key=st.session_state.google_api_key,
                            vector_store_instance=st.session_state.get("vector_store"),
                            chat_history=st.session_state.messages
                        )
                    assistant_reply_content = response.get("answer", "Sorry, I couldn't process that query effectively.")
                    assistant_debug_info = response.get("debug_info", "No debug information available for this query.")
                    st.session_state.messages.append({"role": "assistant", "content": assistant_reply_content, "debug_info": assistant_debug_info})
                    st.session_state.current_suggestions = [] # Clear suggestions on new query
                    st.session_state.suggestions_for_idx = -1
                    st.rerun()
        elif not st.session_state.google_api_key:
            st.caption("🔑 Add your API key to unlock resume analysis features")
        else:
            st.caption("📤 Process some resumes first to enable these queries")

    if "vector_store" not in st.session_state and st.session_state.google_api_key:
        with st.spinner("Checking for existing resume data..."):
             st.session_state.vector_store = resume_processor.get_existing_vector_store(st.session_state.google_api_key)

    # --- Handle suggestion generation if a thumbs down was clicked ---
    if "generate_suggestions_for_idx" in st.session_state and st.session_state.generate_suggestions_for_idx != -1:
        idx = st.session_state.generate_suggestions_for_idx
        if idx > 0 and idx < len(st.session_state.messages):
            user_msg_content = st.session_state.messages[idx-1]['content']
            bot_msg_obj = st.session_state.messages[idx]
            
            with st.spinner("Generating alternative prompts..."):
                suggestions = qa_handler.get_alternative_prompts(
                    user_msg_content, 
                    bot_msg_obj['content'], 
                    bot_msg_obj.get('debug_info'),
                    st.session_state.google_api_key # Pass API key
                )
            st.session_state.current_suggestions = suggestions
            st.session_state.suggestions_for_idx = idx # Store which message these suggestions are for
        del st.session_state.generate_suggestions_for_idx # Clear the trigger

    # --- Chat Interface ---    
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Replace \n with proper line breaks for markdown
            formatted_content = message["content"].replace("\\n", "\n")
            st.markdown(formatted_content)
            if message["role"] == "assistant":
                # Feedback buttons
                col1, col2, col_spacer = st.columns([1, 1, 10])
                with col1:
                    if st.button("👍", key=f"thumb_up_{i}", help="Good response!"):
                        st.session_state.feedback_data[i] = "up"
                        st.session_state.current_suggestions = []
                        st.session_state.suggestions_for_idx = -1
                        st.rerun() 
                with col2:
                    if st.button("👎", key=f"thumb_down_{i}", help="Needs improvement"):
                        st.session_state.feedback_data[i] = "down"
                        st.session_state.generate_suggestions_for_idx = i
                        st.rerun()
                
                # Display debug info if available
                if "debug_info" in message and message["debug_info"]:
                    with st.expander("Show Debug Info"):
                        st.text(message["debug_info"])
                
                # Display alternative suggestions if they exist for this message
                if st.session_state.suggestions_for_idx == i and st.session_state.current_suggestions:
                    st.markdown("--- Suggested Prompts ---")
                    for sugg in st.session_state.current_suggestions:
                        st.markdown(f"- *{sugg}*")
                    # Add a button to clear suggestions manually if needed, or they clear on new msg/thumb up
                    if st.button("Clear Suggestions", key=f"clear_sugg_{i}"):
                        st.session_state.current_suggestions = []
                        st.session_state.suggestions_for_idx = -1
                        st.rerun()

    if prompt := st.chat_input("Ask me anything about the event..."):
        if not st.session_state.google_api_key:
            st.warning("Please enter your Google API Key in the sidebar to use the bot.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Clear previous suggestions before getting a new response
        st.session_state.current_suggestions = []
        st.session_state.suggestions_for_idx = -1

        with st.spinner("Thinking..."):
            response_data = qa_handler.get_bot_response(
                prompt, 
                api_key=st.session_state.google_api_key,
                vector_store_instance=st.session_state.get("vector_store"),
                chat_history=st.session_state.messages
            )
        
        bot_answer = response_data.get("answer", "Sorry, I encountered an issue and couldn't form a proper response.")
        bot_debug_info = response_data.get("debug_info", "No debug information available.")

        with st.chat_message("assistant"):
            st.markdown(bot_answer)
            if bot_debug_info: 
                with st.expander("Show Debug Info"):
                    st.text(bot_debug_info)
        
        st.session_state.messages.append({"role": "assistant", "content": bot_answer, "debug_info": bot_debug_info})
        st.rerun() # Rerun to display new messages and feedback buttons correctly
    
    # --- Quick Actions in Sidebar (Modified to clear suggestions) ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        col1, col2 = st.columns(2)
        
        def handle_quick_action(query_text):
            if not st.session_state.google_api_key:
                st.warning("🔑 Please enter API key first")
                return
            response_data = qa_handler.get_bot_response(
                query_text, 
                api_key=st.session_state.google_api_key,
                vector_store_instance=st.session_state.get("vector_store"),
                chat_history=st.session_state.messages
            )
            action_info = response_data.get("answer", f"Could not retrieve {query_text.lower()}.")
            st.session_state.messages.append({"role": "assistant", "content": action_info, "debug_info": response_data.get("debug_info")})
            st.session_state.current_suggestions = [] # Clear suggestions
            st.session_state.suggestions_for_idx = -1
            st.rerun()

        with col1:
            if st.button("📅 Agenda", help="View the event schedule"):
                handle_quick_action("What's the agenda?")
            
            if st.button("🚽 Washroom", help="Find washroom locations"):
                handle_quick_action("Where is the washroom?")
        
        with col2:
            if st.button("🍽️ Time to Lunch", help="Check time until lunch"):
                handle_quick_action("How long until lunch?")
            
            if st.button("💭 Feedback", help="Share your thoughts about the event"):
                initial_feedback_prompt = "I'd like to provide feedback about the event."
                st.session_state.messages.append({"role": "user", "content": initial_feedback_prompt})
                with st.spinner("Starting feedback collection..."):
                    response_data = qa_handler.get_bot_response(
                        initial_feedback_prompt,
                        api_key=st.session_state.google_api_key,
                        vector_store_instance=st.session_state.get("vector_store"),
                        chat_history=st.session_state.messages
                    )
                bot_answer = response_data.get("answer", "Sorry, I couldn't process that feedback request.")
                bot_debug_info = response_data.get("debug_info", "No debug info for feedback init.")
                st.session_state.messages.append({"role": "assistant", "content": bot_answer, "debug_info": bot_debug_info})
                st.session_state.current_suggestions = [] # Clear suggestions
                st.session_state.suggestions_for_idx = -1
                st.rerun()

if __name__ == "__main__":
    main()
