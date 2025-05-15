import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import datetime
import re

# Assuming utils.py and resume_processor.py are in the same directory
import utils
import resume_processor 

# It's good practice to load API key from environment variables
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     st.error("GOOGLE_API_KEY environment variable not set for QA Handler.")
# else:
#     genai.configure(api_key=GOOGLE_API_KEY)

# --- Constants for Prompts ---
GENERAL_SYSTEM_PROMPT = """You are a friendly and helpful AI Event Bot for a workshop. 
Your goal is to assist participants with useful, real-time information about the event. 
Be conversational and proactive. If you use information from resumes, mention it discreetly.
If you don't know the answer, say so politely. Do not make up information.
Today's date is {current_date}.
"""

RESUME_CONTEXT_PROMPT = """Based on the provided resume context and the user's question, answer the user. 
If the context doesn't fully answer the question, use your general knowledge about event interactions but prioritize the context.
Context from resumes: {context}
Question: {question}
Answer:"""

FEEDBACK_SYSTEM_PROMPT = """You are a friendly AI assistant collecting feedback about a workshop session. 
Be conversational and try to elicit specific positive and negative points. 
Ask open-ended follow-up questions if the user gives a very short answer. 
Example: Instead of just 'Did you like it?', ask 'What did you find most valuable about the session?' or 'Were there any areas you felt could be improved?'
Start the conversation by asking for general feedback about the session they just attended.
"""

# --- Module-Level Constants ---

SKILL_CATEGORIES = {
    "Programming Languages": {
        "python": ["python", "py", "python3", "pytorch", "pandas"],
        "java": ["java", "java8", "java11", "jvm", "spring", "hibernate"],
        "javascript": ["javascript", "js", "es6", "typescript", "ts", "node.js", "nodejs", "react", "angular", "vue"],
        "c++": ["c++", "cpp", "c plus plus"],
        "c#": ["c#", "csharp", ".net", "dotnet", "asp.net"],
        "ruby": ["ruby", "ruby on rails", "rails", "rb"],
        "go": ["golang", "go lang"],
        "php": ["php", "php7", "php8", "laravel", "symfony"],
        "swift": ["swift", "ios development", "xcode"],
        "kotlin": ["kotlin", "android development"],
        "rust": ["rust", "rustlang"],
        "scala": ["scala", "spark"]
    },
    "Web Technologies": {
        "frontend": ["frontend", "front-end", "front end", "html", "css", "sass", "scss", "bootstrap"],
        "react": ["react", "reactjs", "react.js", "redux", "hooks"],
        "angular": ["angular", "angularjs", "angular2+", "ng"],
        "vue": ["vue", "vuejs", "vue.js", "vuex"],
        "node": ["node", "nodejs", "node.js", "express", "nestjs"],
        "django": ["django", "django rest", "drf"],
        "flask": ["flask", "flask-restful"],
        "html/css": ["html", "html5", "css", "css3", "sass", "scss", "less"],
        "web components": ["web components", "custom elements", "shadow dom"],
        "webpack": ["webpack", "babel", "rollup"]
    },
    "Databases & Storage": {
        "sql": ["sql", "mysql", "postgresql", "postgres", "oracle", "pl/sql"],
        "nosql": ["nosql", "mongodb", "mongo", "dynamodb", "cassandra", "couchbase"],
        "redis": ["redis", "cache", "in-memory"],
        "elasticsearch": ["elasticsearch", "elk", "kibana", "solr"],
        "graphql": ["graphql", "apollo", "hasura"],
        "firebase": ["firebase", "firestore", "realtime database"]
    },
    "Cloud & DevOps": {
        "aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "cloudformation"],
        "azure": ["azure", "microsoft azure", "azure functions"],
        "gcp": ["gcp", "google cloud", "app engine", "cloud functions"],
        "docker": ["docker", "container", "containerization"],
        "kubernetes": ["kubernetes", "k8s", "helm"],
        "ci/cd": ["ci/cd", "jenkins", "gitlab ci", "github actions", "travis"],
        "terraform": ["terraform", "infrastructure as code", "iac"],
        "monitoring": ["prometheus", "grafana", "datadog", "newrelic"]
    },
    "AI & Data Science": {
        "machine learning": ["machine learning", "ml", "deep learning", "neural networks", "ai"],
        "data science": ["data science", "data analysis", "data analytics", "statistics"],
        "frameworks": ["tensorflow", "pytorch", "keras", "scikit-learn", "sklearn"],
        "nlp": ["nlp", "natural language processing", "nltk", "spacy"],
        "computer vision": ["computer vision", "opencv", "image processing"],
        "big data": ["hadoop", "spark", "kafka", "big data", "data pipeline"]
    },
    "Tools & Practices": {
        "version control": ["git", "github", "gitlab", "bitbucket", "svn"],
        "agile": ["agile", "scrum", "kanban", "jira"],
        "testing": ["unit testing", "integration testing", "jest", "pytest", "selenium"],
        "api": ["rest", "restful", "api", "swagger", "openapi"],
        "security": ["security", "authentication", "authorization", "oauth", "jwt"],
        "microservices": ["microservices", "service mesh", "api gateway"]
    }
}

# Helper to initialize the LLM
def get_llm(api_key, temperature=0.7, model_name="gemini-2.0-flash"):
    if not api_key:
        st.error("Google API Key not provided to LLM initializer.")
        return None
    try:
        # Ensure genai is configured with the provided API key.
        # The genai.configure() call can be made multiple times; the SDK handles it.
        genai.configure(api_key=api_key)
        
        # Explicitly pass the API key to the ChatGoogleGenerativeAI constructor
        llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temperature, 
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def get_target_keywords_for_query(skill_query_text, skill_map):
    query_lower = skill_query_text.lower().strip()
    target_keywords = set()

    # Check if query matches a specific skill or its variations directly
    for category, skills_in_category in skill_map.items():
        for skill_name, variations in skills_in_category.items():
            if query_lower == skill_name.lower() or query_lower in [v.lower() for v in variations]:
                target_keywords.update(v.lower() for v in variations) # Add all variations for that specific skill
                print(f"[DEBUG] Query '{skill_query_text}' matched specific skill '{skill_name}'. Keywords: {list(target_keywords)}")
                return list(target_keywords) 

    # Check if query matches a broader category concept (heuristic based on keywords in query)
    category_mapping = {
        "cloud": "Cloud & DevOps", "aws": "Cloud & DevOps", "azure": "Cloud & DevOps", "gcp": "Cloud & DevOps", "docker": "Cloud & DevOps", "kubernetes": "Cloud & DevOps", "devops": "Cloud & DevOps", "containers": "Cloud & DevOps",
        "database": "Databases & Storage", "databases": "Databases & Storage", "sql": "Databases & Storage", "nosql": "Databases & Storage", "storage": "Databases & Storage", "mongodb": "Databases & Storage", "postgresql": "Databases & Storage",
        "programming": "Programming Languages", "language": "Programming Languages", "languages": "Programming Languages", "coding": "Programming Languages", "scripting": "Programming Languages",
        "web": "Web Technologies", "frontend": "Web Technologies", "front-end": "Web Technologies", "backend": "Web Technologies", "back-end": "Web Technologies", "fullstack": "Web Technologies", "full-stack": "Web Technologies", "website": "Web Technologies", "http": "Web Technologies",
        "ai": "AI & Data Science", "artificial intelligence": "AI & Data Science", "ml": "AI & Data Science",
        "machine learning": "AI & Data Science", "deep learning": "AI & Data Science", "neural network": "AI & Data Science",
        "data science": "AI & Data Science", "data analysis": "AI & Data Science", "analytics": "AI & Data Science", "statistics": "AI & Data Science", "big data": "AI & Data Science",
        "security": "Tools & Practices", "cybersecurity": "Tools & Practices", "auth": "Tools & Practices", "authentication": "Tools & Practices",
        "testing": "Tools & Practices", "qa": "Tools & Practices", "quality assurance": "Tools & Practices", "test automation": "Tools & Practices",
        "agile": "Tools & Practices", "scrum": "Tools & Practices", "kanban": "Tools & Practices", "jira": "Tools & Practices",
        "mobile": "Programming Languages", "android": "Programming Languages", "ios": "Programming Languages", # Assumes mobile skills (Swift, Kotlin etc.) are in Prog Langs or a dedicated mobile category in SKILL_CATEGORIES
        "api": "Tools & Practices", "apis": "Tools & Practices", "rest": "Tools & Practices", "graphql": "Databases & Storage" # GraphQL often associated with DBs/data access but also APIs
    }

    matched_category_name = None
    for keyword, cat_name in category_mapping.items():
        if keyword in query_lower:
            matched_category_name = cat_name
            break
    
    if matched_category_name and matched_category_name in skill_map:
        print(f"[DEBUG] Query '{skill_query_text}' matched category '{matched_category_name}'.")
        for skill_name, variations in skill_map[matched_category_name].items():
            target_keywords.update(v.lower() for v in variations)
    
    if target_keywords:
        return list(target_keywords)

    # Fallback: if no specific mapping, use the query words themselves
    print(f"[DEBUG] Query '{skill_query_text}' using fallback keywords: {query_lower.split()}")
    return query_lower.split() # Split the query into words as a last resort

def get_details_for_all_resumes(vector_store_instance, detail_type_for_display, extraction_query_template, llm, debug_accumulator_parent=None):
    """
    Iterates through all resumes, extracts specific details using an LLMChain with a given query template,
    and formats the results for display.
    """
    local_debug_info = debug_accumulator_parent if debug_accumulator_parent is not None else []
    local_debug_info.append(f"[FUNC_CALL] get_details_for_all_resumes (Detail Type: '{detail_type_for_display}')")
    local_debug_info.append(f"  Extraction Template: {extraction_query_template}")

    if not vector_store_instance:
        answer = "I need access to resume information to extract details. Please process resumes first."
        local_debug_info.append("[ERROR] Vector store not available for detail extraction.")
        return {"result": answer, "debug_info_list": local_debug_info}
    if not llm:
        answer = "LLM not available, cannot extract details."
        local_debug_info.append("[ERROR] LLM not available for detail extraction.")
        return {"result": answer, "debug_info_list": local_debug_info}

    print(f"[DEBUG] Extracting '{detail_type_for_display}' for ALL resumes.")
    
    all_resumes_query = "resume name contact information"
    all_resume_docs_for_sources = vector_store_instance.similarity_search(all_resumes_query, k=200)
    local_debug_info.append(f"  Retrieved {len(all_resume_docs_for_sources)} docs to find unique sources using query: '{all_resumes_query}' (k=200)")

    unique_sources = sorted(list(set(
        doc.metadata.get('source') 
        for doc in all_resume_docs_for_sources 
        if doc.metadata.get('source')
    )))
    local_debug_info.append(f"  Unique sources found: {len(unique_sources)} -> {unique_sources if len(unique_sources) < 5 else str(unique_sources[:5]) + '...'}")

    if not unique_sources:
        answer = f"No resumes found in the vector store to extract {detail_type_for_display} from."
        local_debug_info.append("  No unique sources found.")
        return {"result": answer, "debug_info_list": local_debug_info}

    final_response = f"📄 **{detail_type_for_display} for All Processed Resumes**\\n\\n"
    any_details_found_overall = False

    # Define the prompt for the LLMChain
    prompt = PromptTemplate(template=extraction_query_template, input_variables=["context"])
    chain = LLMChain(llm=llm, prompt=prompt)
    local_debug_info.append(f"  LLMChain Initialized. Prompt input variables: {prompt.input_variables}")

    for source_file_name in unique_sources:
        participant_name = os.path.splitext(source_file_name)[0].replace('_', ' ').title()
        final_response += f"--- **{participant_name} ({source_file_name})** ---\\n"
        local_debug_info.append(f"  Processing source: {source_file_name} (Participant: {participant_name})")
        
        # Get comprehensive content for this specific resume
        source_filter = {"source": source_file_name}
        
        # --- Attempt a more targeted search first for the specific detail type ---
        targeted_search_query = f"{detail_type_for_display.lower()} section content details"
        if "education" in detail_type_for_display.lower():
            targeted_search_query = "education section academic qualifications degrees university graduation year"
        elif "work experience" in detail_type_for_display.lower():
            targeted_search_query = "work experience summary job titles companies responsibilities achievements duration"
        elif "skills" in detail_type_for_display.lower():
            targeted_search_query = "technical skills programming languages frameworks tools technologies summary"
        # Add more specific targeted queries for other detail_types if needed

        print(f"[DEBUG] Targeted search for '{detail_type_for_display}' in {source_file_name} with: '{targeted_search_query}'")
        targeted_docs = vector_store_instance.similarity_search(
            targeted_search_query, 
            k=10, # Get fewer, but hopefully more relevant, chunks
            filter=source_filter
        )
        local_debug_info.append(f"    Targeted search ('{targeted_search_query}', k=10) -> Got {len(targeted_docs)} docs.")
        
        # --- Then, do a broader search to supplement, especially if targeted search was insufficient ---
        broader_search_query = f"full resume content including {detail_type_for_display.lower()} projects skills summary education experience achievements responsibilities"
        print(f"[DEBUG] Broader search for '{detail_type_for_display}' in {source_file_name} with: '{broader_search_query}'")
        broader_docs = vector_store_instance.similarity_search(
            broader_search_query, 
            k=15, # Adjust k as needed, k=25 was used before
            filter=source_filter
        )
        local_debug_info.append(f"    Broader search ('{broader_search_query}', k=15) -> Got {len(broader_docs)} docs.")
        
        # Combine and de-duplicate documents from both searches
        all_retrieved_docs_for_llm = [] # Renamed
        seen_docs_content = set() # To avoid adding the exact same chunk content twice

        for doc_llm in targeted_docs + broader_docs: # Renamed
            if doc_llm.page_content not in seen_docs_content:
                all_retrieved_docs_for_llm.append(doc_llm)
                seen_docs_content.add(doc_llm.page_content)
        
        local_debug_info.append(f"    Total unique docs for LLM context for {participant_name}: {len(all_retrieved_docs_for_llm)}")
        print(f"[DEBUG] Total unique docs for {participant_name} for '{detail_type_for_display}': {len(all_retrieved_docs_for_llm)} (Targeted: {len(targeted_docs)}, Broader: {len(broader_docs)})")

        if not all_retrieved_docs_for_llm:
            final_response += f"I couldn't find enough information in this resume to extract a {detail_type_for_display.lower()}. This might happen if the resume is image-based, very short, or doesn't contain clearly identifiable sections for this type of information.\n\n"
            local_debug_info.append("    No docs for LLM context after combining searches.")
            continue

        # Combine page content of all retrieved documents for this resume
        resume_full_text_context = " \\n ".join([doc.page_content for doc in all_retrieved_docs_for_llm])
        context_preview_for_debug = resume_full_text_context[:300].replace("\\n", " ") + "..."
        local_debug_info.append(f"    Context for LLM (first 300 chars): {context_preview_for_debug}")
        # Limit context size to avoid exceeding token limits for the LLM, if necessary
        # max_context_length = 7000 # Example limit, depends on model (Gemini Flash has large context)
        # if len(resume_full_text_context) > max_context_length:
        #     resume_full_text_context = resume_full_text_context[:max_context_length]
            
        print(f"[DEBUG] Context for {participant_name} for '{detail_type_for_display}' (first 300 chars): {resume_full_text_context[:300]}")

        try:
            # Run the LLM chain to extract details
            extracted_details = chain.run(context=resume_full_text_context)
            local_debug_info.append(f"    LLMChain.run executed. Extracted details (raw): '{extracted_details[:100].replace('\\n',' ')}...'")
            
            if extracted_details and extracted_details.strip() and extracted_details.lower() not in ["not found", "none", "n/a", "not specified"]:
                final_response += f"{extracted_details.strip()}\n\n"
                any_details_found_overall = True
            else:
                final_response += f"While I could access the resume, I wasn't able to identify specific {detail_type_for_display.lower()} details within its content. The information might be missing or in a format I couldn't parse.\n\n"
        except Exception as e:
            print(f"[DEBUG] LLM extraction error for {participant_name} ('{detail_type_for_display}'): {e}")
            final_response += f"An error occurred while trying to extract {detail_type_for_display.lower()} for this resume. I'll skip this one and continue with others.\n\n"

    if not any_details_found_overall and unique_sources:
         final_response = f"I processed all resumes but couldn't extract significant {detail_type_for_display} details. Ensure resumes contain this information clearly."
         local_debug_info.append("  No details found overall for any resume.")
    
    local_debug_info.append("  Finished processing all resumes for detail extraction.")
    return {"result": final_response, "debug_info_list": local_debug_info}

# Main Q&A function
def get_bot_response(user_query, api_key, vector_store_instance=None, chat_history=None):
    debug_info_accumulator = [f"User Query: {user_query}"]
    query_lower = user_query.lower().strip() # ADDED DEFINITION HERE
    
    # Initialize chat memory if not already done
    if "chat_memory_main" not in st.session_state:
        st.session_state.chat_memory_main = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.session_state.chat_memory_main.output_key = "answer"  # Explicitly set the output key for memory

    # Initialize feedback mode if not already done
    if "feedback_mode" not in st.session_state:
        st.session_state.feedback_mode = False
    if "waiting_for_user_background_for_recommendation" not in st.session_state: # New state
        st.session_state.waiting_for_user_background_for_recommendation = False
    if "user_provided_background" not in st.session_state: # New state
        st.session_state.user_provided_background = None

    llm = get_llm(api_key)
    if not llm:
        return {
            "answer": "Sorry, I'm having trouble connecting to my brain right now. Please ensure the API key is set.",
            "debug_info": "\n".join(debug_info_accumulator) + "\n[ERROR] LLM not initialized."
        }

    # Load chat history into memory if provided
    if chat_history:
        # Clear existing memory to prevent duplication
        st.session_state.chat_memory_main.clear()
        for msg in chat_history[:-1]:  # Don't add the current user query yet
            if msg["role"] == "user":
                st.session_state.chat_memory_main.chat_memory.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                ai_message_content_to_add = ""
                if isinstance(msg.get("content"), dict) and "answer" in msg["content"]:
                    ai_message_content_to_add = msg["content"]["answer"]
                elif isinstance(msg.get("content"), str):
                    ai_message_content_to_add = msg["content"]
                else:
                    # Fallback for unexpected format, try to stringify, and log
                    ai_message_content_to_add = str(msg.get("content", ""))
                    print(f"[WARNING] Unexpected AI message content format in chat history, attempting to stringify: {msg.get('content')}")
                st.session_state.chat_memory_main.chat_memory.add_ai_message(ai_message_content_to_add)

    # --- Handle time-based queries ---
    time_keywords = ["time", "when", "how long", "until", "left", "remaining", "schedule", "start"]
    event_keywords = {
        "lunch": ["lunch", "food", "meal", "eating"],
        "coffee": ["coffee", "break", "refreshment"],
        "next break": ["next break", "break time", "rest"],
        "keynote": ["keynote", "opening", "main talk"],
        "registration": ["registration", "check-in", "sign in"]
    }

    # --- Priority 1: Direct Event Time Queries (e.g., "lunch?", "coffee time?") ---
    for event_name_key, specific_event_keywords_list in event_keywords.items():
        if query_lower in specific_event_keywords_list: # e.g. if query_lower is "lunch"
            debug_info_accumulator.append(f"[REASONING] Query directly matched event keyword for '{event_name_key}'.")
            try:
                event_time_info = utils.get_time_until_event(event_name_key)
                debug_info_accumulator.append(f"[UTIL_CALL] utils.get_time_until_event('{event_name_key}') -> Result: {event_time_info}")
                
                answer_text = event_time_info
                # Special handling for lunch to provide more context if desired (as was in original)
                if event_name_key == "lunch":
                    agenda = utils.load_agenda()
                    lunch_event_details = None # Renamed from lunch_event to avoid scope confusion
                    for item in agenda.get('schedule', []):
                        if 'event' in item and 'lunch' in item['event'].lower():
                            lunch_event_details = item
                            break
                    if lunch_event_details:
                        answer_text = f"🍽️ {event_time_info}\n\n"
                        if 'time' in lunch_event_details:
                            answer_text += f"Lunch is scheduled for {lunch_event_details['time']}"
                        if lunch_event_details.get('location'): # Check if location exists
                            answer_text += f"\nLocation: {lunch_event_details['location']}"
                
                return {"answer": answer_text, "debug_info": "\n".join(debug_info_accumulator)}
            except Exception as e:
                print(f"[DEBUG] Error getting time for {event_name_key}: {e}")
                debug_info_accumulator.append(f"[ERROR] Error getting time for {event_name_key}: {e}")
                return {
                    "answer": f"I'm having trouble calculating the time for {event_name_key}. Please check the agenda.",
                    "debug_info": "\n".join(debug_info_accumulator)
                }

    # --- Priority 2: General Time-based Queries (e.g., "how long until the keynote?") ---
    is_time_query = any(keyword in query_lower for keyword in time_keywords)
    if is_time_query:
        debug_info_accumulator.append("[REASONING] Query identified as a time-based query (contains general time keyword).")
        
        # Check for specific events within these more general time queries
        for event_name_key, specific_event_keywords_list in event_keywords.items():
            if any(keyword in query_lower for keyword in specific_event_keywords_list):
                debug_info_accumulator.append(f"[REASONING] Identified as '{event_name_key}' time query within general time query.")
                try:
                    event_time_info = utils.get_time_until_event(event_name_key)
                    debug_info_accumulator.append(f"[UTIL_CALL] utils.get_time_until_event('{event_name_key}') -> Result: {event_time_info}")
                    
                    answer_text = event_time_info
                    # Special handling for lunch to provide more context (as was in original)
                    if event_name_key == "lunch":
                        agenda = utils.load_agenda()
                        lunch_event_details = None
                        for item in agenda.get('schedule', []):
                            if 'event' in item and 'lunch' in item['event'].lower():
                                lunch_event_details = item
                                break
                        if lunch_event_details:
                            answer_text = f"🍽️ {event_time_info}\n\n"
                            if 'time' in lunch_event_details:
                                answer_text += f"Lunch is scheduled for {lunch_event_details['time']}"
                            if lunch_event_details.get('location'): # Check if location exists
                                answer_text += f"\nLocation: {lunch_event_details['location']}"

                    return {"answer": answer_text, "debug_info": "\n".join(debug_info_accumulator)}
                except Exception as e:
                    print(f"[DEBUG] Error getting time for {event_name_key}: {e}")
                    debug_info_accumulator.append(f"[ERROR] Error getting time for {event_name_key}: {e}")
                    return {
                        "answer": f"I'm having trouble calculating the time for {event_name_key}. Please check the agenda.",
                        "debug_info": "\n".join(debug_info_accumulator)
                    }
        
        # If no specific event keyword was matched within the general time query, but it IS a time query
        # (e.g., "what time is it?", "what is the schedule like?")
        # This can be handled by a general time/schedule response or fall through. For now, we let it fall through
        # as the agenda response later might be more appropriate for a generic "schedule" query.
        debug_info_accumulator.append("[INFO] General time query did not match a specific event. May fall through or be handled by agenda.")

    # --- Resume-based Q&A ---
    resume_keywords = ["who", "participants", "attendees", "worked on", "similar technical area", 
                      # "relevant to me", "my background", "my resume", # Removed "me/my" for now to separate logic
                      "suggest sessions", "recommend sessions",
                      "resume", "resumes", "skill", "skills", "profile", "profiles", "experience", "qualification", "qualifications",
                      "similar background", "similar experience", "networking", "connect with", "match", "matching", # "like me" removed
                      "technical skills", "education", "work experience", "full-stack", "cloud", "data", "ml", "machine learning",
                      "educational background", "degree", "university", "college", "academic", "which sessions", "relevant sessions",
                      "workshop recommendations", "session suggestions", "what should I attend", "similar technical areas",
                      "find", "search", "looking for", "with experience in", "who knows", "who has experience"]
    
    # --- Specific "For Me" (Chatbot User) Recommendation Flow ---
    # Check if we are waiting for user's background for recommendations
    if st.session_state.get("waiting_for_user_background_for_recommendation", False):
        debug_info_accumulator.append("[REASONING] Was waiting for user background for recommendations.")
        st.session_state.user_provided_background = user_query # Assume current query is the background
        st.session_state.waiting_for_user_background_for_recommendation = False
        debug_info_accumulator.append(f"  User provided background: {st.session_state.user_provided_background}")
        # Now, directly get recommendations using this background
        response_data = get_session_recommendations(
            user_query="session recommendations for user", # Generic query, background is key
            vector_store_instance=vector_store_instance, 
            process_for_all_resumes=False, 
            api_key=api_key, 
            debug_accumulator_parent=debug_info_accumulator,
            user_specific_background=st.session_state.user_provided_background
        )
        return {"answer": response_data["result"], "debug_info": "\n".join(response_data.get("debug_info_list",[]))}

    # Check for queries asking for recommendations for "me" (the chatbot user)
    # More specific than general session recommendation keywords to avoid conflict
    user_specific_recommendation_keywords = [
        "recommend sessions for me", "suggest workshops for me", "what sessions are good for me",
        "sessions for my background", "workshops relevant to my skills", "what should i attend based on my interests",
        "recommend something for me", "can you suggest sessions for me personally"
    ]
    if any(keyword in query_lower for keyword in user_specific_recommendation_keywords):
        debug_info_accumulator.append("[REASONING] Query matched user-specific session recommendation keywords (for chatbot user).")
        if st.session_state.get("user_provided_background"):
            debug_info_accumulator.append(f"  Using previously provided user background: {st.session_state.user_provided_background}")
            response_data = get_session_recommendations(
                user_query, # Pass original query
                vector_store_instance=vector_store_instance, 
                process_for_all_resumes=False, 
                api_key=api_key, 
                debug_accumulator_parent=debug_info_accumulator,
                user_specific_background=st.session_state.user_provided_background
            )
            return {"answer": response_data["result"], "debug_info": "\n".join(response_data.get("debug_info_list",[]))}
        else:
            st.session_state.waiting_for_user_background_for_recommendation = True
            answer_text = "To give you the best workshop recommendations, could you please tell me a bit about your skills, interests, or background? For example: 'I am a Python developer interested in AI' or 'My background is in cloud technologies and data engineering.'"
            debug_info_accumulator.append("  No user background found. Asking user to provide it.")
            return {"answer": answer_text, "debug_info": "\n".join(debug_info_accumulator)}

    # --- General Resume-based Q&A (excluding "for me" which is now handled above) ---
    # Note: `resume_keywords` list was slightly modified earlier
    if vector_store_instance and any(keyword in query_lower for keyword in resume_keywords):
        debug_info_accumulator.append("[REASONING] Query identified as potentially resume-based (general or for specific processed resumes).")
        try:
            # query_lower = user_query.lower().strip() # REMOVED REDUNDANT DEFINITION

            # --- Priority 1: Specific Person Skill Search (e.g., "John Doe skills") ---
            # CORRECTED \\s to \s in regex
            name_extraction_pattern = r"^(?:what are|what is|tell me about|get|find|list|show|extract)?\s*(?:the)?\s*([A-Za-z]+\s+[A-Za-z]+(?:'s)?)\s*(?:technical skills|skills|background|experience|qualifications|resume|profile|information|details)$"
            name_match = re.search(name_extraction_pattern, query_lower)
            if not name_match:
                 # CORRECTED \\s to \s in regex
                 name_match = re.search(r"([A-Za-z]+\s+[A-Za-z]+(?:'s)?)\s*(?:technical skills|skills|background|experience)", query_lower)

            if name_match:
                name = name_match.group(1).replace("'s", "").strip()
                debug_info_accumulator.append(f"[REASONING] Query Tentatively Matched 'Specific Person Skill Search'. Name: {name}")
                excluded_keywords_in_name = ["participant", "attendee", "experience", "skill", "with", "in", "on", "for", "of", "list", "show", "find", "get", "based", "years of"]
                # More robust check to differentiate from general skill search
                is_likely_a_name_for_specific_search = (
                    len(name.split()) >= 2 and 
                    len(name) > 4 and 
                    not any(kw in name.lower() for kw in excluded_keywords_in_name) and
                    name.lower() not in ["technical skills", "educational background", "work experience", "full-stack development", "data analysis", "machine learning", "cloud technologies"] # common skill phrases
                )
                if is_likely_a_name_for_specific_search:
                    debug_info_accumulator.append(f"[REASONING] Confirmed 'Specific Person Skill Search' for Name: {name}")
                    resume_data = get_resume_by_name(vector_store_instance, name, debug_accumulator_parent=debug_info_accumulator)
                    docs, error = resume_data["docs"], resume_data["error"]
                    if error:
                        return {"answer": error, "debug_info": "\\n".join(resume_data.get("debug_info_list",[]))}
                    if not docs:
                        answer_text = f"No resume or technical skills information found for {name}."
                        debug_info_accumulator.append(f"  No documents found by get_resume_by_name for {name}.")
                        return {"answer": answer_text, "debug_info": "\\n".join(resume_data.get("debug_info_list",[]))}
                    skills = extract_technical_skills(docs)
                    debug_info_accumulator.append(f"  Extracted skills for {name}: {skills}")
                    source_doc = docs[0].metadata.get('source') if docs else None
                    answer_text = format_skills_response(name, skills, source_doc)
                    return {"answer": answer_text, "debug_info": "\\n".join(resume_data.get("debug_info_list",[]))}
                else:
                    debug_info_accumulator.append(f"  Extracted term '{name}' from name_match was rejected as a specific person for skill search; might be a general skill phrase.")
            
            # --- New Priority (before General Skill Search): "All Resumes" Query ---
            all_resumes_keywords = ["list all resumes", "show all resumes", "all resumes", "what resumes are processed", "available resumes"]
            if any(keyword in query_lower for keyword in all_resumes_keywords):
                debug_info_accumulator.append("[REASONING] Query matched 'All Resumes' keywords.")
                all_resume_docs_for_sources = vector_store_instance.similarity_search("resume name contact information", k=200) 
                unique_sources = sorted(list(set(
                    doc.metadata.get('source') 
                    for doc in all_resume_docs_for_sources 
                    if doc.metadata.get('source')
                )))
                if not unique_sources:
                    answer_text = "No resumes have been processed or found in the system yet."
                else:
                    formatted_names = [os.path.splitext(s)[0].replace('_', ' ').title() for s in unique_sources]
                    answer_text = "📄 **Available Resumes:**\\n- " + "\\n- ".join(formatted_names)
                return {"answer": answer_text, "debug_info": "\\n".join(debug_info_accumulator)}

            # --- New Priority: "Resume <Name>" Summary Query ---
            # CORRECTED \\s to \s in regex
            resume_summary_match = re.search(r"^(?:summarize resume for|resume summary for|resume for|resume)\s+([A-Za-z]+\s+[A-Za-z]+(?:\s+[A-Za-z]+)?)$", query_lower) # query_lower is already stripped
            if resume_summary_match:
                name_for_summary = resume_summary_match.group(1).strip()
                debug_info_accumulator.append(f"[REASONING] Query matched 'Resume <Name> Summary' for Name: {name_for_summary}")
                
                resume_data = get_resume_by_name(vector_store_instance, name_for_summary, debug_accumulator_parent=debug_info_accumulator)
                docs, error = resume_data["docs"], resume_data["error"]

                if error:
                    return {"answer": error, "debug_info": "\\n".join(resume_data.get("debug_info_list",[]))}
                if not docs:
                    answer_text = f"Could not retrieve content for {name_for_summary}'s resume to summarize."
                    debug_info_accumulator.append(f"  No documents found by get_resume_by_name for {name_for_summary} for summary.")
                    return {"answer": answer_text, "debug_info": "\\n".join(resume_data.get("debug_info_list",[]))}

                resume_content_for_summary = " \\n ".join([doc.page_content for doc in docs])
                
                summary_prompt_template = """Based strictly on the following resume content for {person_name}, provide a concise professional summary (3-5 sentences). 
Highlight key areas such as their primary role/expertise, total years of directly mentioned experience (if explicitly stated), core skills/technologies they have worked with, and perhaps one significant achievement or area of focus if clearly described. 
Do not infer information not present in the text. Do not list skills directly unless they are part of a synthesized statement about their expertise. 
If the provided content is insufficient for a meaningful professional summary (e.g., it's too short, lacks detail, or is not a resume), clearly state that a detailed summary cannot be generated from the provided text instead of attempting to create one.

Resume Content:
{resume_content}

Concise Professional Summary for {person_name}:"""
                prompt = PromptTemplate(template=summary_prompt_template, input_variables=["person_name", "resume_content"])
                chain = LLMChain(llm=llm, prompt=prompt)
                
                try:
                    summary = chain.run(person_name=name_for_summary, resume_content=resume_content_for_summary)
                    answer_text = f"📄 **Resume Summary for {name_for_summary.title()}**:\\n\\n{summary}"
                    debug_info_accumulator.append(f"  LLM summary generated for {name_for_summary}.")
                except Exception as e:
                    debug_info_accumulator.append(f"[ERROR] LLM summary generation error for {name_for_summary}: {e}")
                    answer_text = f"Sorry, I encountered an error while trying to summarize {name_for_summary}'s resume."
                
                return {"answer": answer_text, "debug_info": "\\n".join(debug_info_accumulator)}

            # --- NEW: Role-Based Profile Search ---
            # Regex to capture role name (group 1) and skills list (group 2)
            # This pattern was recently added and seems to use \s correctly.
            role_profile_pattern = r"(?:identify|find|search for|list|get|show|can we get|can you find|could we see)\s*(?:me|us)?\s*(?:participants?|attendees?|candidates?|profiles?|resumes?)\s*(?:whose resumes align with|who align with|that align with|that match|matching|for|as|who are)\s*(?:a|an)?\s*([A-Za-z\s\-]+?)\s*(?:role|profile)?\s*(?:,\s*)?.*?\s*(?:(?:look|looking)\s+for\s+skills?|skills?\s+like|with\s+skills?|requiring\s+skills?|including\s+skills?)\s*([A-Za-z0-9\s,&_()\-./#+]+?)(?:\?|\.|based on|for example|e\.g\.|$)"
            role_match = re.search(role_profile_pattern, query_lower, re.IGNORECASE)

            if role_match:
                role_name = role_match.group(1).strip()
                skills_string = role_match.group(2).strip()
                debug_info_accumulator.append(f"[REASONING] Query matched 'Role-Based Profile Search'. Role: '{role_name}', Skills String: '{skills_string}'")

                raw_skills = [s.strip() for s in skills_string.split(',')]
                parsed_skills = []
                for skill_part in raw_skills:
                    sub_skills = re.split(r'\s+and\s+', skill_part, flags=re.IGNORECASE)
                    for s_skill in sub_skills:
                        s_skill = s_skill.strip().rstrip('.') # also strip trailing dots
                        if s_skill and s_skill.lower() not in ['and', 'etc']:
                            parsed_skills.append(s_skill)
                
                target_skills_for_match = [s.lower() for s in parsed_skills if s] # Lowercase for matching
                
                if not target_skills_for_match:
                    debug_info_accumulator.append("[ERROR] No skills extracted for role-based search from string: '{skills_string}'. Falling through.")
                else:
                    debug_info_accumulator.append(f"  Target Skills for Role '{role_name}': {target_skills_for_match}")

                    all_resume_docs_for_sources = vector_store_instance.similarity_search("resume name contact information", k=200)
                    unique_sources_with_content = {}
                    processed_sources_for_content = set()

                    for doc_source in all_resume_docs_for_sources:
                        source_path = doc_source.metadata.get('source')
                        if source_path and source_path not in processed_sources_for_content:
                            # Retrieve more comprehensive content for this specific resume
                            # Using a generic query for the source to get its varied chunks
                            source_specific_docs = vector_store_instance.similarity_search(
                                f"content of resume {os.path.basename(source_path)}", 
                                k=25, # Increased k for better coverage of the resume
                                filter={"source": source_path}
                            )
                            full_content = " ".join([d.page_content for d in source_specific_docs]).lower()
                            unique_sources_with_content[source_path] = full_content
                            processed_sources_for_content.add(source_path)
                    
                    debug_info_accumulator.append(f"  Found {len(unique_sources_with_content)} unique resumes to scan for role profile.")

                    if not unique_sources_with_content:
                        answer_text = "No resumes found in the vector store to search for role profiles."
                        return {"answer": answer_text, "debug_info": "\\n".join(debug_info_accumulator)}

                    matching_participants_details = []
                    min_skills_to_match = max(1, int(len(target_skills_for_match) * 0.5)) # Match at least 50% of skills, or 1 if only 1 skill
                    debug_info_accumulator.append(f"  Minimum skills to match for '{role_name}' profile: {min_skills_to_match} (out of {len(target_skills_for_match)})")

                    for source_file, content_lower in unique_sources_with_content.items():
                        participant_name_display = os.path.splitext(os.path.basename(source_file))[0].replace('_', ' ').title()
                        current_matched_skills_count = 0
                        found_skills_for_this_participant = []
                        for skill_to_find in target_skills_for_match:
                            # CORRECTED \\b to \b in regex
                            if re.search(r'\b' + re.escape(skill_to_find) + r'\b', content_lower):
                                current_matched_skills_count += 1
                                found_skills_for_this_participant.append(skill_to_find)
                        
                        debug_info_accumulator.append(f"    Participant: {participant_name_display}, Matched Skills Count: {current_matched_skills_count}, Found: {found_skills_for_this_participant}")

                        if current_matched_skills_count >= min_skills_to_match:
                            matching_participants_details.append(
                                f"{participant_name_display} (Matched: {current_matched_skills_count}/{len(target_skills_for_match)} skills - {', '.join(sorted(found_skills_for_this_participant))})"
                            )
                    
                    if matching_participants_details:
                        answer_text = f"🔎 Participants who may align with a **{role_name}** role (requiring skills like: {', '.join(parsed_skills)}):\n\n"
                        # Sort by number of matched skills (descending), then by name (ascending)
                        # To sort, we need to parse the match count from the string, or store it separately
                        # For now, simple alphabetical sort of the detail strings
                        answer_text += "\\n- ".join([""] + sorted(matching_participants_details))
                    else:
                        answer_text = f"I couldn't find participants strongly matching the **{role_name}** profile with skills like {', '.join(parsed_skills)} among the processed resumes."
                    
                    return {"answer": answer_text, "debug_info": "\\n".join(debug_info_accumulator)}

            # --- Priority 2 (now 3): General Skill Search (e.g., "who has Python skill") ---
            skill_search_patterns = [
                # CORRECTED \\s to \s in regex patterns below
                r"(?:find|search for|looking for|who has|who knows|list participants with|participants with experience in|tell me who has)\s+(?:\w+\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years(?:\s+of)?\s+experience\s*(?:in|with|related to)?\s+(.+?)(?:\?|$)",
                r"(?:find|search for|looking for|who has|who knows|list participants with|participants with experience in|tell me who has|show me who has|show who has)\s+(?:experience|skills|knowledge)?\s*(?:in|with|of|related to|on)\s+([A-Za-z0-9 .+#/\-]+?)(?:\s*skills|\s*experience)?(?:\?|$)",
                r"who (?:\s*has|\s*have|\s*knows|\s*is experienced in|\s*is skilled in|\s*worked with|\s*worked on|\s*worked in)\s+([A-Za-z0-9 .+#/\-]+?)(?:\s*skills|\s*experience)?(?:\\?|$)", # Added \s* around verb
                r"([A-Za-z0-9 .+#/\-]+?)\s+(?:skills?|experience|expertise|knowledge)\s+(?:for|of|among|in|possessed by)\s*(?:participants?|attendees?|people|individuals|users|folks|candidates|team members?)(?:\?|$)",
                r"(?:participants?|attendees?|people|individuals|users|folks|candidates|team members?)\s*(?:with|having|possessing)\s+([A-Za-z0-9 .+#/\-]+?)\s*(?:skills?|experience|expertise|knowledge)?(?:\?|$)",
                r"(?:get|show|find|list|can we get|can you find|could we see)\s*(?:me|us)?\s*([A-Za-z0-9 .+#/\-]+?)\s*(?:resumes?|profiles?|cvs?|candidates?|participants?)(?:\?|$)"
            ]
            skill_query_term = None
            full_experience_phrase_for_retriever = None
            
            for pattern_idx, pattern in enumerate(skill_search_patterns):
                match = re.search(pattern, query_lower)
                if match:
                    potential_skill_phrase = match.group(1).strip() if match.groups() and match.group(1) else ""
                    # Avoid treating very short or overly broad phrases extracted by (.+?) as skills if they are likely just filler.
                    # Example: "experience in..." should not make "..." the skill.
                    if not potential_skill_phrase or len(potential_skill_phrase) < 2: continue

                    is_likely_name_fragment_for_skill_search = any(name_part in potential_skill_phrase for name_part in ["sharma", "singh", "kumar", "doe", "patel"]) or \
                                            (len(potential_skill_phrase.split()) > 4 and "years of experience" not in match.group(0).lower())
                    
                    if not potential_skill_phrase and "years of experience" in match.group(0).lower() and pattern_idx == 0: 
                        skill_query_term = "professional experience" 
                        full_experience_phrase_for_retriever = match.group(0) 
                        break
                    elif potential_skill_phrase and not is_likely_name_fragment_for_skill_search:
                        if "years of experience" in match.group(0).lower() and pattern_idx == 0:
                            skill_query_term = potential_skill_phrase 
                            full_experience_phrase_for_retriever = match.group(0) 
                        else:
                            skill_query_term = potential_skill_phrase
                        break
            
            if skill_query_term:
                debug_info_accumulator.append(f"[REASONING] Query matched 'General Skill Search'. Skill Query Term: {skill_query_term}")
                # ... (rest of the existing General Skill Search logic)
                if full_experience_phrase_for_retriever:
                    debug_info_accumulator.append(f"  Full Experience Phrase for Retriever: {full_experience_phrase_for_retriever}")
                target_search_keywords = get_target_keywords_for_query(skill_query_term, SKILL_CATEGORIES)
                debug_info_accumulator.append(f"  Target Search Keywords: {target_search_keywords}")
                search_text_for_retriever = full_experience_phrase_for_retriever if full_experience_phrase_for_retriever else f"technical skills, experience, or expertise in {skill_query_term}"
                debug_info_accumulator.append(f"  Retriever Search Text: {search_text_for_retriever}")
                retriever = vector_store_instance.as_retriever(search_kwargs={"k": 25})
                relevant_docs = retriever.get_relevant_documents(search_text_for_retriever)
                debug_info_accumulator.append(f"  Retrieved {len(relevant_docs)} docs by retriever.")
                if relevant_docs:
                    found_sources_debug = {}
                    for doc_idx, doc in enumerate(relevant_docs):
                        source_file = doc.metadata.get('source')
                        if source_file:
                            content_lower = doc.page_content.lower()
                            matching_keywords_for_doc = [kw for kw in target_search_keywords if kw.lower() in content_lower]
                            if matching_keywords_for_doc:
                                base_name = os.path.basename(source_file)
                                if base_name not in found_sources_debug:
                                    found_sources_debug[base_name] = set()
                                found_sources_debug[base_name].update(matching_keywords_for_doc)
                    found_sources = set(found_sources_debug.keys())
                    if found_sources:
                        formatted_names = sorted([os.path.splitext(s)[0].replace('_', ' ').title() for s in found_sources])
                        answer_text = f"Participants with experience in {skill_query_term}:\\n- " + "\\n- ".join(formatted_names)
                        return {"answer": answer_text, "debug_info": "\\n".join(debug_info_accumulator)}
                    else:
                        # Refined message if docs found but no keyword matches
                        all_resume_sources_list = sorted([os.path.splitext(os.path.basename(doc.metadata.get('source')))[0].replace('_', ' ').title() 
                                                  for doc in vector_store_instance.similarity_search("resume name", k=100) 
                                                  if doc.metadata.get('source')])
                        available_resumes_msg = ""
                        if all_resume_sources_list:
                             available_resumes_msg = "\\n\\nProcessed resumes include: " + ", ".join(list(set(all_resume_sources_list))[:5]) + (", ..." if len(set(all_resume_sources_list)) > 5 else "")
                        answer_text = (f"I found some documents related to '{skill_query_term}' but couldn't definitively confirm who has this specific skill "
                                       f"based on keywords: {', '.join(target_search_keywords[:5])}{(', ...' if len(target_search_keywords) > 5 else '')}.{available_resumes_msg}")
                        return {"answer": answer_text, "debug_info": "\\n".join(debug_info_accumulator)}
                else:
                    answer_text = f"Sorry, I couldn't find any participants with direct experience in {skill_query_term} among the processed resumes."
                    return {"answer": answer_text, "debug_info": "\\n".join(debug_info_accumulator)}

            # --- Priority 3: Session Recommendations ---
            session_recommendation_keywords = [
                "which sessions", "relevant sessions", "suggest sessions", "recommend sessions",
                "what should I attend", "workshop recommendations", "session suggestions", "workshop sessions",
                "which workshops", "what workshops", "what sessions", "relevant workshops",
                # "sessions for me", "workshops for me", "beneficial for me to attend" # These are handled by user-specific logic now
                # The "Based on my background..." for ALL resumes is still triggered by its specific phrase match logic below
            ]
            if any(keyword in query_lower for keyword in session_recommendation_keywords) or \
               query_lower == "Based on my background and skills, which workshop sessions would be most beneficial for me to attend?".lower(): # This specific phrase still triggers all resumes
                debug_info_accumulator.append("[REASONING] Query matched 'session recommendation' keywords (general or for all resumes).")
                
                all_resumes_trigger_phrase = "Based on my background and skills, which workshop sessions would be most beneficial for me to attend?".lower()
                explicit_all_participants_keywords = ["all participants", "each resume", "every resume", "attendee recommendations", "for everyone"]
                
                process_for_all = (query_lower == all_resumes_trigger_phrase or 
                                   any(keyword in query_lower for keyword in explicit_all_participants_keywords))
                
                # Ensure user-specific flow (asking for background) isn't re-triggered if it fell through
                if not st.session_state.get("waiting_for_user_background_for_recommendation"):
                    response_data = get_session_recommendations(
                        user_query, 
                        vector_store_instance, 
                        process_for_all_resumes=process_for_all, 
                        api_key=api_key, 
                        debug_accumulator_parent=debug_info_accumulator
                        # No user_specific_background here, relies on query or all resumes
                    )
                    return {"answer": response_data["result"], "debug_info": "\n".join(response_data.get("debug_info_list",[]))}
                # If waiting_for_user_background_for_recommendation is true, it means the "for me" logic should have caught it earlier.
                # This path (general session rec) should ideally not be hit if we are waiting.

            # --- Priority 4: Similar Background/Networking ---
            similar_background_keywords = [
                # Kept specific phrases, removed overly general ones like "who else" when a skill might be present
                "similar technical area for networking", "similar background for networking", 
                "match for networking", "matching profiles for collaboration",
                "suggest participants with similar technical backgrounds",
                "find people with very similar overall experience",
                "connect with someone like me technically"
            ]
            # Add a check: if the query contains specific skill keywords from SKILL_CATEGORIES, 
            # it's less likely to be a pure "similar background" query unless explicitly stated.
            contains_specific_skill_mention = False
            flat_skill_variations = [var for cat_skills in SKILL_CATEGORIES.values() for skill_vars in cat_skills.values() for var in skill_vars]
            if any(skill_variation in query_lower for skill_variation in flat_skill_variations):
                contains_specific_skill_mention = True

            is_similar_background_query = False
            if any(keyword in query_lower for keyword in similar_background_keywords):
                if "python" in query_lower and "who else worked" in query_lower: # Specific case from user
                    is_similar_background_query = False # Should be handled by general skill search
                    debug_info_accumulator.append("[REASONING] 'who else worked on python' treated as general skill search, not similar background.")
                elif not contains_specific_skill_mention: # If no specific skill, more likely a general similar background query
                    is_similar_background_query = True
                elif "similar" in query_lower or "networking" in query_lower: # If it explicitly asks for similarity/networking despite skill mention
                    is_similar_background_query = True

            if is_similar_background_query:
                debug_info_accumulator.append("[REASONING] Query matched 'similar background/networking' keywords.")
                response_data = get_similar_backgrounds(vector_store_instance, debug_accumulator_parent=debug_info_accumulator)
                return {"answer": response_data["result"], "debug_info": "\\n".join(response_data.get("debug_info_list",[]))}

            # --- Priority 5: General Detail Extraction from All Resumes (e.g., "list all technical skills") ---
            general_query_mappings = {
                "technical_skills": {
                    "keywords": ["technical skills"], 
                    "detail_type": "Technical Skills Summary",
                    "template": "From the provided resume content: {context}, list all explicitly mentioned technical skills, programming languages, frameworks, libraries, tools, and technologies. Present the information clearly: use headings for skill categories (e.g., 'Programming Languages', 'Frameworks', 'Databases') and use bullet points for individual skills. If no skills are found in a category, either omit the category or state 'None found under this category'. If no relevant technical skills are mentioned at all, state 'No specific technical skills were found in the provided content.'. Ensure double newlines between categories for readability."
                },
                "education": {
                    "keywords": ["education", "educational background", "qualification"],
                    "detail_type": "Educational Background",
                    "template": "From the provided resume content: {context}, extract and list all explicitly mentioned educational qualifications, degrees, universities, and graduation years. Format the output clearly with headings for each qualification, and use bullet points for details. Only include information directly related to education. If no educational qualifications are found, state 'No educational background details were found in the provided content.'. Ensure good spacing between entries."
                },
                "work_experience": {
                    "keywords": ["work experience"], 
                    "detail_type": "Work Experience Summary",
                    "template": "From the provided resume content: {context}, summarize explicitly mentioned work experience. For each role, include job title, company, duration, and key responsibilities/achievements. Focus on summarizing actual roles and responsibilities. Format the summary with clear paragraphs or distinct sections for each role and use bullet points for responsibilities/achievements. If no specific work experience is detailed, state 'No work experience details were found in the provided content.'. Ensure double newlines between different roles for clarity."
                },
                "full_stack": {
                    "keywords": ["full-stack development", "full stack developer"],
                    "detail_type": "Full-Stack Development Experience",
                    "template": "From the provided resume content: {context}, identify and list explicitly mentioned technologies, languages, and frameworks that indicate full-stack development experience (covering both frontend and backend). Organize the findings under relevant headings (e.g., 'Frontend Technologies', 'Backend Technologies', 'Databases') and use bullet points. If no such experience is found, state 'No full-stack development experience details were found.'. Ensure clear separation between categories."
                },
                "data_ml": {
                    "keywords": ["data & ml experience", "machine learning experience", "data science experience"],
                    "detail_type": "Data Science & ML Experience",
                    "template": "From the provided resume content: {context}, extract and list explicitly mentioned experience, projects, and skills related to data analysis, machine learning, AI, or data science. Structure the output with clear headings for aspects like 'Projects', 'Skills', 'Tools', using bullet points. If no such experience/skills are found, state 'No data science, ML, or AI experience details were found.'. Ensure good readability."
                },
                "cloud_experience": {
                    "keywords": ["cloud technologies experience", "cloud experience"],
                    "detail_type": "Cloud Technologies Experience",
                    "template": "From the provided resume content: {context}, extract and list explicitly mentioned experience, skills, and projects related to cloud platforms (like AWS, Azure, GCP), containerization (Docker, Kubernetes), and DevOps. Categorize information (e.g., 'Cloud Platforms', 'Containerization', 'DevOps Practices') using bullet points. If no such experience is found, state 'No cloud technologies or DevOps experience details were found.'. Ensure output is well-spaced."
                }
            }
            matched_query_config = None
            # Check if the query primarily asks for a "list of all X" or "summary of X" rather than "who has X"
            is_general_extraction_intent = any(phrase in query_lower for phrase in ["list all", "summary of all", "show all", "what are the technical skills", "what is the education background"]) and not any(phrase in query_lower for phrase in ["who has", "participants with", "find people with"])

            if is_general_extraction_intent: # Only enter this if the intent is clearly general extraction
                for config_name, config_values in general_query_mappings.items():
                    if any(keyword in query_lower for keyword in config_values["keywords"]):
                        matched_query_config = config_values
                        debug_info_accumulator.append(f"[REASONING] Matched general_query_mapping: {config_name} (Intent: General Extraction)")
                        break
            
            if matched_query_config:
                response_data = get_details_for_all_resumes(
                    vector_store_instance=vector_store_instance,
                    detail_type_for_display=matched_query_config["detail_type"],
                    extraction_query_template=matched_query_config["template"],
                    llm=llm,
                    debug_accumulator_parent=debug_info_accumulator
                )
                return {"answer": response_data["result"], "debug_info": "\\n".join(response_data.get("debug_info_list",[]))}

            # --- Priority 6: Fallback General ConversationalRetrievalChain ---
            debug_info_accumulator.append("[REASONING] Fallback: No specific resume query type matched. Using general ConversationalRetrievalChain.")
            # ... (rest of the existing Fallback logic) ...
            qa_chain_prompt_template_str = "System: You are a helpful AI assistant for an event. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Context: {context} Question: {question} Helpful Answer:"
            debug_info_accumulator.append(f"  ConversationalRetrievalChain Prompt (simplified conceptual structure): {qa_chain_prompt_template_str}")
            
            retriever_instance = vector_store_instance.as_retriever(search_kwargs={"k": 5})
            retrieved_fallback_docs = retriever_instance.get_relevant_documents(user_query)
            fallback_doc_summary = [f"    Doc {i}: source='{d.metadata.get('source', 'N/A')}', content='{d.page_content[:70].replace('\n', ' ')}...'" for i, d in enumerate(retrieved_fallback_docs)]
            debug_info_accumulator.append(f"  Retriever context for Fallback (k=5 for '{user_query}'):\\n" + "\\n".join(fallback_doc_summary))

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever_instance, 
                memory=st.session_state.chat_memory_main,
                return_source_documents=True 
            )
            result = qa_chain({"question": user_query}) 
            answer = result["answer"]
            debug_info_accumulator.append(f"  ConversationalRetrievalChain Result (Answer): {answer}")

            if result.get("source_documents") and "resume:" not in answer.lower() and "source:" not in answer.lower():
                sources = set(os.path.basename(doc.metadata.get('source')) for doc in result["source_documents"] if doc.metadata.get('source'))
                if sources:
                    answer += "\\n\\nSource Resumes:\\n- " + "\\n- ".join(sorted(list(sources)))
                    debug_info_accumulator.append(f"  Appended source documents: {sources}")
            return {"answer": answer, "debug_info": "\\n".join(debug_info_accumulator)}

        except Exception as e:
            print(f"[DEBUG RESUME Q&A ERROR] An exception occurred: {e}")
            debug_info_accumulator.append(f"[ERROR] Resume Q&A Error: {e}")
            suggestions = [
                "Try rephrasing your resume-related question.",
                "Ask for a list of all available resumes.",
                "Try a simpler skill query, like 'Who knows Python?'.",
                "Ensure resumes have been processed recently."
            ]
            answer = (
                "I encountered an error while searching through the resumes. Please try again or rephrase your query. You could also try:\\n"
                + "\\n".join([f"- {s}" for s in suggestions])
            )
            return {
                "answer": answer,
                "debug_info": "\\n".join(debug_info_accumulator)
            }

    # 1. What's the meeting agenda?
    if any(keyword in user_query.lower() for keyword in ["agenda", "schedule", "plan"]):
        debug_info_accumulator.append("[REASONING] Query matched 'agenda/schedule' keywords.")
        agenda_str = utils.get_agenda_string()
        debug_info_accumulator.append(f"[UTIL_CALL] utils.get_agenda_string() -> Result length: {len(agenda_str)}")
        return {"answer": agenda_str, "debug_info": "\n".join(debug_info_accumulator)}

    # 4. Where's the washroom? (simple location guidance)
    if any(keyword in user_query.lower() for keyword in ["washroom", "restroom", "toilet", "bathroom"]):
        debug_info_accumulator.append("[REASONING] Query matched 'washroom' keywords.")
        location_info = utils.get_location_info("washroom")
        debug_info_accumulator.append(f"[UTIL_CALL] utils.get_location_info('washroom') -> Result: {location_info}")
        return {"answer": location_info, "debug_info": "\n".join(debug_info_accumulator)}

    if "where is" in user_query.lower() or "find" in user_query.lower() or "location of" in user_query.lower():
        debug_info_accumulator.append("[REASONING] Query matched 'where is/find/location of' pattern.")
        query_parts = user_query.lower().split()
        potential_place = " ".join(query_parts[query_parts.index("is") + 1:]) if "is" in query_parts else \
                          " ".join(query_parts[query_parts.index("find") + 1:]) if "find" in query_parts else \
                          " ".join(query_parts[query_parts.index("of") + 1:]) if "of" in query_parts else user_query
        potential_place = potential_place.replace("?","").strip()
        debug_info_accumulator.append(f"  Potential place identified: {potential_place}")
        location_info = utils.get_location_info(potential_place)
        debug_info_accumulator.append(f"[UTIL_CALL] utils.get_location_info('{potential_place}') -> Result: {location_info}")
        return {"answer": location_info, "debug_info": "\n".join(debug_info_accumulator)}

    # --- General Q&A and Conversational Feedback (Question 6) ---
    try:
        debug_info_accumulator.append("[REASONING] Entering General Q&A / Feedback block.")
        # Check for feedback-related queries
        feedback_keywords = ["feedback", "suggest", "improve", "opinion", "thoughts", "comment"]
        is_feedback_query = any(keyword in user_query.lower() for keyword in feedback_keywords)
        
        if is_feedback_query or st.session_state.feedback_mode:
            debug_info_accumulator.append(f"[REASONING] Feedback mode active or query matched feedback keywords. Current mode: {st.session_state.feedback_mode}")
            # Enter feedback mode if not already in it
            if not st.session_state.feedback_mode:
                st.session_state.feedback_mode = True
                answer_text = ("I'd be happy to collect your feedback about the event! 🎯\n\n"
                       "Please share your thoughts on:\n"
                       "• The sessions you attended\n"
                       "• The speakers and presentations\n"
                       "• The venue and facilities\n"
                       "• What you liked most\n"
                       "• What could be improved\n\n"
                       "Feel free to be as specific as you'd like!")
                debug_info_accumulator.append("  Entered feedback mode. Initial prompt sent.")
                return {"answer": answer_text, "debug_info": "\n".join(debug_info_accumulator)}

            # Check for exit conditions
            exit_keywords = ["thank", "thanks", "that's all", "stop", "end", "finish", "done", "exit", "bye"]
            if any(keyword in user_query.lower() for keyword in exit_keywords) and len(user_query.split()) < 5:
                st.session_state.feedback_mode = False
                answer_text = ("Thank you for your valuable feedback! 🙏\n"
                       "Your input helps us improve future events.\n"
                       "Is there anything else I can help you with?")
                debug_info_accumulator.append("  Exited feedback mode due to exit keyword.")
                return {"answer": answer_text, "debug_info": "\n".join(debug_info_accumulator)}

            # Process feedback while in feedback mode
            feedback_prompt_template_str = """You are collecting feedback about an event. Respond naturally and ask relevant follow-up questions.
                Current feedback: {feedback}
                
                Guidelines:
                1. Acknowledge specific points mentioned
                2. Ask for clarification on vague points
                3. Encourage detailed examples
                4. Show genuine interest in their experience
                5. If they mention any issues, ask for suggestions on how to improve
                
                Respond to their feedback and ask an appropriate follow-up question:"""
            
            feedback_prompt = PromptTemplate(
                template=feedback_prompt_template_str,
                input_variables=["feedback"]
            )
            debug_info_accumulator.append(f"  Feedback LLMChain Prompt Template: {feedback_prompt_template_str}")
            debug_info_accumulator.append(f"  Input to feedback chain (user_query): {user_query}")

            chain = LLMChain(llm=llm, prompt=feedback_prompt)
            response = chain.run(feedback=user_query)
            debug_info_accumulator.append(f"  Feedback LLMChain Response: {response}")
            return {"answer": response, "debug_info": "\n".join(debug_info_accumulator)}

        # If not in feedback mode, proceed with general Q&A
        debug_info_accumulator.append("[REASONING] Standard General Q&A path.")
        general_prompt_template_obj = PromptTemplate(
            template=GENERAL_SYSTEM_PROMPT,
            input_variables=["current_date"]
        )
        system_prompt_for_general = SystemMessagePromptTemplate(prompt=general_prompt_template_obj)

        # Explicitly define PromptTemplate for FEEDBACK_SYSTEM_PROMPT
        feedback_prompt_template_obj = PromptTemplate(
            template=FEEDBACK_SYSTEM_PROMPT,
            input_variables=[] # No input variables for the feedback system prompt itself
        )
        system_prompt_for_feedback = SystemMessagePromptTemplate(prompt=feedback_prompt_template_obj)

        system_prompt_template_to_use = system_prompt_for_general # Default
        current_date_str = utils.datetime.date.today().strftime("%Y-%m-%d")

        if "collect feedback" in user_query.lower() or st.session_state.get("feedback_mode", False):
            st.session_state.feedback_mode = True # Enter feedback mode
            system_prompt_template_to_use = system_prompt_for_feedback
            if "thank you" in user_query.lower() or "that's all" in user_query.lower() or ("stop" in user_query.lower() and len(user_query)<20):
                st.session_state.feedback_mode = False # Exit feedback mode
                return "Thank you for your feedback! I hope you enjoy the rest of the event."
        
        # Create the chat prompt with appropriate messages
        messages = []
        if system_prompt_template_to_use == system_prompt_for_general:
            # For general prompts, include the date
            messages.append(SystemMessagePromptTemplate(prompt=PromptTemplate(
                template=GENERAL_SYSTEM_PROMPT,
                input_variables=["current_date"]
            )))
        else:
            # For feedback prompts, no variables needed
            messages.append(SystemMessagePromptTemplate(prompt=PromptTemplate(
                template=FEEDBACK_SYSTEM_PROMPT,
                input_variables=[]
            )))
        
        # Add the human message template
        messages.append(HumanMessagePromptTemplate.from_template("{question}"))
        
        # Create the chat prompt
        chat_prompt = ChatPromptTemplate.from_messages(messages)
        
        # Initialize the chain
        chain = LLMChain(llm=llm, prompt=chat_prompt, memory=st.session_state.chat_memory_main)
        
        debug_info_accumulator.append(f"  General Q&A LLMChain Prompt (System part): {system_prompt_template_to_use.prompt.template}")
        debug_info_accumulator.append(f"  Input to general Q&A chain (user_query): {user_query}")
        if system_prompt_template_to_use == system_prompt_for_general:
            debug_info_accumulator.append(f"  current_date for general Q&A: {current_date_str}")

        # Prepare the input dictionary based on the prompt type
        if system_prompt_template_to_use == system_prompt_for_general:
            response = chain.run({
                "question": user_query,
                "current_date": current_date_str
                # chat_history is implicitly handled by the memory
            })
        else: # This is the feedback path
            # Feedback prompt only expects "feedback" which maps to "question" here
            response = chain.run(user_query) # or chain.run({"feedback": user_query}) if prompt variable is "feedback"

        debug_info_accumulator.append(f"  General Q&A LLMChain Response: {response}")
        return {"answer": response, "debug_info": "\n".join(debug_info_accumulator)}
    except Exception as e:
        st.error(f"Error during general LLM call: {e}")
        debug_info_accumulator.append(f"[ERROR] General LLM call error: {e}")
        
        suggestions = [ # General suggestions
            "Try rephrasing your question.",
            "Ask about the event agenda.",
            "Ask about session locations.",
            "If you're asking about resumes, ensure they are processed."
        ]
        answer = (
            "I encountered an issue trying to respond. Please try rephrasing your question, or you could try one of these options:\n" 
            + "\n".join([f"- {s}" for s in suggestions])
        )
        return {
            "answer": answer,
            "debug_info": "\n".join(debug_info_accumulator)
        }

def get_session_recommendations(user_query, vector_store_instance, process_for_all_resumes=False, api_key=None, debug_accumulator_parent=None, user_specific_background=None): # Added user_specific_background
    """
    Get personalized session recommendations.
    If process_for_all_resumes is True, it iterates through all resumes.
    If user_specific_background is provided, it uses that.
    Otherwise, it provides recommendations based on the user_query or a general context.
    """
    local_debug_info = debug_accumulator_parent if debug_accumulator_parent is not None else []
    local_debug_info.append(f"[FUNC_CALL] get_session_recommendations (process_for_all={process_for_all_resumes}, user_specific_bg_provided={bool(user_specific_background)})")

    # If a specific background is provided by the user, prioritize it.
    if user_specific_background:
        local_debug_info.append(f"[INFO] Using user-specific background for recommendations: {user_specific_background[:100]}...")
        print(f"[DEBUG] Using user-specific background for recommendations: {user_specific_background[:100]}...")
        recommendations = utils.get_workshop_recommendations(user_specific_background)
        local_debug_info.append(f"  utils.get_workshop_recommendations (with user_specific_background) returned {len(recommendations)} recommendations.")
        
        if not recommendations:
            answer = ("I couldn't find any workshop sessions to recommend based on the background you provided: "
                       f"'{user_specific_background[:100]}{'...' if len(user_specific_background)>100 else ''}'. "
                       "Please ensure either workshops.pdf is available or workshops are listed in the agenda. "
                       "You can also try rephrasing your background/interests.")
            local_debug_info.append("[INFO] No recommendations found for user_specific_background.")
            return {"result": answer, "debug_info_list": local_debug_info}

        response = "🎯 **Based on the background you provided, here are some workshop recommendations for you:**\n\n"
        # (Formatting logic remains the same as below, so can be consolidated or duplicated)
        for i, rec in enumerate(recommendations, 1):
            response += f"**{i}. {rec.get('title', 'N/A')}**\n"
            if rec.get('time'):
                response += f"🕒 Time: {rec['time']}\n"
            desc = rec.get('description', '')
            prereq = rec.get('prerequisites', '')
            if desc:
                 response += f"📝 Desc: {desc[:150] + '...' if len(desc) > 150 else desc}\n"
            if prereq:
                 response += f"⚠️ Needs: {prereq[:100] + '...' if len(prereq) > 100 else prereq}\n"
            if rec.get('relevance_reasons'):
                response += "*Possible Relevance:*\n"
                for reason in rec['relevance_reasons'][:2]: # Show top 2 reasons
                    response += f"• {reason[:100] + '...' if len(reason) > 100 else reason}\n"
        response += "\nConsider also checking the full agenda or workshop list for more options!"
        return {"result": response, "debug_info_list": local_debug_info}

    # Original logic if not user_specific_background
    if not vector_store_instance and not process_for_all_resumes: # Allow process_for_all even if VS is none, it will be caught
        answer = "I need access to resume information to provide personalized session recommendations. Please process resumes first, or tell me your background."
        local_debug_info.append("[ERROR] Vector store not available for non-user-specific session recommendations.")
        return {"result": answer, "debug_info_list": local_debug_info}
    
    if process_for_all_resumes:
        if not vector_store_instance : # Double check for this specific path
            answer = "I need access to resume information to provide session recommendations for all participants. Please process resumes first."
            local_debug_info.append("[ERROR] Vector store not available for process_for_all_resumes.")
            return {"result": answer, "debug_info_list": local_debug_info}

        local_debug_info.append("[INFO] Generating session recommendations for ALL resumes.")
        print("[DEBUG] Generating session recommendations for ALL resumes.")
        all_resume_docs_for_sources = vector_store_instance.similarity_search("resume name contact information", k=200) 
        unique_sources = sorted(list(set(
            doc.metadata.get('source') 
            for doc in all_resume_docs_for_sources 
            if doc.metadata.get('source')
        )))

        if not unique_sources:
            answer = "No resumes found in the vector store to generate recommendations for."
            local_debug_info.append("[INFO] No unique sources found for all-resume recommendations.")
            return {"result": answer, "debug_info_list": local_debug_info}

        all_resumes_trigger_phrase_check = "Based on my background and skills, which workshop sessions would be most beneficial for me to attend?".lower()
        if user_query.lower() == all_resumes_trigger_phrase_check: # Use user_query passed to function
            final_response = "🎯 You asked for sessions beneficial based on 'your background'. Here's a breakdown of recommendations for each processed resume:\n\n"
        else:
            final_response = "🎯 **Workshop Recommendations for All Processed Resumes**\n\n"
        
        any_recommendations_found_overall = False

        for source_file_name in unique_sources:
            participant_name = os.path.splitext(source_file_name)[0].replace('_', ' ').title()
            final_response += f"--- **Recommendations for {participant_name} ({source_file_name})** ---\n"
            local_debug_info.append(f"  Processing recommendations for: {participant_name} ({source_file_name})")
            
            bg_search_query = "technical skills experience background expertise projects programming languages frameworks tools technologies education summary"
            source_filter = {"source": source_file_name}
            docs = vector_store_instance.similarity_search(
                bg_search_query, 
                k=10, 
                filter=source_filter
            )
            local_debug_info.append(f"    Retrieved {len(docs)} docs for background using query: '{bg_search_query}'")

            if not docs:
                background_info = "event attendee looking for general workshop recommendations" 
                local_debug_info.append(f"    No specific background docs found for {participant_name}. Using default background: '{background_info}'")
            else:
                background_info = " ".join([doc.page_content for doc in docs])
            
            local_debug_info.append(f"    Background for {participant_name} (first 100 chars): {background_info[:100].replace('\n',' ')}...")
            print(f"[DEBUG] Background for {participant_name} (first 200 chars): {background_info[:200]}")
            
            recommendations = utils.get_workshop_recommendations(background_info)
            
            if not recommendations:
                final_response += "No specific workshop sessions found to recommend based on this resume's content.\n\n"
            else:
                any_recommendations_found_overall = True
                for i, rec in enumerate(recommendations, 1):
                    final_response += f"**{i}. {rec.get('title', 'N/A')}**\n"
                    if rec.get('time'):
                        final_response += f"🕒 Time: {rec['time']}\n"
                    desc = rec.get('description', '')
                    prereq = rec.get('prerequisites', '')
                    if desc:
                         final_response += f"📝 Desc: {desc[:150] + '...' if len(desc) > 150 else desc}\n"
                    if prereq:
                         final_response += f"⚠️ Needs: {prereq[:100] + '...' if len(prereq) > 100 else prereq}\n"

                    if rec.get('relevance_reasons'):
                        final_response += "*Possible Relevance:*\n"
                        for reason in rec['relevance_reasons'][:2]:
                            final_response += f"• {reason[:100] + '...' if len(reason) > 100 else reason}\n"
                    final_response += "\n"
                final_response += "\n"
        
        if not any_recommendations_found_overall and unique_sources:
             final_response = ("I processed all resumes but couldn't find specific workshop recommendations. "
                     "Ensure workshops.pdf or agenda workshops are configured, and resumes contain relevant skills.")
             local_debug_info.append("[INFO] No recommendations found overall for any resume.")
        return {"result": final_response, "debug_info_list": local_debug_info}

    else: 
        local_debug_info.append(f"[INFO] Session recommendation for single user context with query: {user_query}")
        print(f"[DEBUG] Session recommendation for single user context with query: {user_query}")
        extracted_skills_for_background = set()
        query_lower = user_query.lower().strip() # DEFINED query_lower HERE

        for category, skills_in_category in SKILL_CATEGORIES.items():
            for skill_name, variations in skills_in_category.items():
                if skill_name.lower() in query_lower or any(v.lower() in query_lower for v in variations):
                    extracted_skills_for_background.add(skill_name)
        
        if extracted_skills_for_background:
            background_info = " ".join(list(extracted_skills_for_background))
            local_debug_info.append(f"  Using skill-based background from query: {background_info}")
            print(f"[DEBUG] Using skill-based background from query: {background_info}")
        else:
            local_debug_info.append("[INFO] No specific skills in query for 'my background', using general context search.")
            print("[DEBUG] No specific skills in query for 'my background', using general context search.")
            single_user_bg_query = "technical skills programming languages software development interests expertise"
            docs = vector_store_instance.similarity_search(
                single_user_bg_query, 
                k=3 
            )
            local_debug_info.append(f"  Retrieved {len(docs)} docs for general background using query: '{single_user_bg_query}'")
            background_info = " ".join([doc.page_content for doc in docs])
            if not background_info.strip(): 
                background_info = "general technology interest" 
                local_debug_info.append("    Background empty, defaulting to 'general technology interest'.")
        
        local_debug_info.append(f"  Using background for single user (first 100 chars): {background_info[:100].replace('\n',' ')}...")
        print(f"[DEBUG] Using background for single user (first 200 chars): {background_info[:200]}...")
        recommendations = utils.get_workshop_recommendations(background_info)
        local_debug_info.append(f"  utils.get_workshop_recommendations returned {len(recommendations)} recommendations.")
        
        if not recommendations:
            answer = ("I couldn't find any workshop sessions to recommend based on the provided background or query. "
                   "Please ensure either workshops.pdf is available or workshops are listed in the agenda. "
                   "You can also try being more specific about your interests.")
            local_debug_info.append("[INFO] No recommendations found for single user context.")
            return {"result": answer, "debug_info_list": local_debug_info}
        
        response = "🎯 **Based on your query/assumed background, here are some workshop recommendations for you:**\n\n"
        for i, rec in enumerate(recommendations, 1):
            response += f"**{i}. {rec.get('title', 'N/A')}**\n"
            if rec.get('time'):
                response += f"🕒 Time: {rec['time']}\n"
            desc = rec.get('description', '')
            prereq = rec.get('prerequisites', '')
            if desc:
                 response += f"📝 Desc: {desc[:150] + '...' if len(desc) > 150 else desc}\n"
            if prereq:
                 response += f"⚠️ Needs: {prereq[:100] + '...' if len(prereq) > 100 else prereq}\n"

            if rec.get('relevance_reasons'):
                response += "*Possible Relevance:*\n"
                for reason in rec['relevance_reasons'][:2]:
                    response += f"• {reason[:100] + '...' if len(reason) > 100 else reason}\n"
            response += "\n"
        
        response += "\nConsider also checking the full agenda or workshop list for more options!"
        return {"result": response, "debug_info_list": local_debug_info}

def get_similar_backgrounds(vector_store_instance, debug_accumulator_parent=None):
    """Analyze all resumes to find and group participants with similar technical backgrounds."""
    local_debug_info = debug_accumulator_parent if debug_accumulator_parent is not None else []
    local_debug_info.append("[FUNC_CALL] get_similar_backgrounds")
    try:
        # First, get a list of all unique resume sources
        sim_bg_all_docs_query = "name contact information resume profile"
        all_docs = vector_store_instance.similarity_search(
            sim_bg_all_docs_query,
            k=100  # Increased to ensure we get all resumes
        )
        local_debug_info.append(f"  Retrieved {len(all_docs)} docs for all sources using query: '{sim_bg_all_docs_query}' (k=100)")
        
        # Get unique sources
        unique_sources = {doc.metadata.get('source', 'Unknown') for doc in all_docs}
        local_debug_info.append(f"  Unique sources found: {len(unique_sources)}")
        
        if len(unique_sources) <= 1:
            answer = "I need at least two resumes to find similar technical backgrounds. Please ensure multiple resumes are processed."
            local_debug_info.append("  Not enough unique sources to find similarities.")
            return {"result": answer, "debug_info_list": local_debug_info}
        
        # Now get content for each resume separately
        resume_contents = {}
        sim_bg_content_query = "technical skills experience background expertise programming languages frameworks tools projects"
        local_debug_info.append(f"  Fetching content for each source using query: '{sim_bg_content_query}' (k=10)")
        for source in unique_sources:
            # Search specifically within this resume's content
            source_filter = {"source": source}
            source_docs = vector_store_instance.similarity_search(
                sim_bg_content_query,
                k=10,
                filter=source_filter
            )
            local_debug_info.append(f"    Source '{source}': Retrieved {len(source_docs)} docs.")
            # Combine all content for this resume
            resume_contents[source] = " ".join([doc.page_content for doc in source_docs])
        
        # Analyze and group similar backgrounds
        background_groups = {
            "Programming Languages": {},
            "Web Technologies": {},
            "Cloud & DevOps": {},
            "Data Science & ML": {},
            "Domain Expertise": {}
        }
        
        # Process each resume
        for source, content in resume_contents.items():
            content = content.lower()
            
            # Check for each category and its keywords
            for category, keyword_groups in SKILL_CATEGORIES.items():
                for main_keyword, variations in keyword_groups.items():
                    if any(variation in content for variation in variations):
                        if main_keyword not in background_groups[category]:
                            background_groups[category][main_keyword] = []
                        if source not in background_groups[category][main_keyword]:
                            background_groups[category][main_keyword].append(source)
        
        # Format the response
        response = "👥 **Participants with Similar Technical Backgrounds**\\n\\n"
        
        for category, skills in background_groups.items():
            if any(skills.values()):  # Only show categories that have matches
                response += f"**{category}**\\n"
                for skill, sources in skills.items():
                    if len(sources) > 1:  # Only show skills shared by at least 2 people
                        # Remove .pdf extension and format names
                        formatted_sources = [os.path.splitext(src)[0] for src in sources]
                        response += f"- {skill.title()}: {', '.join(formatted_sources)}\n"
                response += "\\n"
        
        if response == "👥 **Participants with Similar Technical Backgrounds**\\n\\n":
            answer = "I couldn't find any shared technical backgrounds between the resumes. Try processing more resumes or check if the resumes contain detailed technical information."
            local_debug_info.append("  No shared technical backgrounds found after analysis.")
            return {"result": answer, "debug_info_list": local_debug_info}
        
        # Add source attribution
        response += "\\nAnalysis based on resumes from:\\n- " + "\\n- ".join(sorted(list(unique_sources)))
        local_debug_info.append("  Successfully generated similar backgrounds response.")
        return {"result": response, "debug_info_list": local_debug_info}
        
    except Exception as e:
        print(f"[DEBUG] Error in get_similar_backgrounds: {e}")
        local_debug_info.append(f"[ERROR] in get_similar_backgrounds: {e}")
        answer = f"I encountered an error while analyzing technical backgrounds: {str(e)}"
        return {"result": answer, "debug_info_list": local_debug_info}

def normalize_name(name):
    """Normalize a name by removing spaces and converting to lowercase."""
    return ''.join(name.lower().split())

def get_resume_by_name(vector_store_instance, name, debug_accumulator_parent=None):
    """Search for a specific person's resume."""
    local_debug_info = debug_accumulator_parent if debug_accumulator_parent is not None else []
    local_debug_info.append(f"[FUNC_CALL] get_resume_by_name (Name: '{name}')")
    try:
        print(f"[DEBUG] Searching for resume with name: {name}")
        
        # First, let's check what resumes are actually in the directory
        resumes_dir = resume_processor.RESUMES_DIR_NAME
        if not os.path.exists(resumes_dir):
            return None, f"Resumes directory '{resumes_dir}' not found."
            
        physical_resumes = [f for f in os.listdir(resumes_dir) 
                          if f.lower().endswith(('.pdf', '.docx'))]
        print(f"[DEBUG] Physical resumes in directory: {physical_resumes}")
        local_debug_info.append(f"  Physical resumes in '{resumes_dir}': {physical_resumes}")
        
        # Normalize the search name
        normalized_search_name = normalize_name(name)
        local_debug_info.append(f"  Normalized search name: {normalized_search_name}")
        print(f"[DEBUG] Normalized search name: {normalized_search_name}")
        
        # First try to match against physical files
        matching_files = []
        for resume_file in physical_resumes:
            base_name = os.path.splitext(resume_file)[0]
            normalized_base = normalize_name(base_name)
            # Check for partial matches
            if (normalized_search_name in normalized_base or 
                normalized_base in normalized_search_name or
                any(part in normalized_base for part in normalized_search_name.split())):
                matching_files.append(resume_file)
                print(f"[DEBUG] Found matching file: {resume_file} (normalized: {normalized_base})")
        local_debug_info.append(f"  Matching physical files: {matching_files}")
        
        # Get all documents from vector store with multiple search approaches
        all_docs_from_vs = [] # Renamed to avoid confusion
        search_queries_for_name = [
            f"name {name}",  # Try exact name
            *[f"name {part}" for part in name.split()],  # Try each part of the name
            "full name contact information"  # Backup to get all names
        ]
        
        local_debug_info.append(f"  Vector store search queries for name '{name}': {search_queries_for_name}")
        for query in search_queries_for_name:
            docs = vector_store_instance.similarity_search(
                query,
                k=100  # Increased to ensure we get all resumes
            )
            all_docs_from_vs.extend(docs) # Use the renamed list
            local_debug_info.append(f"    Query '{query}' (k=100) -> Got {len(docs)} docs.")
            
        # Remove duplicates while preserving order
        seen_sources = set() # Renamed for clarity
        unique_all_docs_from_vs = [doc for doc in all_docs_from_vs # Use the renamed list
                   if not (doc.metadata.get('source') in seen_sources or seen_sources.add(doc.metadata.get('source')))]
        local_debug_info.append(f"  Total unique docs from vector store after de-duplication: {len(unique_all_docs_from_vs)}")
        
        # Get all unique sources
        all_sources_in_vs = {doc.metadata.get('source', 'Unknown') for doc in unique_all_docs_from_vs} # Use the renamed list
        print(f"[DEBUG] Sources in vector store: {all_sources_in_vs}")
        local_debug_info.append(f"  Unique sources in vector store (from retrieved docs): {all_sources_in_vs}")
        
        # Try to find matches in vector store
        matching_sources_in_vs = set() # Renamed for clarity
        for doc in unique_all_docs_from_vs: # Use the renamed list
            content = doc.page_content.lower()
            source = doc.metadata.get('source', '')
            normalized_content = normalize_name(content)
            
            # Check for partial matches in content
            if (normalized_search_name in normalized_content or
                any(part in normalized_content for part in normalized_search_name.split())):
                matching_sources_in_vs.add(source)
                print(f"[DEBUG] Found match in content of source: {source}")
                continue
                
            # Check filename matches if no content match
            base_name = os.path.splitext(source)[0]
            normalized_base = normalize_name(base_name)
            if (normalized_search_name in normalized_base or
                normalized_base in normalized_search_name or
                any(part in normalized_base for part in normalized_search_name.split())):
                matching_sources_in_vs.add(source)
                print(f"[DEBUG] Found match in filename: {source}")
        
        if not matching_sources_in_vs:
            # If we found physical files but no vector store matches
            if matching_files:
                err_msg = (
                    f"Found resume file(s) ({', '.join(matching_files)}) that might match '{name}', but they may need to be processed or re-processed. "
                    f"Please click 'Process Resumes' in the sidebar. You can also try these prompts:\n"
                    f"- List all available resumes.\n"
                    f"- What are the skills of [another name]?"
                )
                local_debug_info.append("  No VS match, but physical files found. Advised reprocessing and provided suggestions.")
                return {"docs": None, "error": err_msg, "debug_info_list": local_debug_info}
            
            # Try to suggest similar names from available resumes
            available_names_from_physical = [] # Renamed for clarity
            for resume in physical_resumes:
                base_name = os.path.splitext(resume)[0].replace('_', ' ').title()
                available_names_from_physical.append(base_name)
            
            suggestion_msg_parts = []
            if available_names_from_physical:
                suggestion_msg_parts.append(f"Available resumes you can ask about:\n- " + "\n- ".join(available_names_from_physical))
            
            suggestion_msg_parts.append("You can also try:\n- Double-check the spelling of the name.")
            suggestion_msg_parts.append("- Ask for a list of all participants.")
            suggestion_msg_parts.append("- Search for skills instead of a name, e.g., 'Who has Java skills?'.")

            err_msg = f"Could not find a resume for '{name}'.\n\n" + "\n\n".join(suggestion_msg_parts)
            local_debug_info.append(f"  No VS match, no physical file match (or physical not processed). Suggested names/prompts: {available_names_from_physical if available_names_from_physical else 'None'}")
            return {"docs": None, "error": err_msg, "debug_info_list": local_debug_info}
        
        # Get detailed information from the matching resume(s)
        all_content_docs = [] # Renamed
        skill_search_queries = [ # Renamed
            "technical skills experience programming languages frameworks tools technologies",
            "software development experience projects technologies used",
            "work experience technical responsibilities technologies",
            "education certifications technical qualifications",
            "skills expertise proficiency competencies",
            "programming languages",
            "frameworks libraries",
            "tools platforms",
            "technical projects",
            "development experience"
        ]
        
        local_debug_info.append(f"  Fetching detailed content for matched sources ({matching_sources_in_vs}) using queries: {skill_search_queries}")
        
        for source_vs in matching_sources_in_vs: # Renamed loop var
            source_filter = {"source": source_vs}
            for query_skill in skill_search_queries: # Renamed loop var
                skill_docs_retrieved = vector_store_instance.similarity_search( # Renamed
                    query_skill,
                    k=10,  # Increased from 5 to get more comprehensive results
                    filter=source_filter
                )
                # Only add unique documents
                for doc_retrieved in skill_docs_retrieved: # Renamed loop var
                    if doc_retrieved not in all_content_docs:
                        all_content_docs.append(doc_retrieved)
                print(f"[DEBUG] Retrieved {len(skill_docs_retrieved)} documents for query '{query_skill}' from {source_vs}")
                local_debug_info.append(f"    Source '{source_vs}', query '{query_skill}' (k=10) -> Got {len(skill_docs_retrieved)} docs.")
        
        local_debug_info.append(f"  Total unique detailed content docs retrieved: {len(all_content_docs)}")
        if not all_content_docs:
            err_msg = (
                f"Found a resume for '{name}', but I couldn't extract specific skills information clearly. "
                f"The resume might need to be reprocessed, or the information isn't in a standard format. You could try:\n"
                f"- Asking for general information about {name} from their resume.\n"
                f"- Reprocessing resumes via the sidebar button."
            )
            local_debug_info.append("  Found resume source, but no detailed skill content docs retrieved. Provided suggestions.")
            return {"docs": None, "error": err_msg, "debug_info_list": local_debug_info}
        
        local_debug_info.append("  Successfully retrieved detailed content docs.")
        return {"docs": all_content_docs, "error": None, "debug_info_list": local_debug_info}
        
    except Exception as e:
        print(f"[DEBUG] Error in get_resume_by_name: {e}")
        local_debug_info.append(f"[ERROR] in get_resume_by_name: {e}")
        err_msg = f"Error searching for {name}'s resume: {str(e)}"
        return {"docs": None, "error": err_msg, "debug_info_list": local_debug_info}

def extract_technical_skills(docs):
    """Extract and categorize technical skills from resume content."""
    if not docs:
        return {}
        
    content = " ".join([doc.page_content.lower() for doc in docs])
    
    found_skills = {}
    for category, skills in SKILL_CATEGORIES.items():
        category_skills = []
        for skill, variations in skills.items():
            # Look for skill variations in the content
            if any(variation in content for variation in variations):
                category_skills.append(skill)
            # Also look for variations in context (with word boundaries)
            elif any(f" {variation} " in f" {content} " for variation in variations):
                category_skills.append(skill)
        if category_skills:
            found_skills[category] = sorted(category_skills)  # Sort skills alphabetically
            
    return found_skills

def format_skills_response(name, skills_dict, source=None):
    """Format the skills into a readable response."""
    # This function is primarily formatting, debug info is less critical here unless complex logic is added.
    if not skills_dict:
        return f"No technical skills were found in {name}'s resume."
        
    response = f"🔍 **Technical Skills Summary for {name}**\n\n"
    
    for category, skills in skills_dict.items():
        if skills:  # Only show categories that have skills
            response += f"**{category}**:\n"
            response += ", ".join(skill.title() for skill in skills)
            response += "\n\n"
    
    if source:
        response += f"\nSource Resume:\n- {source}"
        
    return response

def get_alternative_prompts(original_query, bot_answer, bot_debug_info, llm_api_key):
    """
    Generates alternative user prompts based on the original query and the bot's last response (and its debug info).
    Args:
        original_query (str): The user's last query.
        bot_answer (str): The bot's last answer.
        bot_debug_info (str): The debug string associated with the bot's last answer.
        llm_api_key (str): API key for LLM if needed for advanced suggestions.
    Returns:
        list: A list of suggested alternative prompt strings (typically 2-3).
    """
    suggestions = []
    debug_info_str = str(bot_debug_info).lower()
    original_query_lower = original_query.lower()

    # Heuristic 1: Failed name search in resumes
    if "could not find a resume for" in bot_answer.lower() or ("no documents found by get_resume_by_name" in debug_info_str and "specific person skill search" in debug_info_str):
        suggestions.append("Try checking the spelling of the name.")
        suggestions.append("List all available resumes to see participant names.")
        if "skills" in original_query_lower:
            name_parts = [p for p in original_query_lower.split() if p not in ["technical", "skills"]]
            if name_parts:
                suggestions.append(f"Ask for a general summary of '{' '.join(name_parts).title()}'s resume first.")
        else:
             suggestions.append("Search for specific skills instead of a name (e.g., 'Participants with Python skills').")

    # Heuristic 2: Failed skill search or role profile search (no one found)
    elif ("couldn't find any participants with" in bot_answer.lower() or 
          "couldn't find participants strongly matching" in bot_answer.lower() or 
          ("general skill search" in debug_info_str and "couldn't definitively confirm" in bot_answer.lower())):
        suggestions.append("Try using broader skill terms (e.g., 'cloud technologies' instead of 'AWS S3').")
        suggestions.append("Search for a different skill or combination of skills.")
        suggestions.append("List all resumes and check them individually if you suspect the information exists.")

    # Heuristic 3: Time query that didn't find a specific event or was ambiguous
    elif ("general time query did not match a specific event" in debug_info_str or 
          "couldn't find any specific events scheduled" in bot_answer.lower() or 
          "couldn't understand the start time" in bot_answer.lower()):
        suggestions.append("Ask for the full event agenda.")
        suggestions.append("Try specifying AM/PM for the time (e.g., '2 PM' or '14:00').")
        suggestions.append("Ask about a specific named event in the agenda (e.g., 'When is the keynote?').")

    # Heuristic 4: Session recommendations couldn't be made or were very generic
    elif ("couldn't find any workshop sessions to recommend" in bot_answer.lower() or 
          "no specific workshop sessions found to recommend" in bot_answer.lower()):
        suggestions.append("Ensure workshops.pdf is in the 'data' directory or workshops are in the agenda.")
        suggestions.append("If asking for yourself, try phrasing like: 'Recommend sessions for a data scientist.'")
        suggestions.append("View the full agenda for all session details.")

    # Heuristic 5: General fallback was used, or answer is very generic like "I don't know"
    elif ("conversationalretrievalchain result" in debug_info_str and 
          ("don't know" in bot_answer.lower() or "sorry, i couldn't process" in bot_answer.lower() or "sorry, the context you provided does not contain" in bot_answer.lower())):
        suggestions.append("Try rephrasing your question to be more specific.")
        suggestions.append("Ask a simpler, related question first.")
        if "resume" in original_query_lower or "skill" in original_query_lower:
            suggestions.append("Ensure resumes have been processed recently if asking about them.")
        else:
            suggestions.append("Check the agenda for event-related information.")

    # Default suggestions if no specific heuristic matched
    if not suggestions:
        suggestions.append("Try rephrasing your question.")
        suggestions.append("Ask a more general question about the event.")
        suggestions.append("View the event agenda for an overview.")

    # Limit to 3 suggestions for brevity
    return suggestions[:3]

if __name__ == '__main__':
    st.title("QA Handler Test")
    utils.initialize_data_files() # Ensure data files exist

    mock_api_key = st.text_input("Enter Google API Key for testing QA Handler", type="password")

    if mock_api_key:
        genai.configure(api_key=mock_api_key)
        st.info("Google GenAI configured for QA Handler testing.")

        st.subheader("Test Basic Info")
        st.write("Agenda Query: What's the agenda?")
        st.write("Response:", get_bot_response("What's the agenda?", mock_api_key))
        st.write("Washroom Query: Where is the washroom?")
        st.write("Response:", get_bot_response("Where is the washroom?", mock_api_key))
        st.write("Time Query: How long until lunch?")
        st.write("Response:", get_bot_response("How long until lunch?", mock_api_key))
        st.write("Time Range Query: What is from 10 AM to 11 AM?", get_bot_response("What is from 10 AM to 11 AM?", mock_api_key) ) # Test new time range query

        st.subheader("Test Resume Q&A (requires vector store)")
        # To test this, you need to have a vector_store populated via resume_processor.py testing
        # For this standalone test, we'll try to load it if it exists.
        if "test_vector_store" not in st.session_state:
            st.session_state.test_vector_store = resume_processor.get_existing_vector_store(mock_api_key)
        
        if st.session_state.test_vector_store:
            st.success("Test vector store loaded.")
            resume_query = st.text_input("Ask a question based on resumes (e.g., 'Who has experience in Python?')")
            if resume_query:
                st.write("Response:", get_bot_response(resume_query, mock_api_key, vector_store_instance=st.session_state.test_vector_store, chat_history=[]))
        else:
            st.warning("Could not load a test vector store. Run resume_processor.py tests to create one at vector_store/event_attendees_faiss, then rerun this.")

        st.subheader("Test Conversational Feedback")
        if "test_chat_history_feedback" not in st.session_state:
            st.session_state.test_chat_history_feedback = []

        feedback_query = st.text_input("Start feedback (e.g., 'I want to give feedback') or continue conversation:", key="feedback_q")
        if feedback_query:
            st.session_state.test_chat_history_feedback.append({"role": "user", "content": feedback_query})
            response_data = get_bot_response(feedback_query, mock_api_key, chat_history=st.session_state.test_chat_history_feedback) # get_bot_response returns a dict
            st.session_state.test_chat_history_feedback.append({"role": "assistant", "content": response_data.get("answer")})
            st.write("Feedback Response:", response_data.get("answer"))
            # st.write("Current feedback mode:", st.session_state.get("feedback_mode", False))

        st.subheader("Test General Conversation")
        general_query = st.text_input("Ask a general question (e.g., 'Hello')", key="general_q")
        if general_query:
             st.write("Response:", get_bot_response(general_query, mock_api_key, chat_history=[]))
        
        st.subheader("Test Alternative Prompts")
        test_orig_query = "who is rajesh kumar and what are his skills"
        test_bot_answer = "Could not find a resume for rajesh kumar."
        test_debug_info = "[REASONING] Query Tentatively Matched 'Specific Person Skill Search'. Name: rajesh kumar\n  No documents found by get_resume_by_name for rajesh kumar."
        st.write(f"Testing suggestions for: Q='{test_orig_query}', A='{test_bot_answer}'")
        st.write("Suggestions: ", get_alternative_prompts(test_orig_query, test_bot_answer, test_debug_info, mock_api_key))

    else:
        st.warning("Please enter a Google API Key to test QA Handler features.")