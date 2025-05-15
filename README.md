# AI Event Bot

Welcome to the AI Event Bot! This interactive chatbot is designed to assist attendees and organizers during workshops, conferences, or events. It provides information about the event schedule, locations, helps analyze attendee resumes for skill matching and networking, and can offer workshop recommendations.

## Features

*   **Event Information Retrieval:**
    *   Displays the event agenda (loaded from `Agenda.pdf` if present in the `data` directory, otherwise falls back to `data/agenda.json` or a default).
    *   Provides information about event locations (e.g., washrooms, session rooms, lunch area) based on `data/locations.json`.
    *   Calculates and informs about the time remaining until specific events (e.g., "Time to Lunch") with improved time parsing.
    *   Can find events within a specified time range (e.g., "What's happening between 10 AM and 11 AM?").
*   **Advanced Query Understanding & Reasoning:**
    *   The bot employs a multi-priority system in `qa_handler.py` to interpret user queries.
    *   It uses regex, keyword matching, and contextual analysis to differentiate between requests for general event info, specific resume details, skill-based searches across all participants, and session recommendations.
    *   This internal reasoning allows it to route queries to the most appropriate processing logic or LLM prompt.
*   **Resume Analysis & Attendee Insights (Requires Google API Key):**
    *   **Processing:** Ingests and processes multiple resumes (PDF and DOCX formats) from the `resumes/` directory.
    *   **Vector Store:** Creates and utilizes a FAISS vector store (`vector_store/event_attendees_faiss`) for efficient semantic search over resume content using Google's generative AI embeddings.
    *   **Skill-Based Queries:**
        *   Find participants with specific technical skills (e.g., "who has python skill?", "participants with AWS, Docker, or cloud technologies experience").
        *   Identify participants based on years of experience (e.g., "participants with 5 years of experience in Java").
        *   Search for participants matching a specific role profile and associated skills (e.g., "Find Data Engineer profiles with skills like ETL, Python, Spark").
    *   **Educational Background:** Find participants based on their educational qualifications (e.g., "find participants based on their educational background").
    *   **General Detail Extraction:** Supports queries for broader categories of information across all resumes (e.g., "list all technical skills", "summary of work experience for all participants", "find participants with experience in data analysis, machine learning, or AI projects").
    *   **Networking Assistance:** Suggests participants with similar technical backgrounds by grouping them based on shared skills from `SKILL_CATEGORIES`.
    *   **Individual Lookups:**
        *   Retrieves and summarizes technical skills for a specific named participant (e.g., "Priya Sharma skills") with improved name matching and error handling.
        *   Provides a concise professional summary for a specific named participant's resume (e.g., "Summarize Rajesh Kumar's resume").
    *   **Fallback Logic:** If context is missing for a specific resume query, the bot may use the first available resume or guide the user.
    *   **List All Resumes:** Allows users to see a list of all processed and available resumes (e.g., "list all resumes").
*   **Workshop & Session Recommendations (Requires Google API Key):**
    *   Provides personalized workshop recommendations based on an individual's background.
        *   If a user asks for recommendations for "me", the bot will prompt for their background if not previously provided.
        *   If a specific background is given (e.g. "Recommend sessions for a data scientist"), it uses that.
    *   Can generate recommendations for all processed resumes, listing suggestions for each participant (e.g., triggered by "Recommended Sessions for me" button or phrases like "workshop recommendations for all participants").
    *   Utilizes information from `data/workshops.pdf` if available for detailed workshop content, otherwise falls back to workshop/session information in the main agenda.
*   **Conversational Capabilities:**
    *   Engages in general Q&A related to the event.
    *   Collects event feedback in a conversational manner, with improved state management for feedback mode.
    *   Offers alternative prompt suggestions if a user indicates a response was not helpful (via "thumbs down").
*   **User-Friendly Interface:**
    *   Built with Streamlit for an interactive web application.
    *   Sidebar for API key configuration, quick actions (Agenda, Washroom, Time to Lunch, Feedback), and predefined resume queries.
*   **Debugging Mode:**
    *   Displays an expandable section below each bot response showing detailed debug information, including the reasoning path taken, context used (like retrieved document snippets), LLM prompts, and relevant function calls.

## Technology Stack

*   **Python 3.9+**
*   **Streamlit:** For the web application interface.
*   **LangChain:** Framework for developing applications powered by language models.
*   **Google Generative AI:** Utilizes Gemini models for embeddings and chat completions.
*   **FAISS:** For efficient similarity search in the vector store.
*   **PyPDFLoader & python-docx:** For extracting text from PDF and DOCX resume files.
*   **Langchain-Community:** For various Langchain components.
*   **Regular Expressions (re module):** Used extensively for parsing queries, PDF content, and extracting specific information.

## Setup and Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```

2.  **Create a Python Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Ensure you have the `requirements.txt` file from the project.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Google API Key:**
    *   You need a Google API Key with the Generative Language API (Gemini) enabled.
    *   You can obtain one from the [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   The application will prompt you to enter this key in the sidebar when you first run it. The key is stored in the Streamlit session state and is not written to any files.

5.  **Prepare Data Directories and Files:**
    Create the following directory structure at the root of the project:

    ```
    your_project_root/
    ├── data/
    │   ├── agenda.json         # Optional: Custom agenda (see format in utils.py DEFAULT_AGENDA)
    │   ├── locations.json      # Optional: Custom locations (see format in utils.py DEFAULT_LOCATIONS)
    │   ├── Agenda.pdf          # Optional: If provided, this PDF agenda will be parsed and used.
    │   └── workshops.pdf       # Optional: PDF detailing workshops for recommendation.
    ├── resumes/                # Place attendee resume files (PDF or DOCX) here.
    ├── vector_store/           # Auto-generated: Stores the FAISS index (event_attendees_faiss).
    ├── .streamlit/             # Optional: Streamlit configuration files (e.g. config.toml).
    ├── Newchatbot.py           # Main Streamlit application script.
    ├── qa_handler.py           # Handles question answering logic.
    ├── resume_processor.py     # Handles resume parsing and vector store creation.
    ├── utils.py                # Utility functions for agenda, locations, time, etc.
    └── requirements.txt        # Python dependencies.
    ```
    *   Default `agenda.json` and `locations.json` will be created if not found.

## Running the Application

1.  Ensure your virtual environment is activated.
2.  Navigate to the project's root directory in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run Newchatbot.py
    ```
4.  Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

## Using the Bot

*   **API Key:** Enter your Google API Key in the sidebar to enable AI features.
*   **Process Resumes:**
    *   Add PDF or DOCX resume files to the `resumes/` directory. Ensure the `resumes` directory exists.
    *   Click the "Process Resumes" button in the sidebar. This will read the resumes, create embeddings, and build the FAISS vector store. You need to do this for resume-related queries to work.
*   **Chat Interface:** Type your questions into the chat input at the bottom of the screen.
*   **Quick Actions & Queries:** Use the buttons in the sidebar for common questions or to trigger predefined resume analysis queries.
*   **Feedback:** Use the "👍" or "👎" buttons under each assistant response. A "👎" will trigger suggestions for alternative prompts.
*   **Debug Information:** If you need to understand the bot's reasoning, expand the "Show Debug Info" section below an assistant's message.

## Key Files Overview

*   `Newchatbot.py`: The main Streamlit application file that defines the UI and orchestrates calls to the backend logic.
*   `qa_handler.py`: Contains the core logic for interpreting user queries, routing them to the appropriate functions (agenda, location, resume analysis, LLM calls), and formatting responses. It also manages the debug information collection.
*   `resume_processor.py`: Responsible for finding resume files, extracting text, generating embeddings using Google's AI, and creating/loading the FAISS vector store.
*   `utils.py`: Provides helper functions for loading/parsing agenda and location data (from JSON or PDF), workshop information (from PDF or agenda), time calculations, and text extraction from PDFs.

## Architecture & Documentation

A detailed textual description of the system architecture can be found in the conversation history leading to this update.

For a visual representation of the architecture and a video walkthrough/demo, please refer to the `Documentation/` folder located at the root of this project. This folder should contain:
*   An architecture diagram (e.g., `architecture.png` or `architecture.pdf`).
*   A video file (e.g., `event_bot_demo.mp4`).

## Troubleshooting

*   **Missing API Key:** Most features, especially resume analysis and generative Q&A, will not work without a valid Google API Key.
*   **"No resumes processed yet":** Ensure you have placed resume files in the `resumes/` folder and clicked the "Process Resumes" button.
*   **Accuracy of Resume Queries:** The quality of resume parsing, the comprehensiveness of the `SKILL_CATEGORIES` in `qa_handler.py`, and the robustness of regex patterns significantly impact accuracy. For new technical domains or query types, these might need updates.
*   **Time Parsing Issues:** If time-related queries (e.g., "time until lunch") fail, ensure the time formats in `Agenda.pdf` or `agenda.json` are compatible with the parsing logic in `utils.py`. The system now handles a wider variety of formats.
*   **Vector Store Issues:** If resume queries fail with errors like "vector store not populated" or if new resumes aren't found, ensure the "Process Resumes" button was clicked after adding/changing resumes and that there were no errors during processing. Check the `vector_store` directory.
*   **"Could not find resume for [name]":** This can happen if the resume PDF/DOCX is missing, not processed, or if the name matching logic in `qa_handler.py` (function `get_resume_by_name`) couldn't link the query to the file. The bot now provides more informative error messages and checks physical files.
*   **Newline Characters in Debug Info (`\\n`):** If debug information appears with literal `\\n` characters instead of actual newlines, it indicates a string formatting issue in `qa_handler.py` when the debug messages were constructed. This has been largely addressed, but keep an eye out.

## Further Enhancements (TODO)

- More robust Named Entity Recognition (NER) for location, event names, and person names in queries.
- Allow user to upload their own resume directly via the UI for instant personalized recommendations without needing to modify the `resumes` folder and reprocess.
- More sophisticated state management for feedback (e.g., knowing which specific session or topic the feedback pertains to if not explicitly stated).
- Database integration for persistent storage of feedback, event data, and (optionally) user profiles.
- Deployment instructions (e.g., Streamlit Cloud, Docker).
- More comprehensive automated testing for different query types and error conditions.
- UI/UX refinements, possibly theming or more interactive elements for displaying complex information like grouped similar profiles.
- Option to clear/reset the vector store directly from the UI for easier testing or if corruption is suspected.
- Enhanced parsing of `workshops.pdf` to extract more structured data (e.g., speaker, prerequisites, target audience). 