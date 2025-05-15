import json
import os
import streamlit as st
import datetime
from langchain_community.document_loaders import PyPDFLoader
import re # Import regular expressions module

DATA_DIR = "data"
AGENDA_FILE = os.path.join(DATA_DIR, "agenda.json")
LOCATIONS_FILE = os.path.join(DATA_DIR, "locations.json")
PDF_AGENDA_FILENAME = "Agenda.pdf"
PDF_AGENDA_FILE_PATH = os.path.join(DATA_DIR, PDF_AGENDA_FILENAME)
WORKSHOPS_PDF_FILE = os.path.join(DATA_DIR, "workshops.pdf")

DEFAULT_AGENDA = {
    "title": "Workshop on AI Innovations",
    "date": "2024-07-28",
    "schedule": [
        {"time": "09:00 AM", "event": "Registration & Welcome Coffee"},
        {"time": "09:30 AM", "event": "Opening Keynote: The Future of AI", "speaker": "Dr. AI Expert"},
        {"time": "10:30 AM", "event": "Session 1: Deep Learning Advances", "track": "Technical"},
        {"time": "10:30 AM", "event": "Session A: AI in Business", "track": "Business"},
        {"time": "11:30 AM", "event": "Coffee Break & Networking"},
        {"time": "12:00 PM", "event": "Session 2: Natural Language Processing Trends", "track": "Technical"},
        {"time": "12:00 PM", "event": "Session B: Ethical AI", "track": "Business"},
        {"time": "01:00 PM", "event": "Lunch Break"},
        {"time": "02:00 PM", "event": "Session 3: Computer Vision Applications", "track": "Technical"},
        {"time": "02:00 PM", "event": "Session C: AI Startups Showcase", "track": "Business"},
        {"time": "03:00 PM", "event": "Workshop: Hands-on with Generative AI", "materials_needed": "Laptop with internet access"},
        {"time": "04:30 PM", "event": "Closing Remarks & Feedback Collection"},
        {"time": "05:00 PM", "event": "Networking Reception"}
    ]
}

DEFAULT_LOCATIONS = {
    "washroom_men": "Down the main hall, to your left, Room 101.",
    "washroom_women": "Down the main hall, to your right, Room 102.",
    "washroom_accessible": "Next to the registration desk, Room 103.",
    "main_hall": "You are currently in or near the main hall where keynotes and large sessions take place.",
    "session_1_technical_room": "Room 201, second floor.",
    "session_a_business_room": "Room 205, second floor.",
    "lunch_area": "Cafeteria, ground floor, at the end of the west wing.",
    "registration_desk": "At the main entrance of the venue.",
    "first_aid": "Room G02, near the security office."
}

def initialize_data_files():
    """Creates data directory and default data files if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(AGENDA_FILE):
        with open(AGENDA_FILE, 'w') as f:
            json.dump(DEFAULT_AGENDA, f, indent=4)
        st.info(f"Created default agenda file: {AGENDA_FILE}")
    
    if not os.path.exists(LOCATIONS_FILE):
        with open(LOCATIONS_FILE, 'w') as f:
            json.dump(DEFAULT_LOCATIONS, f, indent=4)
        st.info(f"Created default locations file: {LOCATIONS_FILE}")

initialize_data_files() # Ensure files are created when module is imported

def extract_text_from_pdf(pdf_file_path):
    """Extracts all text from a PDF file."""
    try:
        loader = PyPDFLoader(pdf_file_path)
        documents = loader.load()
        if not documents: # Check if PyPDFLoader returned any documents
            st.warning(f"No documents were loaded from PDF: {os.path.basename(pdf_file_path)}. The PDF might be empty or unreadable by the loader.")
            return None
        
        page_contents = [doc.page_content for doc in documents if hasattr(doc, 'page_content') and doc.page_content]
        if not page_contents:
            st.warning(f"Documents loaded from PDF {os.path.basename(pdf_file_path)}, but they contained no extractable page content.")
            return None
            
        full_text = "\n".join(page_contents)
        if not full_text.strip(): # Check if extracted text is just whitespace
            st.warning(f"Extracted text from PDF {os.path.basename(pdf_file_path)} is empty or contains only whitespace.")
            return None
        return full_text
    except Exception as e:
        st.error(f"Error extracting text from PDF {os.path.basename(pdf_file_path)}: {e}")
        return None

def parse_schedule_from_text(text_content):
    """
    Attempts to parse a schedule (list of time-event dicts) from raw text 
    using a simple regex. This is highly dependent on the PDF text format.
    Example format it looks for: 'HH:MM AM/PM - Event Description' on a new line.
    """
    parsed_schedule = []
    # New regex for formats like "10:00 - 11:00 AM: Event Description" or "1:00 PM: Event" 
    regex = r"^\s*(\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?\s*(?:AM|PM|am|pm))\s*:\s*(.+?)\s*$"
    
    if not text_content:
        return parsed_schedule

    for line in text_content.split('\n'):
        match = re.match(regex, line)
        if match:
            full_time_str = match.group(1).strip() # e.g., "10:00 - 11:00 AM" or "1:00 PM"
            event_desc = match.group(2).strip()
            
            # For time calculations, we need a single start time.
            # If it's a range, take the first part. Otherwise, it's already a single time.
            # Example: "10:00 - 11:00 AM" -> "10:00 AM"
            # Example: "1:00 PM" -> "1:00 PM"
            start_time_match = re.match(r"^(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))", full_time_str)
            if start_time_match:
                simple_time_str = start_time_match.group(1)
                parsed_schedule.append({"time": simple_time_str, "event": event_desc, "original_time_display": full_time_str})
            else: # Should not happen if outer regex matched, but as a safeguard
                parsed_schedule.append({"time": full_time_str, "event": event_desc, "original_time_display": full_time_str})
    
    if not parsed_schedule:
        st.info("Could not parse a structured schedule (time - event) from the provided text using the current regex.")
        
    return parsed_schedule

def load_agenda():
    """Loads the event agenda from a JSON file."""
    try:
        with open(AGENDA_FILE, 'r') as f:
            agenda = json.load(f)
        return agenda
    except FileNotFoundError:
        st.error(f"Agenda file not found: {AGENDA_FILE}. Returning default.")
        return DEFAULT_AGENDA
    except json.JSONDecodeError:
        st.error(f"Error decoding agenda JSON from {AGENDA_FILE}. Returning default.")
        return DEFAULT_AGENDA

def get_agenda_string():
    """
    Returns the agenda as a formatted string.
    Prioritizes Agenda.pdf from the DATA_DIR if available,
    then falls back to agenda.json, then to the default hardcoded agenda.
    Sets session_state variables 'is_pdf_agenda_active' and 'agenda_source_message'.
    """
    # Clear previous state vars to ensure they are fresh for current call context
    st.session_state.is_pdf_agenda_active = False
    st.session_state.agenda_source_message = ""
    st.session_state.parsed_pdf_schedule = [] # Initialize/clear parsed schedule from PDF
    st.session_state.loaded_pdf_agenda_text = "" # Initialize/clear raw PDF text

    # 1. Try to load from Agenda.pdf
    if os.path.exists(PDF_AGENDA_FILE_PATH):
        pdf_text = extract_text_from_pdf(PDF_AGENDA_FILE_PATH)
        if pdf_text:
            st.session_state.is_pdf_agenda_active = True
            st.session_state.agenda_source_message = f"Event agenda loaded from '{PDF_AGENDA_FILENAME}'."
            st.session_state.loaded_pdf_agenda_text = pdf_text
            
            # Attempt to parse the schedule from the PDF text
            st.session_state.parsed_pdf_schedule = parse_schedule_from_text(pdf_text)
            if st.session_state.parsed_pdf_schedule:
                st.session_state.agenda_source_message += " Successfully parsed schedule items from PDF."
            else:
                st.session_state.agenda_source_message += " Could not parse detailed schedule items from PDF text; showing raw text."

            return f"**Agenda (from {PDF_AGENDA_FILENAME}):**\\n\\n{pdf_text}"
        else:
            # PDF exists but couldn't be read
            st.session_state.agenda_source_message = (
                f"Found '{PDF_AGENDA_FILENAME}', but could not extract text. "
                f"Falling back to '{os.path.basename(AGENDA_FILE)}'."
            )
            # Proceed to JSON fallback
    else:
        # Agenda.pdf not found
        st.session_state.agenda_source_message = (
            f"'{PDF_AGENDA_FILENAME}' not found in '{DATA_DIR}'. "
            f"Looking for '{os.path.basename(AGENDA_FILE)}'."
        )
        # Proceed to JSON fallback

    # 2. Try to load from agenda.json (or use default if json also fails)
    # Ensure is_pdf_agenda_active is False if we are here
    st.session_state.is_pdf_agenda_active = False 
    # Clear any loaded PDF text and parsed schedule if we are falling back
    if "loaded_pdf_agenda_text" in st.session_state:
        del st.session_state.loaded_pdf_agenda_text
    if "parsed_pdf_schedule" in st.session_state:
        del st.session_state.parsed_pdf_schedule
        
    agenda = load_agenda() # This loads from JSON or DEFAULT_AGENDA
    
    # Update source message if it was about PDF failure/absence
    if AGENDA_FILE in st.session_state.agenda_source_message or PDF_AGENDA_FILENAME in st.session_state.agenda_source_message :
        if os.path.exists(AGENDA_FILE) and agenda != DEFAULT_AGENDA:
             st.session_state.agenda_source_message += f" Using '{os.path.basename(AGENDA_FILE)}'."
        else: # Indicates default agenda is likely used
             st.session_state.agenda_source_message += " Using default built-in agenda."
    else: # Should not happen if logic above is correct, but as a safeguard
        if os.path.exists(AGENDA_FILE) and agenda != DEFAULT_AGENDA:
            st.session_state.agenda_source_message = f"Agenda loaded from '{os.path.basename(AGENDA_FILE)}'."
        else:
            st.session_state.agenda_source_message = "Using default built-in agenda."

    agenda_str = f"**{agenda.get('title', 'Event Agenda')} ({agenda.get('date', 'N/A')})**\\n\\n"
    for item in agenda.get('schedule', []):
        # When displaying JSON/Default agenda, use the original structure
        time_display = item.get('original_time_display', item.get('time'))
        agenda_str += f"**{time_display}**: {item['event']}"
        if item.get('speaker'):
            agenda_str += f" (Speaker: {item['speaker']})"
        if item.get('track'):
            agenda_str += f" [Track: {item['track']}]"
        if item.get('materials_needed'):
             agenda_str += f" (Bring: {item['materials_needed']})"
        agenda_str += "\\n"
    return agenda_str

def load_locations():
    """Loads location information from a JSON file."""
    try:
        with open(LOCATIONS_FILE, 'r') as f:
            locations = json.load(f)
        return locations
    except FileNotFoundError:
        st.error(f"Locations file not found: {LOCATIONS_FILE}. Returning default.")
        return DEFAULT_LOCATIONS
    except json.JSONDecodeError:
        st.error(f"Error decoding locations JSON from {LOCATIONS_FILE}. Returning default.")
        return DEFAULT_LOCATIONS

def get_location_info(place_query):
    """Returns information about a specific location if found."""
    locations = load_locations()
    # Simple keyword matching, can be improved with fuzzy matching or NLP
    for key, value in locations.items():
        if place_query.lower() in key.lower().replace("_", " "):
            return value
    return "I'm sorry, I don't have information about that specific location. You can ask about common places like washrooms, session rooms, or the lunch area."

def get_time_until_event(event_name):
    """Calculates the time remaining until a specific event (e.g., lunch)."""
    schedule_to_use = None
    agenda_source_for_time = "structured (JSON/default)"

    if st.session_state.get("is_pdf_agenda_active", False) and st.session_state.get("parsed_pdf_schedule"):
        schedule_to_use = st.session_state.parsed_pdf_schedule
        agenda_source_for_time = f"parsed from {PDF_AGENDA_FILENAME}"
    elif st.session_state.get("is_pdf_agenda_active", False):
        # PDF is active, but we couldn't parse a schedule from it.
        return (
            f"The agenda is currently loaded from '{PDF_AGENDA_FILENAME}', but I couldn't extract specific event timings from its content. "
            "Please refer to the displayed PDF agenda text for details."
        )
    else:
        # Use JSON/default agenda
        loaded_json_agenda = load_agenda() # This loads from JSON or default
        schedule_to_use = loaded_json_agenda.get('schedule', [])

    if not schedule_to_use:
        return f"No schedule information found from {agenda_source_for_time} agenda to look for '{event_name}'."

    now = datetime.datetime.now()
    event_time_str = None
    original_time_display_for_event = None # Store the original display string

    for item in schedule_to_use:
        if isinstance(item, dict) and 'event' in item and 'time' in item:
            if event_name.lower() in item['event'].lower():
                event_time_str = item['time']
                original_time_display_for_event = item.get('original_time_display', event_time_str) # Get original if available
                break
    
    if not event_time_str:
        return f"I couldn't find '{event_name}' in the agenda to calculate the time."

    # Use original_event_time_str_for_display for the final message, but parsing_time_str for logic
    # Ensure original_event_time_str_for_display is set from the loop's finding
    final_display_time = original_time_display_for_event if original_time_display_for_event else event_time_str
    parsing_time_str = event_time_str # This is usually the simplified start time

    if "-" in parsing_time_str and "am" not in parsing_time_str.lower() and "pm" not in parsing_time_str.lower():
        # If it's a range like "1:00-2:00" and AM/PM is missing from the start part,
        # but might be in the original_time_display_for_event (e.g., "1:00-2:00 PM")
        # We will rely on the subsequent parsing logic and potential PM inference.
        parsing_time_str = parsing_time_str.split("-")[0].strip()
    elif "-" in parsing_time_str: # Handles "1:00 AM - 2:00 AM"
        parsing_time_str = parsing_time_str.split("-")[0].strip()


    print(f"[DEBUG TIME_UTIL] Event: '{event_name}', Original Display: '{final_display_time}', Initial Parsing Str: '{parsing_time_str}'")

    try:
        now = datetime.datetime.now()
        parsed_successfully = False
        
        # Attempt 1: Parse with AM/PM (e.g., "1:00 PM" or "01:00 PM")
        try:
            event_time_dt_obj = datetime.datetime.strptime(parsing_time_str, "%I:%M %p")
            print(f"[DEBUG TIME_UTIL] Parsed with %I:%M %p: {event_time_dt_obj.time()}")
            parsed_successfully = True
        except ValueError:
            pass

        # Attempt 2: Parse as 24-hour format (e.g., "13:00")
        if not parsed_successfully:
            try:
                event_time_dt_obj = datetime.datetime.strptime(parsing_time_str, "%H:%M")
                print(f"[DEBUG TIME_UTIL] Parsed with %H:%M: {event_time_dt_obj.time()}")
                # If this results in an early hour (e.g. 1 AM for "1:00")
                # and the original display string had "PM", adjust it.
                if 1 <= event_time_dt_obj.hour <= 11 and original_time_display_for_event and "pm" in original_time_display_for_event.lower():
                    event_time_dt_obj = event_time_dt_obj.replace(hour=event_time_dt_obj.hour + 12)
                    print(f"[DEBUG TIME_UTIL] Adjusted to PM based on original_time_display: {event_time_dt_obj.time()}")
                elif event_time_dt_obj.hour == 12 and original_time_display_for_event and "am" in original_time_display_for_event.lower(): # Handle "12:xx AM"
                     event_time_dt_obj = event_time_dt_obj.replace(hour=0) # 12 AM is hour 0
                     print(f"[DEBUG TIME_UTIL] Adjusted 12:xx AM to hour 0: {event_time_dt_obj.time()}")

                parsed_successfully = True
            except ValueError:
                pass
        
        # Attempt 3: Fallback for "H:MM" or "HH:MM" without AM/PM, with more explicit PM inference
        if not parsed_successfully:
            try:
                hour, minute = map(int, parsing_time_str.split(':'))
                print(f"[DEBUG TIME_UTIL] Fallback parsing: H={hour}, M={minute}")
                # If original_time_display_for_event had "PM" and hour is 1-11, it's PM.
                # Or if it's 12 and original had "PM", it's 12 PM (noon).
                if (1 <= hour <= 11 and original_time_display_for_event and "pm" in original_time_display_for_event.lower()) or \
                   (hour == 12 and original_time_display_for_event and "pm" in original_time_display_for_event.lower()):
                    if hour != 12: # 1 PM to 11 PM
                        hour += 12
                    # hour is already 12 for 12 PM
                    print(f"[DEBUG TIME_UTIL] Fallback inferred PM. Hour set to: {hour}")
                # If original_time_display_for_event had "AM" and hour is 12, it's 12 AM (midnight).
                elif hour == 12 and original_time_display_for_event and "am" in original_time_display_for_event.lower():
                    hour = 0 # 12 AM is hour 0
                    print(f"[DEBUG TIME_UTIL] Fallback inferred 12 AM (midnight). Hour set to: {hour}")
                # Add a general heuristic for times typically in PM if AM/PM totally missing
                elif not original_time_display_for_event or ("am" not in original_time_display_for_event.lower() and "pm" not in original_time_display_for_event.lower()):
                    if 1 <= hour <= 7: # Heuristic: 1 PM to 7 PM are common event end times if AM/PM is missing
                        if event_name.lower() in ["lunch", "closing remarks", "reception", "dinner"]: # More likely PM
                             hour += 12
                             print(f"[DEBUG TIME_UTIL] Fallback heuristic PM for typical event. Hour set to: {hour}")

                event_time_dt_obj = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                parsed_successfully = True
            except ValueError:
                return f"Could not parse time string '{parsing_time_str}' for event '{event_name}'. Original display: '{final_display_time}'."

        if not parsed_successfully: # Should be caught by the last return, but as a safeguard
             return f"Failed to parse time for '{event_name}' from '{parsing_time_str}'. Original: '{final_display_time}'"

        event_datetime = now.replace(hour=event_time_dt_obj.hour, minute=event_time_dt_obj.minute, second=0, microsecond=0)
        print(f"[DEBUG TIME_UTIL] Event datetime: {event_datetime}, Now: {now}")

        if now > event_datetime:
            return f"The event '{event_name}' (scheduled for {final_display_time} from {agenda_source_for_time}) has already passed today."
        else:
            time_diff = event_datetime - now
            hours, remainder = divmod(time_diff.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            if hours > 0:
                return f"Time until '{event_name}' (scheduled for {final_display_time} from {agenda_source_for_time}): {hours} hour(s) and {minutes} minute(s)."
            else:
                return f"Time until '{event_name}' (scheduled for {final_display_time} from {agenda_source_for_time}): {minutes} minute(s)."
    except Exception as e:
        error_message = f"Error calculating time for '{event_name}' scheduled at {final_display_time} (from {agenda_source_for_time}). Error: {e}"
        print(f"Error in get_time_until_event for '{event_name}'. Original time string: '{final_display_time}', Parsed as: '{parsing_time_str}'. Error: {e}") 
        return error_message

def parse_time_string_to_datetime(time_str, original_display_str=None, now_for_date=None):
    """
    Helper function to parse various time string formats (e.g., "1:00 PM", "13:00", "1:00")
    into a datetime.time object or a full datetime.datetime object if now_for_date is provided.
    Uses original_display_str for AM/PM inference if primary parsing fails.
    Returns a datetime.time object.
    """
    if now_for_date is None:
        now_for_date = datetime.datetime.now()

    parsed_time_obj = None
    parsing_attempts_debug = []

    # Attempt 1: Parse with AM/PM (e.g., "1:00 PM", "01:00 PM")
    try:
        dt_obj = datetime.datetime.strptime(time_str, "%I:%M %p")
        parsed_time_obj = dt_obj.time()
        parsing_attempts_debug.append(f"Parsed '{time_str}' with %I:%M %p -> {parsed_time_obj}")
    except ValueError:
        parsing_attempts_debug.append(f"Failed to parse '{time_str}' with %I:%M %p")

    # Attempt 2: Parse as 24-hour format (e.g., "13:00")
    if not parsed_time_obj:
        try:
            dt_obj = datetime.datetime.strptime(time_str, "%H:%M")
            parsed_time_obj = dt_obj.time()
            parsing_attempts_debug.append(f"Parsed '{time_str}' with %H:%M -> {parsed_time_obj}")
            # If this results in an early hour (e.g. 1 AM for "1:00")
            # and the original display string had "PM", adjust it.
            if 1 <= parsed_time_obj.hour <= 11 and original_display_str and "pm" in original_display_str.lower():
                parsed_time_obj = parsed_time_obj.replace(hour=parsed_time_obj.hour + 12)
                parsing_attempts_debug.append(f"Adjusted to PM based on original_display: {parsed_time_obj}")
            elif parsed_time_obj.hour == 12 and original_display_str and "am" in original_display_str.lower(): # Handle "12:xx AM"
                 parsed_time_obj = parsed_time_obj.replace(hour=0) # 12 AM is hour 0
                 parsing_attempts_debug.append(f"Adjusted 12:xx AM to hour 0: {parsed_time_obj}")
        except ValueError:
            parsing_attempts_debug.append(f"Failed to parse '{time_str}' with %H:%M")
    
    # Attempt 3: Fallback for "H:MM" or "HH:MM" without AM/PM, with more explicit PM inference
    if not parsed_time_obj:
        try:
            hour, minute = map(int, time_str.split(':'))
            parsing_attempts_debug.append(f"Fallback parsing for '{time_str}': H={hour}, M={minute}")
            
            temp_time_obj_for_inference = now_for_date.replace(hour=hour, minute=minute, second=0, microsecond=0)

            if (1 <= hour <= 11 and original_display_str and "pm" in original_display_str.lower()) or \
               (hour == 12 and original_display_str and "pm" in original_display_str.lower()): # Covers 12 PM
                if hour != 12: # 1 PM to 11 PM
                    hour += 12
                parsing_attempts_debug.append(f"Fallback inferred PM. Hour set to: {hour}")
            elif hour == 12 and original_display_str and "am" in original_display_str.lower(): # Covers 12 AM
                hour = 0 
                parsing_attempts_debug.append(f"Fallback inferred 12 AM (midnight). Hour set to: {hour}")
            # No general heuristic for AM/PM if completely missing, rely on explicit parsing or context.
            
            parsed_time_obj = temp_time_obj_for_inference.replace(hour=hour).time()
            parsing_attempts_debug.append(f"Fallback resulted in time: {parsed_time_obj}")

        except ValueError:
            parsing_attempts_debug.append(f"Fallback parsing failed for '{time_str}'")
            print(f"[DEBUG TIME_PARSE_HELPER] All parsing attempts failed for '{time_str}'. Attempts: {parsing_attempts_debug}")
            return None # Indicate failure

    print(f"[DEBUG TIME_PARSE_HELPER] Parsed '{time_str}' (original: '{original_display_str}') -> {parsed_time_obj}. Attempts: {parsing_attempts_debug}")
    return parsed_time_obj

def get_events_in_time_range(start_time_str, end_time_str=None):
    """
    Finds agenda events that occur within a specified time range.
    Args:
        start_time_str (str): The start of the time range (e.g., "12:00 PM", "10:30").
        end_time_str (str, optional): The end of the time range. 
                                      If None, looks for events starting exactly at start_time_str 
                                      or events that span over start_time_str if they are ranges.
    Returns:
        str: A formatted string listing events in the range, or a message if none are found.
    """
    # Determine the schedule source (PDF or JSON/default)
    # This reuses some logic from get_agenda_string() initialization
    # and get_time_until_event() to decide which schedule to use.
    
    schedule_to_use = None
    agenda_source_for_time_range = "structured (JSON/default)" # Default assumption
    now = datetime.datetime.now() # For constructing full datetime objects

    # Call get_agenda_string once to ensure session state for agenda source is set
    _ = get_agenda_string() # Result not directly used here, but sets session_state

    if st.session_state.get("is_pdf_agenda_active", False) and st.session_state.get("parsed_pdf_schedule"):
        schedule_to_use = st.session_state.parsed_pdf_schedule
        agenda_source_for_time_range = f"parsed from {PDF_AGENDA_FILENAME}"
        print(f"[DEBUG TIME_RANGE] Using PDF schedule. Items: {len(schedule_to_use)}")
    elif st.session_state.get("is_pdf_agenda_active", False):
        # PDF is active, but we couldn't parse a schedule from it.
        return (
            f"The agenda is currently loaded from '{PDF_AGENDA_FILENAME}', but I couldn't extract specific event timings "
            "from its content to check for events in the range {start_time_str}{'-' + end_time_str if end_time_str else ''}."
        )
    else:
        # Use JSON/default agenda
        loaded_json_agenda = load_agenda() # This loads from JSON or default
        schedule_to_use = loaded_json_agenda.get('schedule', [])
        print(f"[DEBUG TIME_RANGE] Using JSON/Default schedule. Items: {len(schedule_to_use)}")

    if not schedule_to_use:
        return f"No schedule information found from {agenda_source_for_time_range} agenda to find events for {start_time_str}."

    # Parse user's start and end times
    user_start_dt_time = parse_time_string_to_datetime(start_time_str, original_display_str=start_time_str)
    user_end_dt_time = None
    if end_time_str:
        user_end_dt_time = parse_time_string_to_datetime(end_time_str, original_display_str=end_time_str)

    if not user_start_dt_time:
        return f"I couldn't understand the start time '{start_time_str}'. Please use a format like '10:00 AM' or '2 PM'."
    if end_time_str and not user_end_dt_time:
        return f"I couldn't understand the end time '{end_time_str}'. Please use a format like '11:00 AM' or '3 PM'."
    
    if user_end_dt_time and user_start_dt_time > user_end_dt_time:
        return f"The start time '{start_time_str}' seems to be after the end time '{end_time_str}'. Please check the times."

    print(f"[DEBUG TIME_RANGE] User Query Times: Start='{user_start_dt_time}' ({start_time_str}), End='{user_end_dt_time}' ({end_time_str})")

    found_events = []
    for item in schedule_to_use:
        if not isinstance(item, dict) or 'event' not in item or 'time' not in item:
            continue

        event_desc = item['event']
        event_time_display = item.get('original_time_display', item['time']) # For display in result
        
        # The 'time' field in schedule_to_use is usually the simplified start time for parsing.
        # The 'original_time_display' field holds the full string like "10:00 - 11:00 AM".
        
        # We need to parse the event's start and potentially end time from its schedule entry.
        # Use original_time_display for robust parsing of ranges.
        event_time_for_parsing = item['time'] # This is often the already simplified start time
        
        # Check if the event_time_display itself is a range
        time_range_match = re.match(r"^\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\s*-\s*(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\s*$", event_time_display, re.IGNORECASE)
        
        event_start_dt_time = None
        event_end_dt_time = None

        if time_range_match: # Event is a range like "10:00 AM - 11:00 AM"
            raw_event_start_str = time_range_match.group(1).strip()
            raw_event_end_str = time_range_match.group(2).strip()
            
            event_start_dt_time = parse_time_string_to_datetime(raw_event_start_str, original_display_str=event_time_display) # Pass full original for context
            event_end_dt_time = parse_time_string_to_datetime(raw_event_end_str, original_display_str=event_time_display)
            print(f"[DEBUG TIME_RANGE] Event '{event_desc}': Parsed range {raw_event_start_str}->{event_start_dt_time} to {raw_event_end_str}->{event_end_dt_time}")
        else: # Event is a single point in time like "1:00 PM"
            event_start_dt_time = parse_time_string_to_datetime(event_time_for_parsing, original_display_str=event_time_display)
            # For single point events, we can consider them to have a nominal duration, e.g., 1 hour, or just check start time
            # For simplicity now, if no user_end_dt_time, we match exact start. If user_end_dt_time exists, we check if event_start is within range.
            if event_start_dt_time and user_end_dt_time: # If user provides a range, this single point event should also have an "end"
                 # Assume a 1-hour duration for single point events when comparing against a user's range
                 # This is a heuristic. A better approach might be to define durations in agenda.json
                 event_end_dt_time = (now.replace(hour=event_start_dt_time.hour, minute=event_start_dt_time.minute) + datetime.timedelta(hours=1)).time()
            print(f"[DEBUG TIME_RANGE] Event '{event_desc}': Parsed single point {event_time_for_parsing}->{event_start_dt_time}, assumed end for range check: {event_end_dt_time}")


        if not event_start_dt_time:
            print(f"[DEBUG TIME_RANGE] Could not parse start time for event: '{event_desc}' with time string '{event_time_for_parsing}'. Skipping.")
            continue

        # Overlap/Containment Logic:
        # User range: [U_start, U_end]
        # Event range: [E_start, E_end]
        # Overlap if: E_start < U_end AND E_end > U_start

        overlap = False
        if user_end_dt_time: # User specified a range
            if event_end_dt_time: # Event is also a range (or assumed range)
                # Scenario 1: Event [E_start, E_end] overlaps/is within User [U_start, U_end]
                if event_start_dt_time < user_end_dt_time and event_end_dt_time > user_start_dt_time:
                    overlap = True
            else: # Event is a single point, User specified a range
                # Scenario 2: Event E_start is within User [U_start, U_end]
                if user_start_dt_time <= event_start_dt_time < user_end_dt_time:
                    overlap = True
        else: # User specified a single point in time (U_start), no U_end
            if event_end_dt_time: # Event is a range
                # Scenario 3: User's U_start is within Event [E_start, E_end)
                if event_start_dt_time <= user_start_dt_time < event_end_dt_time:
                    overlap = True
            else: # Both user and event are single points
                # Scenario 4: User's U_start is exactly Event's E_start
                if event_start_dt_time == user_start_dt_time:
                    overlap = True
        
        if overlap:
            event_info_str = f"**{event_time_display}**: {event_desc}"
            if item.get('speaker'):
                event_info_str += f" (Speaker: {item['speaker']})"
            if item.get('track'):
                event_info_str += f" [Track: {item['track']}]"
            if item.get('materials_needed'):
                event_info_str += f" (Bring: {item['materials_needed']})"
            found_events.append(event_info_str)
            print(f"[DEBUG TIME_RANGE] Found overlapping event: {event_info_str}")

    if not found_events:
        user_range_str = f"around {start_time_str}"
        if end_time_str:
            user_range_str = f"between {start_time_str} and {end_time_str}"
        return f"I couldn't find any specific events scheduled {user_range_str} based on the {agenda_source_for_time_range} agenda. You can ask for the full agenda to see all timings."
    else:
        return (f"🗓️ Here's what I found scheduled for **{start_time_str}{'-' + end_time_str if end_time_str else ''}** "
                f"(from {agenda_source_for_time_range} agenda):\\n\\n" + "\\n".join(found_events))

def parse_workshops_from_pdf():
    """Extracts and parses workshop information from the workshops PDF file."""
    if not os.path.exists(WORKSHOPS_PDF_FILE):
        return None
        
    workshops_text = extract_text_from_pdf(WORKSHOPS_PDF_FILE)
    if not workshops_text:
        return None
        
    workshops = []
    current_workshop = {}
    
    # Split text into lines and process
    lines = workshops_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            if current_workshop:
                workshops.append(current_workshop)
                current_workshop = {}
            continue
            
        # Look for workshop title (usually starts with "Workshop:" or contains common title patterns)
        if line.lower().startswith('workshop:') or re.match(r'^[A-Z].*\b(workshop|session)\b', line, re.IGNORECASE):
            if current_workshop:
                workshops.append(current_workshop)
            current_workshop = {'title': line.replace('Workshop:', '').strip()}
            continue
            
        # Look for time information
        time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)(?:\s*-\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))?)', line)
        if time_match:
            current_workshop['time'] = time_match.group(1)
            continue
            
        # Look for prerequisites or requirements
        if 'require' in line.lower() or 'prerequisite' in line.lower():
            current_workshop['prerequisites'] = line
            continue
            
        # Look for description
        if 'description' not in current_workshop:
            current_workshop['description'] = line
        else:
            current_workshop['description'] += ' ' + line
            
    # Add the last workshop if exists
    if current_workshop:
        workshops.append(current_workshop)
        
    return workshops

def get_workshop_recommendations(participant_background):
    """
    Matches participant background with available workshops to provide personalized recommendations.
    Args:
        participant_background (str): Text describing participant's skills and interests
    Returns:
        list: List of recommended workshops with relevance explanations
    """
    workshops = parse_workshops_from_pdf()
    if not workshops:
        # Fallback to agenda-based workshops if no separate workshop PDF
        agenda = load_agenda()
        workshops = [
            item for item in agenda.get('schedule', [])
            if 'Workshop' in item.get('event', '') or 'Session' in item.get('event', '')
        ]
    
    recommendations = []
    for workshop in workshops:
        # For workshops from PDF
        if isinstance(workshop, dict) and 'title' in workshop:
            title = workshop['title']
            description = workshop.get('description', '')
            time = workshop.get('time', '')
            prerequisites = workshop.get('prerequisites', '')
        # For workshops from agenda
        else:
            title = workshop.get('event', '')
            description = workshop.get('description', '')
            time = workshop.get('time', '')
            prerequisites = workshop.get('materials_needed', '')
        
        # Calculate relevance score based on keyword matching
        relevance_score = 0
        relevance_reasons = []
        
        # Extract keywords from participant background
        background_keywords = set(participant_background.lower().split())
        
        # Compare with workshop content
        title_lower = title.lower()
        description_lower = description.lower()
        prerequisites_lower = prerequisites.lower()

        workshop_title_keywords = set(title_lower.split())
        workshop_desc_prereq_keywords = set(description_lower.split()) | set(prerequisites_lower.split())

        title_matches = background_keywords.intersection(workshop_title_keywords)
        desc_prereq_matches = background_keywords.intersection(workshop_desc_prereq_keywords)
        
        # Combined matching keywords for general reason (can be refined)
        # For now, we will build reasons based on title and desc/prereq matches separately.
        # all_matching_keywords = title_matches | desc_prereq_matches

        if title_matches:
            relevance_score += len(title_matches) * 2 # Weight title matches more
            relevance_reasons.append(f"Title mentions relevant terms: {', '.join(sorted(list(title_matches)))}")
        
        # Add to score from description/prerequisites, avoiding double counting keywords already in title_matches
        # and ensuring we only add reasons for NEW keywords found here.
        unique_desc_prereq_matches = desc_prereq_matches - title_matches
        if unique_desc_prereq_matches:
            relevance_score += len(unique_desc_prereq_matches)
            relevance_reasons.append(f"Description/prerequisites also mention: {', '.join(sorted(list(unique_desc_prereq_matches)))}")
        elif not title_matches and desc_prereq_matches: # If no title matches, but desc_prereq_matches exist
            relevance_score += len(desc_prereq_matches)
            relevance_reasons.append(f"Description/prerequisites mention: {', '.join(sorted(list(desc_prereq_matches)))}")


        # Add learning opportunity if workshop contains keywords not in background
        workshop_content_keywords = workshop_title_keywords | workshop_desc_prereq_keywords
        workshop_unique_keywords = workshop_content_keywords - background_keywords
        learning_keywords = {word for word in workshop_unique_keywords 
                           if len(word) > 3 and word not in {'workshop', 'session', 'the', 'and', 'for'}}
        
        if learning_keywords:
            relevance_reasons.append(f"Learning opportunity: {', '.join(list(learning_keywords)[:3])}")
        
        if relevance_score > 0 or learning_keywords:
            recommendations.append({
                'title': title,
                'time': time,
                'relevance_score': relevance_score,
                'relevance_reasons': relevance_reasons,
                'description': description,
                'prerequisites': prerequisites
            })
    
    # Sort by relevance score
    recommendations.sort(key=lambda x: x['relevance_score'], reverse=True)
    return recommendations

if __name__ == '__main__':
    st.title("Utilities Test")
    st.subheader("Agenda")
    st.markdown(get_agenda_string())

    st.subheader("Locations")
    st.write("Washroom (Men):", get_location_info("washroom men"))
    st.write("Lunch:", get_location_info("lunch area"))
    st.write("Unknown:", get_location_info("meeting room x"))

    st.subheader("Time Until Event")
    st.write("Time until lunch:", get_time_until_event("lunch"))
    st.write("Time until Opening Keynote:", get_time_until_event("Opening Keynote"))
    st.write("Time until an old event (e.g. Welcome Coffee if it\'s past 9 AM):")
    st.write(get_time_until_event("Welcome Coffee"))
    st.write("Time until a non-existent event:", get_time_until_event("My Secret Meeting"))

    # Test PDF Agenda (assuming a test_agenda.pdf exists in DATA_DIR)
    # To test this section: create a simple PDF named 'test_agenda.pdf' in your 'data' directory.
    # For example, it could just contain the text: "10:00 AM - Test Event from PDF"
    # test_pdf_path = os.path.join(DATA_DIR, "test_agenda.pdf")
    # if os.path.exists(test_pdf_path):
    #     st.session_state.agenda_pdf_text = extract_text_from_pdf(test_pdf_path)
    #     if st.session_state.agenda_pdf_text:
    #         st.subheader("Agenda from PDF Test")
    #         st.markdown(get_agenda_string())
    #     del st.session_state.agenda_pdf_text # Clean up for other tests
    # else:
    #     st.info(f"To test PDF agenda, place a 'test_agenda.pdf' in the '{DATA_DIR}' directory.") 

    # Updated test for get_agenda_string considering Agenda.pdf
    st.subheader("Agenda Logic Test")
    st.markdown("**Instructions for testing `get_agenda_string` with your PDF format:**")
    st.markdown(
        "1. Ensure `Agenda.pdf` (with format like '10:00 - 11:00 AM: Event') is in the `data` directory.\n"
        "2. Check the 'Agenda Status' in the sidebar for parsing messages.\n"
        "3. Verify the displayed agenda and test 'Time until lunch'."
    )
    st.markdown("---CURRENT AGENDA LOADED--- mesons ")
    current_agenda_display = get_agenda_string() # This call will set the session_state vars
    st.info(st.session_state.get("agenda_source_message", "No agenda source message set."))
    st.markdown(current_agenda_display)
    st.markdown("-----------------------------")

    # Test get_time_until_event based on the current agenda source
    st.subheader("Time Until Event (logic depends on agenda source)")
    st.write(f"(is_pdf_agenda_active: {st.session_state.get('is_pdf_agenda_active', False)}, has_parsed_pdf_schedule: {bool(st.session_state.get('parsed_pdf_schedule'))})")
    st.write("Time until lunch:", get_time_until_event("lunch"))
    st.write("Time until Opening Keynote:", get_time_until_event("Opening Keynote"))
    st.write("Time until an old event (e.g. Welcome Coffee if it\'s past 9 AM):")
    st.write(get_time_until_event("Welcome Coffee"))
    st.write("Time until a non-existent event:", get_time_until_event("My Secret Meeting")) 