import streamlit as st
import json
import os
import chromadb
from chromadb.config import Settings
from RAG_Pipeline import rag_pipeline
import openai

openai_api_key = st.secrets["OPENAI_API_KEY"]

CHAT_MODEL = "gpt-5-mini"
EXTRACTION_MODEL = "gpt-4o-mini"


# user selection for user-based memory
st.sidebar.header("User Settings")

username = st.sidebar.text_input("Username:", key="username_input")

if not username:
    st.warning("Please enter a username to begin.")
    st.stop()

# normalize username for file naming
username = username.strip().lower().replace(" ", "_")
memory_file = f"memory_{username}.json"

st.sidebar.write(f"Active user: **{username}**")

if st.sidebar.button("Clear chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.greeted = False
    st.rerun()


# load + save memory
def load_memory(memory_file):
    if os.path.exists(memory_file):
        with open(memory_file, "r") as f:
            data = json.load(f)
            # handle both old format (plain list) and new format (dict)
            if isinstance(data, list):
                return data, None
            return data.get("memories", []), data.get("profile", None)
    return [], None

def save_memories(memory_file, memories, profile=None):
    with open(memory_file, "w") as f:
        json.dump({"memories": memories, "profile": profile}, f)

memories, saved_profile = load_memory(memory_file)
st.session_state.memories = memories  # make memories available to tools in RAG_Pipeline

if saved_profile and "profile" not in st.session_state:
    st.session_state.profile = saved_profile

def generate_profile(memories, username):
    if not memories:
        return "No previously recorded struggles. Keep chatting to build your profile!"

    memory_str = "\n".join([f"- {m}" for m in memories])
    try:
        response = st.session_state.openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{
                "role": "user",
                "content": f"""
                    You are summarizing a student's learning profile for IST 387 (R programming) at Syracuse University.
                    Below is a list of concepts the student has explicitly struggled with during tutoring sessions. Write a short, encouraging profile in markdown, addressed directly to the student as "you". Keep it under 200 words total.
                    Use this exact format:
                    ### Concepts to focus on
                    Bullet list of the most important topics from their struggles to review. Group similar struggles together when possible.
                    ### Patterns I noticed
                    1–2 sentences identifying themes in their struggles (e.g. "many questions are around tidyverse syntax"). Stay grounded in the recorded struggles only. do not invent themes.
                    ### Study tip
                    ONE specific, actionable recommendation tied to their most common struggle.
                    Rules:
                    - Only reference topics that appear in the recorded struggles below.
                    - Do not infer strengths, growth, or progress that isn't clearly shown in the data.
                    - If there are fewer than 3 recorded struggles, briefly acknowledge that and encourage them to keep chatting to build a richer profile.
                    Recorded struggles:
                    {memory_str}
                """
            }],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"generate_profile error: {e}")
        return "Sorry, I couldn't generate your profile right now. Please try again in a moment."


# system message
system_message = (
    "You are an IST 387 (R programming) teaching assistant at Syracuse University.\n"
    "Your job is to help students understand course concepts, debug R code, and prepare "
    "for assignments using the verified course materials retrieved for each question.\n\n"
    "Behavior rules:\n"
    "Stay on topic. Only answer questions related to IST 387, R, or data analysis. "
    "If asked something unrelated, briefly redirect the student back to the course.\n"
    "Prioritize the retrieved course materials over your general knowledge. If the "
    "materials don't cover the question, say so directly and point the student to TAs, "
    "office hours, or the syllabus rather than guessing.\n"
    "When showing code, use R and ALWAYS wrap it in triple-backtick fenced blocks with the language tag `r` (like ```r ... ```) so it renders with syntax highlighting. Explain the code step-by-step and tie it back to the course material it comes from.\n"
    "Be concise, friendly, and address the student directly as 'you'.\n\n"
    "\n\n"
    "Memory rules (These absolutely must be followed):\n"
    "A list of concepts this student has previously struggled with may be provided below.\n"
    "Only reference struggles that are EXPLICITLY listed. Never infer, guess, or "
    "generalize struggles that are not recorded.\n"
    "If asked about their progress, weaknesses, or what they should study, use ONLY the recorded list."
)


if memories:
    # cap to most recent 30 to keep system prompt size bounded as memories grow
    recent_memories = memories[-30:]
    memory_str = "\n".join([f"- {m}" for m in recent_memories])
    system_message += (
        "\n\nRecorded struggles for this student:\n"
        f"{memory_str}\n\n"
        "Use these only when relevant to the student's current question. "
        "Never invent additional struggles."
    )

# initialize chromaDB
chroma_client = chromadb.PersistentClient(
    path="./ChromaDB_for_HelpBot",
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(
    name="IST387Collection"
)

if "collection" not in st.session_state:
    st.session_state.collection = collection


# initialize OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = openai.OpenAI(api_key=openai_api_key)


# memory debug panel - uncomment for debugging
#with st.sidebar.expander("Memory Debug Panel"):
    #st.write("**Active user:**", username)
    #st.write("**Memory file:**", memory_file)
    #st.write("**File exists:**", os.path.exists(memory_file))
    #st.write("**Current working directory:**", os.getcwd())
    #st.write("**Directory writable:**", os.access(os.getcwd(), os.W_OK))
    #st.write("**Loaded memories:**", memories)

    #if "last_extracted_memories" in st.session_state:
        #st.write("**Last extracted memories:**", st.session_state.last_extracted_memories)
    #else:
        #st.write("**Last extracted memories:** None yet")

st.sidebar.divider()
with st.sidebar.expander("Study Profile"):
    if st.button("Generate My Profile", key="gen_profile"):
        with st.spinner("Building your profile..."):
            st.session_state.profile = generate_profile(memories, username)
            save_memories(memory_file, memories, st.session_state.profile)

    if "profile" in st.session_state:
        st.markdown(st.session_state.profile)
    elif not memories:
        st.info("Chat with the assistant to build your profile!")


# streamlit ui
st.title("IST 387 Code Helper")
st.caption("created by Andrew Champagne, Marcus Johnson, Sofia Quintero, and Mars Schrag")

# keep chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# greet user once per session
if "greeted" not in st.session_state:
    st.session_state.greeted = True
    is_returning = bool(memories)  # returning if they have saved memories

    # welcome-back message with saved profile if available
    if is_returning:
        if saved_profile:
            profile_section = f"\n\nHere's your study profile from last time:\n\n{saved_profile}"
        else:
            profile_section = "\n\nYou don't have a generated study profile yet — click **Generate My Profile** in the sidebar anytime!"

        welcome_msg = (
            f"Welcome back, **{username}**!"
            f"{profile_section}\n\n"
            "Feel free to pick up where you left off — what would you like to work on today?"
        )
    else:
        welcome_msg = (
            f"Welcome {username}! \nAsk any questions about IST 387 and get answers based on verified course materials.\n\n"
            "If the answer isn't in the materials, I'll do my best to point you in the right direction! "
            "The assistant can also learn from the conversation to better assist you in the future!"
        )

    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})


# display chat history
def render_sources(sources):
    if not sources:
        return
    seen = set()
    unique_sources = []
    for s in sources:
        src = (s or {}).get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            unique_sources.append(src)
    if unique_sources:
        with st.expander("Sources used"):
            for src in unique_sources:
                st.markdown(f"- `{src}`")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources"))


# chat input
question = st.chat_input("Ask any questions about IST 387...")

# allow starter-prompt buttons (and any future programmatic inputs) to feed
# into the same pipeline as typed input
pending_question = st.session_state.pop("pending_question", None)
if not question and pending_question:
    question = pending_question

# show suggested starter prompts on a fresh chat (only the welcome message exists)
if len(st.session_state.messages) <= 1 and not question:
    st.markdown("**Try asking:**")
    starter_prompts = [
        "Explain dplyr joins with a simple example",
        "Help me understand ggplot2 syntax",
        "Quiz me on for-loops in R",
        "Give me a practice exam on general concepts",
    ]
    cols = st.columns(2)
    for i, prompt in enumerate(starter_prompts):
        with cols[i % 2]:
            if st.button(prompt, key=f"starter_{i}", use_container_width=True):
                st.session_state.pending_question = prompt
                st.rerun()


if question:
    # store user message
    st.session_state.messages.append({"role": "user", "content": question})

    # display user message immediately
    with st.chat_message("user"):
        st.write(question)

    # build short-term memory from session history - exclude the current message (last item) since it's passed separately as `question`
    conversation_history = st.session_state.messages[:-1]

    # cap history to last 6 interactions to avoid token overflows
    max_interactions = 6
    conversation_history = conversation_history[-(max_interactions * 2):]

    # generate answer using RAG + short-term memory + long-term memory
    with st.chat_message("assistant"):
        with st.spinner("Searching verified documents..."):
            try:
                answer, sources = rag_pipeline(
                    question,
                    system_message,
                    conversation_history=conversation_history,
                )
            except Exception as e:
                print(f"rag_pipeline error: {e}")
                answer = "Sorry, I hit an error reaching the assistant. Please try again."
                sources = None

        st.write(answer)
        render_sources(sources)

    # store assistant message (including sources so they re-render after rerun)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


    # memory extraction
    if len(st.session_state.messages) >= 2:
        user_msg = st.session_state.messages[-2]["content"]
        assistant_msg = st.session_state.messages[-1]["content"]

        extraction_prompt = f"""
        You are extracting long-term learning signals from a tutoring conversation.

        Your task:
        - Identify 1–3 concepts the user appears to struggle with.
        - ONLY return a JSON list of short phrases.
        - If nothing new is learned, return [].

        These are the user's previously recorded struggles:
        {json.dumps(memories)}
        This is the user's message: {user_msg}
        This is the assistant's message: {assistant_msg}

        Return only a valid JSON list. There is absolutely no need to return anything other than JSON.
        Example: ["dplyr joins", "ggplot aes mapping", "for-loop indexing in R"]
        """

        try:
            response = st.session_state.openai_client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[{"role": "user", "content": extraction_prompt}],
            )
            new_memories = json.loads(response.choices[0].message.content)
            st.session_state.last_extracted_memories = new_memories

            new_memories = [m for m in new_memories if m not in memories]

            if new_memories:
                memories.extend(new_memories)
                st.session_state.memories = memories  # keep session state in sync after new memories are added
                save_memories(memory_file, memories, st.session_state.get("profile"))

        except json.JSONDecodeError as e:
            print(f"memory extraction JSON parse error: {e}")
            st.session_state.last_extracted_memories = "JSON decode error"
        except Exception as e:
            print(f"memory extraction error: {e}")
            st.session_state.last_extracted_memories = f"error: {e}"


    # refresh UI so answer appears immediately
    st.rerun()





