import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Meeting Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ---------------- SESSION STATE ---------------- #

if "transcript" not in st.session_state:
    st.session_state.transcript = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

if "mom" not in st.session_state:
    st.session_state.mom = None

# ---------------- HEADER ---------------- #

st.title("🧠 Meeting Intelligence System")
st.markdown("Upload → Ask → Generate Insights → Export")

st.divider()

# ---------------- RESET BUTTON ---------------- #

col_reset, _ = st.columns([1, 6])
with col_reset:
    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# ---------------- UPLOAD SECTION ---------------- #

st.header("📤 Upload Meeting Audio")

uploaded_file = st.file_uploader(
    "Upload your meeting audio",
    type=["mp3", "wav", "m4a"]
)

if uploaded_file:
    if st.button("🚀 Process Audio"):
        with st.spinner("Transcribing audio... ⏳"):
            response = requests.post(
                f"{API_BASE}/upload_audio",
                files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            )

            if response.status_code == 200:
                data = response.json()
                st.session_state.transcript = data["transcript"]
                st.success("✅ Audio processed successfully!")
            else:
                st.error(response.text)

# Show transcript
if st.session_state.transcript:
    st.subheader("📝 Transcript")
    st.text_area(
        "Transcript",
        st.session_state.transcript,
        height=200
    )

st.divider()

# ---------------- ASK SECTION ---------------- #

st.header("❓ Ask Questions")

query = st.text_input("Ask something about the meeting")

col1, col2 = st.columns(2)

with col1:
    if st.button("💬 Get Answer"):
        if not query.strip():
            st.warning("Enter a question first")
        else:
            with st.spinner("Thinking... 🤔"):
                response = requests.post(
                    f"{API_BASE}/ask",
                    json={"query": query}
                )

                if response.status_code == 200:
                    st.session_state.answer = response.json()["answer"]
                    st.session_state.audio_file = None
                else:
                    st.error(response.text)

with col2:
    if st.button("🔊 Answer with Voice"):
        if not query.strip():
            st.warning("Enter a question first")
        else:
            with st.spinner("Generating voice response... 🎧"):
                response = requests.post(
                    f"{API_BASE}/ask-tts",
                    json={"query": query}
                )

                if response.status_code == 200:
                    data = response.json()
                    st.session_state.answer = data["answer"]
                    st.session_state.audio_file = data["audio_file"]
                else:
                    st.error(response.text)

# Show answer
if st.session_state.answer:
    st.subheader("💬 Answer")
    st.write(st.session_state.answer)

# Show audio
if st.session_state.audio_file:
    audio_url = f"{API_BASE}/audio/{st.session_state.audio_file}"
    st.audio(audio_url)

st.divider()

# ---------------- MOM SECTION ---------------- #

st.header("📄 Generate Minutes of Meeting")

if st.button("🧾 Generate MoM"):
    with st.spinner("Creating MoM..."):
        response = requests.post(f"{API_BASE}/generate-mom")

        if response.status_code == 200:
            st.session_state.mom = response.json()["minutes_of_meeting"]
        else:
            st.error(response.text)

if st.session_state.mom:
    st.subheader("📝 Minutes of Meeting")
    st.text_area("MoM", st.session_state.mom, height=250)

st.divider()

# ---------------- PPT SECTION ---------------- #

st.header("📊 Export Presentation")

if st.button("📊 Generate PPT"):
    with st.spinner("Building presentation..."):
        response = requests.post(f"{API_BASE}/generate-ppt")

        if response.status_code == 200:
            st.success("✅ PPT Ready!")

            st.download_button(
                label="📥 Download PPT",
                data=response.content,
                file_name="meeting_summary.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
        else:
            st.error(response.text)