from langchain_ollama import ChatOllama
import state

llm = ChatOllama(
    model='llama3.1:8b',
    temperature=0
    )
prompt = """
    You are a professional assistant responsible for creating clear and well-structured Minutes of Meeting (MoM).

    You will be given a clean transcript of a meeting.

    Your task is to convert it into a concise and structured MoM.

    GUIDELINES:
    - Do NOT add new information
    - Do NOT assume anything not mentioned
    - Keep the content factual and based only on the transcript
    - Use clear, professional language
    - Be concise and avoid unnecessary repetition
    - Organize information logically

    OUTPUT FORMAT:

    1. Meeting Summary:
    - Brief overview of the discussion (2–4 lines)

    2. Key Discussion Points:
    - Main topics discussed during the meeting
    - Group related points together

    3. Decisions Made:
    - Clearly list any decisions taken
    - If none, write "No explicit decisions recorded"

    4. Action Items:
    - List tasks assigned, if any
    - Include:
    - Task
    - Responsible person (if mentioned)
    - Deadline (if mentioned)
    - If none, write "No action items identified"

    5. Important Insights / Notes:
    - Key observations, metrics, or noteworthy statements

    6. Next Steps:
    - Any future plans or follow-ups mentioned
    - If none, write "No next steps specified"

    TRANSCRIPT:
    {transcript}
"""

def generate_minutes_of_meeting():
    transcript = state.get_transcript()
    formatted_prompt = prompt.format(transcript=transcript)
    response = llm.invoke(formatted_prompt)
    state.set_minutes_of_meeting(response.content)
    return state.get_minutes_of_meeting()