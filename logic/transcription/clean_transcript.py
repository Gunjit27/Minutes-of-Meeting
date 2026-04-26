from langchain_ollama import ChatOllama
import state

llm = ChatOllama(
    model='llama3.1:8b',
    temperature=0
)

prompt = """
    You are an expert in cleaning transcripts.

    You will be given a raw meeting transcript.

    INSTRUCTIONS:
    - Fix grammar and sentence structure
    - Merge broken sentences
    - Remove repeated words or phrases
    - Convert spoken numbers into digits with units 
    (e.g., "twenty five" → "25", "one hundred million" → "$100 million")
    - DO NOT rephrase or summarize
    - DO NOT add new information
    - Keep the original meaning EXACTLY the same
    - Output only the cleaned transcript (no explanations)

    TRANSCRIPT:
    {transcript}
"""

def clean_transcript(transcript):
    text = ' '.join([segment['text'] for segment in transcript])

    response = llm.invoke(prompt.format(transcript=text))
    val = response.content.strip()
    state.set_tramscript(val)
    return state.get_transcript()