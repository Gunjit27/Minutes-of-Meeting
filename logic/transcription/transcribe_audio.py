from faster_whisper import WhisperModel
import state

model = WhisperModel(
                    'small',
                    device='cpu',
                    compute_type='int8'
                )

def transcribe_audio(audio_path: str):
    """Transcribe the audio file using Whisper Model."""

    segments,info = model.transcribe(
        audio_path, #Path
        language='en', #Setting language to Enlish to speed up process
        beam_size=5, #Beam size = how many guesses the model keeps while deciding the sentence
        best_of=3, #Best of = how many sentences the model considers before picking the best one
        vad_filter=True #Removes silence from audio to speed up process
    )

    val = [
        {
            "start": segment.start, #Timestamp
            "end": segment.end,
            "text": segment.text.strip()
        }
        for segment in segments
    ]
    state.set_tramscript(val)
    return state.get_transcript()
