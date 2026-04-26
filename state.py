TRANSCRIPT = None
def set_tramscript(transcript):
    global TRANSCRIPT
    TRANSCRIPT = transcript

def get_transcript():
    global TRANSCRIPT
    return TRANSCRIPT



TRANSCRIPT_TIMESTAMP = None
def set_transcript_timestamp(timestamp):
    global TRANSCRIPT_TIMESTAMP
    TRANSCRIPT_TIMESTAMP = timestamp

def get_transcript_timestamp():
    global TRANSCRIPT_TIMESTAMP
    return TRANSCRIPT_TIMESTAMP



VECTORSTORE = None
def set_vectorstore(vectorstore):
    global VECTORSTORE
    VECTORSTORE = vectorstore     

def get_vectorstore():
    global VECTORSTORE
    return VECTORSTORE




MINUTES_OF_MEETING = None
def set_minutes_of_meeting(minutes):
    global MINUTES_OF_MEETING
    MINUTES_OF_MEETING = minutes   

def get_minutes_of_meeting():
    global MINUTES_OF_MEETING
    return MINUTES_OF_MEETING