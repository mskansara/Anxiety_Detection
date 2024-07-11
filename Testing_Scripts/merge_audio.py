# Assign test case number to the different test cases
# Using command line arguements, select the test case required
# Based on the test case number, using switch cases, pick up the original transcript (pre-decided the name)
# Compare the new transcription with the original and calculate the error rate

# Task 1
# Gather audio files for three different accents -

from pydub import AudioSegment

sound1 = AudioSegment.from_mp3("../clips/common_voice_en_39586336.mp3")
sound2 = AudioSegment.from_mp3("../clips/common_voice_en_39586337.mp3")
sound3 = AudioSegment.from_mp3("../clips/common_voice_en_39586338.mp3")
sound4 = AudioSegment.from_mp3("../clips/common_voice_en_39586339.mp3")
sound5 = AudioSegment.from_mp3("../clips/common_voice_en_39586340.mp3")
sound6 = AudioSegment.from_mp3("../clips/common_voice_en_39586347.mp3")
sound7 = AudioSegment.from_mp3("../clips/common_voice_en_39586348.mp3")
sound8 = AudioSegment.from_mp3("../clips/common_voice_en_39586349.mp3")
sound9 = AudioSegment.from_mp3("../clips/common_voice_en_39586350.mp3")
combined_sounds = (
    sound1 + sound2 + sound3 + sound4 + sound5 + sound6 + sound7 + sound8 + sound9
)
combined_sounds.export("../clips/Testing_Clips/accent3.mp3", format="mp3")
