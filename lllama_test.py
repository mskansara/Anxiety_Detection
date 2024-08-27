import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from dotenv import load_dotenv
import os
from rouge_score import rouge_scorer

# Reference summary for comparison
referenceSummary = """
**Patient's Name:** Jane Doe
**Session Date:** 2023-07-30
**Session Number:** 2

**Summary of the Session:**

The patient expressed feelings of anxiety, which started in high school during class presentations. These feelings have worsened over time, and now even mundane tasks like grocery shopping trigger anxiety. The patient mentioned that deep breathing helps, but not always. They also tried meditation, but found it difficult to focus. The patient has a supportive family, but they don't fully understand their struggles. They considered joining a support group, which the doctor suggested might be helpful.

The patient smiled and expressed happiness when recalling a recent event where they successfully managed their anxiety during a family gathering. This shows progress in their coping mechanisms. However, there were moments of sadness when discussing their ongoing struggles with anxiety and its impact on daily life. The patient also showed signs of anger when talking about the lack of understanding from some friends and the stigma associated with anxiety.

**Key Points:**

* Anxiety started in high school and worsened over time
* Deep breathing is a coping mechanism, but not always effective
* Meditation is difficult due to focus issues
* Supportive family, but lack of understanding
* Consideration of joining a support group
* Managed anxiety during a family gathering (happy moment)
* Frustration with lack of understanding from friends

**Mood Summary:**

The patient's mood varied throughout the session. They started with a neutral tone, but shifted to anxious and nervous as they discussed their struggles. They expressed frustration and anger when discussing the challenges of managing anxiety and the lack of understanding from friends. However, they also showed hopeful and happy emotions when discussing potential solutions and the doctor's support. The patient ended the session feeling grateful and relieved, showing a mix of emotions including happiness and sadness as they reflected on their progress and ongoing challenges.

**Next Steps:**

The patient will continue to work on managing their anxiety, possibly with the help of a support group. The doctor will provide guidance on coping mechanisms and recommend additional resources if needed. The patient will also explore strategies to improve focus during meditation. Additionally, the patient will be encouraged to engage in activities that previously brought them joy and happiness, to help counteract feelings of sadness and anxiety.

**Additional Notes:**

The patient's emotional expression and tone were closely monitored throughout the session. The detected mood scores and colors provided a visual representation of the patient's emotional state. The doctor will continue to use this data to inform their treatment plan and provide personalized support.

"""

prompt = """
You are a specialist in summarizing the transcriptions and moods of the patient in a psychologist's clinic. Your task is to summarize the patient's transcription and mood into a detailed report that the doctor can use for their notes. The summary should be structured in a formal and clinical manner.

Use the following format:
- **Patient's Details**: [Include the patient's name and relevant identifiers]
- **Session Date**: [Today's date]
- **Session Number**: [Specify which session it is]
- **Summary of the Session**: [Provide a detailed summary of the patient's statements and issues discussed during the session, highlighting any significant concerns or topics]
- **Key Points**: [List important points discussed, particularly any strategies, challenges, or plans mentioned]
- **Mood Summary**: [Summarize the emotional state of the patient throughout the session, noting any significant mood shifts or emotional expressions]
- **Next Steps**: [Outline the recommended actions or strategies for the patient moving forward]
- **Additional Notes**: [Include any extra observations, particularly regarding the patient's emotional tone, body language, or specific concerns raised during the session]

This is an example of the data which contains transcriptions and detected mood of the patient - 
{
  "transcriptions": [
    {
      "start_timestamp": "940710969012",
      "end_timestamp": "941510981137",
      "transcription": "Patient: I feel anxious sometimes. Doctor: Can you tell me more about that?"
    },
    {
      "start_timestamp": "941520981138",
      "end_timestamp": "942310971050",
      "transcription": "Patient: It's like there's always this feeling of unease, especially in social situations. Doctor: When did you first notice these feelings?"
    },
    {
      "start_timestamp": "942320971051",
      "end_timestamp": "943010971550",
      "transcription": "Patient: I think it started in high school. I used to get really nervous before class presentations. Doctor: Have these feelings of anxiety gotten better or worse over time?"
    },
    {
      "start_timestamp": "943020971551",
      "end_timestamp": "943810968887",
      "transcription": "Patient: They have gotten worse. Now, even going to the grocery store makes me anxious. Doctor: That sounds challenging. Have you found any strategies that help manage your anxiety?"
    },
    {
      "start_timestamp": "943820968888",
      "end_timestamp": "944510969387",
      "transcription": "Patient: Sometimes deep breathing helps, but not always. Doctor: It's good that you have a coping mechanism. Have you tried any other techniques or therapies?"
    },
    {
      "start_timestamp": "944520969388",
      "end_timestamp": "945810973062",
      "transcription": "Patient: I tried meditation, but I find it hard to focus. Doctor: Meditation can be difficult at first. It might help to start with shorter sessions and gradually increase the time."
    },
    {
      "start_timestamp": "945820973063",
      "end_timestamp": "946710972300",
      "transcription": "Patient: I will try that. Doctor: Do you have a support system you can rely on, like friends or family?"
    },
    {
      "start_timestamp": "946720972301",
      "end_timestamp": "947710972425",
      "transcription": "Patient: My family is supportive, but they don't really understand what I'm going through. Doctor: It can be hard for others to understand. Have you considered joining a support group?"
    },
    {
      "start_timestamp": "947720972426",
      "end_timestamp": "948710974850",
      "transcription": "Patient: I haven't, but maybe I should. Doctor: Support groups can provide a sense of community and understanding. It might be worth looking into."
    },
    {
      "start_timestamp": "948720974851",
      "end_timestamp": "949610976512",
      "transcription": "Patient: I'll think about it. Doctor: That's good to hear. Remember, you're not alone in this, and there are people who can help."
    },
    {
      "start_timestamp": "949620976513",
      "end_timestamp": "950610966887",
      "transcription": "Patient: Thank you, doctor. Doctor: You're welcome. Let's continue to work on this together."
    }
  ],
  "detected_mood": [
    {
      "timestamp": "940710969012",
      "emotion": "neutral",
      "score": 0.74,
      "colour": "white"
    },
    {
      "timestamp": "941520981138",
      "emotion": "anxious",
      "score": 0.65,
      "colour": "orange"
    },
    {
      "timestamp": "942320971051",
      "emotion": "nervous",
      "score": 0.72,
      "colour": "yellow"
    },
    {
      "timestamp": "943020971551",
      "emotion": "anxious",
      "score": 0.77,
      "colour": "orange"
    },
    {
      "timestamp": "943820968888",
      "emotion": "frustrated",
      "score": 0.68,
      "colour": "red"
    },
    {
      "timestamp": "944520969388",
      "emotion": "neutral",
      "score": 0.70,
      "colour": "white"
    },
    {
      "timestamp": "945820973063",
      "emotion": "hopeful",
      "score": 0.66,
      "colour": "blue"
    },
    {
      "timestamp": "946720972301",
      "emotion": "neutral",
      "score": 0.67,
      "colour": "white"
    },
    {
      "timestamp": "947720972426",
      "emotion": "considerate",
      "score": 0.64,
      "colour": "green"
    },
    {
      "timestamp": "948720974851",
      "emotion": "thoughtful",
      "score": 0.71,
      "colour": "blue"
    },
    {
      "timestamp": "949620976513",
      "emotion": "grateful",
      "score": 0.78,
      "colour": "blue"
    },
    {
      "timestamp": "950610966887",
      "emotion": "relieved",
      "score": 0.81,
      "colour": "blue"
    }
  ]
}

Generate the report accordingly, without including raw transcriptions or mood data or any other details or extra instructions.
"""


# Load environment variables
# load_dotenv()
# HF_TOKEN = os.getenv("AUTH_TOKEN")

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto",
#     token=HF_TOKEN,
#     low_cpu_mem_usage=True,
# )

# text_generator = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     torch_dtype=torch.bfloat16,
#     max_new_tokens=2048,
# )


# def get_response(prompt):
#     sequences = text_generator(prompt)
#     gen_text = sequences[0]["generated_text"]
#     return gen_text


# result = get_response(prompt)
# print(result)

import re


def check_summary_structure(summary):
    required_sections = [
        r"Patient's Details:",
        r"Session Date:",
        r"Session Number:",
        r"Summary of the Session:",
        r"Key Points:",
        r"Mood Summary:",
        r"Next Steps:",
        r"Additional Notes:",
    ]

    for section in required_sections:
        if not re.search(section, summary):
            print(f"Missing section: {section}")
            return False

    print("Summary structure is correct.")
    return True


result = """

Patient's Details: John Doe, ID: JD123
Session Date: 2024-03-10
Session Number: 5
Summary of the Session: The patient discussed feelings of anxiety, particularly in social situations. They reported that these feelings have worsened over time, affecting even mundane tasks like grocery shopping. The patient mentioned using deep breathing as a coping mechanism, but found it not always effective. They also expressed difficulty with meditation due to challenges in focusing. The patient's family is supportive, but they don't fully understand the patient's experiences. The patient considered joining a support group as a potential solution.
Key Points: 
- Anxiety in social situations has worsened over time.
- Deep breathing and meditation are used as coping mechanisms, but with limited success.
- The patient's family is supportive, but lacks understanding.
- Joining a support group is a potential solution.
Mood Summary: The patient's mood shifted throughout the session, starting with a neutral tone, then becoming anxious and nervous, followed by frustration, and eventually expressing hopefulness and gratitude. The patient's emotional state seemed to stabilize towards the end of the session.
Next Steps: The patient should explore joining a support group to connect with others who understand their experiences. The doctor will provide resources for support groups and continue to work on anxiety management strategies.
Additional Notes: The patient's emotional tone and body language appeared to relax as the session progressed, suggesting a positive response to the discussion. The patient's willingness to consider new solutions, such as joining a support group, indicates a proactive approach to managing their anxiety.  The doctor will continue to monitor the patient's progress and adjust the treatment plan as necessary.  The patient will be encouraged to keep a journal to track their anxiety levels and coping mechanisms.  The next session will be scheduled in two weeks to assess the patient's progress.  The doctor will also refer the patient to a therapist specializing in anxiety disorders for further guidance.  The patient will be given a copy of the treatment plan and will be asked to sign off on it.  The doctor will make a note to follow up with the patient's primary care physician to ensure that the patient's anxiety is being managed in conjunction with any other health issues.  The patient will be encouraged to reach out to the doctor's office if they experience any severe anxiety episodes.  The doctor will also provide the patient with a list of emergency contact numbers.  The patient will be asked to bring a list of their current medications to the next session.  The doctor will make a note to review the patient's medication list and to discuss any potential interactions with the patient.  The patient will be encouraged to keep a record of any changes in their symptoms or mood.  The doctor will make a note to review the patient's chart regularly to ensure that the treatment plan is working effectively.  The patient will be asked to attend a follow-up appointment in six weeks to assess their progress.  The doctor will also refer the patient to a psychiatrist for further evaluation and treatment if necessary.  The patient will be given a copy of the doctor's contact information and will be asked to keep it in their wallet or purse.  The doctor will make a note to follow up with the patient in three months to assess their progress.  The patient will be encouraged to reach out to the doctor's office if they experience any severe anxiety episodes.  The doctor will also provide the patient with a list of emergency contact numbers.  The patient will be asked to bring a list of their current medications to the next session.  The doctor will make a note to review the patient's medication list and to discuss any potential interactions with the patient.  The patient will be encouraged to keep a record of any changes in their symptoms or mood.  The doctor will make a note to review the patient's chart regularly to ensure that the treatment plan is working effectively.  The patient will be asked to attend a follow-up appointment in six weeks to assess their progress.  The doctor will also refer the patient to a psychiatrist for further evaluation and treatment if necessary.  The patient will be given a copy of the doctor's contact information and will be asked to keep it in their wallet or purse.  The doctor will make a note to follow up with the patient in three months to assess their progress.  The patient will be encouraged to reach out to the doctor's office if they experience any severe anxiety episodes.  The doctor will also provide the patient with a list of emergency contact numbers.  The patient will be asked to bring a list of their current medications to the next session.  The doctor will make a note to review the patient's medication list and to discuss any potential interactions with the patient.  The patient will be encouraged to keep a record of any changes in their symptoms or mood.  The doctor will make a note to review the patient's chart regularly to ensure that the treatment plan is working effectively.  The patient will be asked to attend a follow-up appointment in six weeks to assess their progress.  The doctor will also refer the patient to a psychiatrist for further evaluation and treatment if necessary.  The patient will be given a copy of the doctor's contact information and will be asked to keep it in their wallet or purse.  The doctor will make a note to follow up with the patient in three months to assess their progress.  The patient will be encouraged to reach out to the doctor's office if they experience any severe anxiety episodes.  The doctor will also provide the patient with a list of emergency contact numbers.  The patient will be asked to bring a list of their current medications to the next session.  The doctor will make a note to review the patient's medication list and to discuss any potential interactions with the patient.  The patient will be encouraged to keep a record of any changes in their symptoms or mood.  The doctor will make a note to review the patient's chart regularly to ensure that the treatment plan is working effectively.  The patient will be asked to attend a follow-up appointment in six weeks to assess their progress.  The doctor will also refer the patient to a psychiatrist for further evaluation and treatment if necessary.  The patient will be given a copy of the doctor's contact information and will be asked to keep it in their wallet or purse.  The doctor will make a note to follow up with the patient in three months to assess their progress.  The patient will be encouraged to reach out to the doctor's office if they experience any severe anxiety episodes.  The doctor will also provide the patient with a list of emergency contact numbers.  The patient will be asked to bring a list of their current medications to the next session.  The doctor will make a note to review the patient's medication list and to discuss any potential interactions with the patient.  The patient will be encouraged to keep a record of any changes in their symptoms or mood.  The doctor will make a note to review the patient's chart regularly to ensure that the treatment plan is working effectively.  The patient will be asked to attend a follow-up appointment in six weeks to assess their progress.  The doctor will also refer the patient to a psychiatrist for further evaluation and treatment if necessary.  The patient will be given a copy of the doctor's contact information and will be asked to keep it in their wallet or purse.  The doctor will make a note to follow up with the patient in three months to assess their progress.  The patient will be encouraged to reach out to the doctor's office if they experience any severe anxiety episodes.  The doctor will also provide the patient with a list of emergency contact numbers.  The patient will be asked to bring a list of their current medications to the next session.  The doctor will make a note to review the patient's medication list and to discuss any potential interactions with the patient.  The patient will be encouraged to keep a record of any changes in their symptoms or mood.  The doctor will make a note to review the patient's chart regularly to ensure that the treatment plan is working effectively.  The patient will be asked to attend a follow-up appointment in six weeks to assess their progress.  The doctor will also refer the patient to a psychiatrist for further evaluation and treatment if necessary.  The patient will be given a copy of the doctor's contact information and will be asked to keep it in their wallet or purse.  The doctor will make a note to follow up with the patient in three months to assess their progress.  The patient will be encouraged to reach out to the doctor's office if they experience any severe anxiety episodes.  The doctor will also provide the patient with a list of emergency contact numbers.  The patient will be asked to bring a list of their current medications to the next session.  The doctor will make a note to review the patient's medication list and to discuss any potential interactions with the patient.  The patient will be encouraged to keep a record of any changes in their symptoms or mood.  The doctor will make a note to review the patient's chart regularly to ensure that the treatment plan is working effectively.  The patient will be asked to attend a follow-up appointment in six weeks to assess their progress.  The doctor will also refer the patient to a psychiatrist for further evaluation and treatment if necessary.  The patient will be given a copy of the doctor's contact information and will be asked to keep it in their wallet or purse.  The doctor will make a note to follow up with the patient in three months to assess their progress.  The patient will be encouraged to reach out to the doctor's office if they experience any severe anxiety episodes.  The doctor will also provide the patient with a list of emergency contact numbers.  The patient will be asked to bring a list of their current medications to the next session.  The doctor will make a note to review the patient's medication list and to discuss any potential interactions with the patient.  The patient will be encouraged to keep a record of any changes in their symptoms or mood.  The doctor will make a note to review the patient's
"""

# scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
# scores = scorer.score(referenceSummary, result)

# print(f"ROUGE-1 Score: {scores['rouge1'].fmeasure}")
# print(f"ROUGE-L Score: {scores['rougeL'].fmeasure}")
is_structure_correct = check_summary_structure(result)
