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
You are a specialist in summarising the transcriptions and moods of the patient in a psychologist clinic. You have summarise the transcription and mood of the patient to the doctor in a paragraph format which looks like notes for the doctor.
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


Summarise this transcription as a long paragraph which can be used as notes for the doctor and the patient.
The format should be as follows - Patient's Details, today's date, Summary of the Session, Key Points, Mood Summary, Next Steps for the patient, Additional Notes. Don't provide the transcription and detected mood in the summary.
The report should be in structured way. Do not provide any additional content like pseudo-code or any content from the prompt.
"""


# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("AUTH_TOKEN")

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
    low_cpu_mem_usage=True,
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    max_new_tokens=2048,
)


def get_response(prompt):
    sequences = text_generator(prompt)
    gen_text = sequences[0]["generated_text"]
    return gen_text


result = get_response(prompt)
print(result)

# scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
# scores = scorer.score(referenceSummary, result)

# print(f"ROUGE-1 Score: {scores['rouge1'].fmeasure}")
# print(f"ROUGE-L Score: {scores['rougeL'].fmeasure}")
