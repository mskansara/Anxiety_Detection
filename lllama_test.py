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

"""


# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("AUTH_KEY")

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
).cpu()


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
