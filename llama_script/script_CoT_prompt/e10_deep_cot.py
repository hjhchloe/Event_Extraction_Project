import json
import requests
import os

API_KEY = os.getenv("FIREWORKS_API_KEY")

MODEL_ID = "accounts/fireworks/models/llama-v3p3-70b-instruct"

with open("/Users/hjhchloe/Documents/GitHub/Event_Extraction_Project/text/event10_deep_text.json", "r", encoding="utf-8") as f:
    data = json.load(f)

text = data["text"]

prompt = f"""extract all events from the following text.

Definition of event:
An event is a specific occurrence that happens at a particular time or place and involves one or more participants.

Follow these steps:

Step 1. Identify all event triggers in the text.
An event trigger is usually a verb or noun that clearly expresses the occurrence of an event.

Step 2. For each trigger, determine the event type.

Step 3. Identify the arguments related to the event (who, what, when, where, how).

Step 4. Assign an argument role for each argument (Agent, Participant, Time, Place, Quantity, Result, etc.).

Step 5. Output the final extracted events.

Return the results strictly in JSON format using the schema:

{{
  "events":[
    {{
      "event_id":"",
      "event_mention":"",
      "event_type":"",
      "trigger":"",
      "arguments":[
        {{
          "role":"",
          "argument":""
        }}
      ]
    }}
  ]
}}

Rules:
- event_id must be sequential (E1, E2, E3...)
- event_mention should be the sentence or phrase describing the event
- Output valid JSON only
- If no arguments exist for a role, omit the role. Do not invent information that does not appear in the text.

Text:
{text}
"""

url = "https://api.fireworks.ai/inference/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL_ID,
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0
}

response = requests.post(url, headers=headers, json=payload, timeout=120)
response.raise_for_status()

result = response.json()["choices"][0]["message"]["content"]

if isinstance(result, str):
    result = result.strip()

    if result.startswith("```json"):
        result = result[len("```json"):].strip()
    if result.startswith("```"):
        result = result[len("```"):].strip()
    if result.endswith("```"):
        result = result[:-3].strip()

    start = result.find("{")
    end = result.rfind("}")

    if start != -1 and end != -1 and end >= start:
        result = result[start:end + 1]

    result = json.loads(result)

output_data = {
    "doc_id": "event10_deep",
    "prompt_type": "CoT",
    "events": result["events"]
}

os.makedirs("result_llama", exist_ok=True)

with open("result_llama/e10_deep_cot.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

