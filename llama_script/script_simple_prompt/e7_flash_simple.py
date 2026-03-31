import json
import requests
import os

API_KEY = os.getenv("FIREWORKS_API_KEY")

MODEL_ID = "accounts/fireworks/models/llama-v3p3-70b-instruct"

with open("/Users/hjhchloe/Documents/GitHub/Event_Extraction_Project/text/event7_flash_text.json", "r", encoding="utf-8") as f:
    data = json.load(f)

text = data["text"]

prompt = f"""Extract all events from the following text.

Definition of event:
An event is a specific occurrence that happens at a particular time or place and involves one or more participants.

For each event identify:

- event_id (E1, E2, E3...)
- event_mention
- event_type
- trigger
- arguments and their roles

Return the results strictly in JSON format using this schema:

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
    "doc_id": "event7_flash",
    "prompt_type": "simple",
    "events": result["events"]
}

os.makedirs("result_llama", exist_ok=True)

with open("result_llama/e7_flash_simple.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

