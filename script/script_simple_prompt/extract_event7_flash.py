import json
from openai import OpenAI

client = OpenAI(api_key="")

with open("text/event7_flash_text.json", "r", encoding="utf-8") as f:
    data = json.load(f)

text = data["text"]

prompt = f"""
Extract all events from the following text.

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

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0,
    response_format={"type": "json_object"}
)

result = response.choices[0].message.content

if isinstance(result, str):
    result = json.loads(result)

output_data = {
    "doc_id": "event7_flash",
    "prompt_type": "simple",
    "events": result["events"]
}

if isinstance(result, str):
    result = json.loads(result)

with open("result_gpt/simple_prompt/e7_flash_simple.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)