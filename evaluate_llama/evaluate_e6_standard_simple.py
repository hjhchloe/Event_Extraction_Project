import sys
sys.argv = [sys.argv[0], r"answer/e6_standard_answer.json", r"result_llama/simple_prompt/e6_standard_simple.json", r"evaluation_llama/evaluation_e6_standard_simple.json"]
import json
import re
import sys
from sentence_transformers import SentenceTransformer, util

GOLD_FILE = sys.argv[1]
PRED_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]

SIMILARITY_THRESHOLD = 0.6

model = SentenceTransformer("all-MiniLM-L6-v2")

TYPE_MAP = {
    "Competition": "Competition",
    "Winning": "Competition",
    "Victory": "Competition",
    "Competition Result": "Competition",
    "CompetitionResult": "Competition",
    "Sports Competition": "Competition",
    "Sports Event": "Competition",
    "Competing": "Competition",
    "Competition Finish": "Competition",
    "Finishing": "Competition",
    "Medal Winning": "Competition",
    "Medal Standings": "Competition",
    "Medal Sweep": "Competition",
    "Award": "Competition",
    "Winning Move": "Competition",
    "Battling": "Competition",
    "Dominating": "Competition",
    "Dominance": "Competition",
    "Lead Change": "Competition",
    "Race Performance": "Competition",
    "Race Start": "Competition",
    "Relay": "Competition",
    "Finish": "Competition",
    "Sporting Achievement": "Competition",
    "Sports Achievement": "Competition",
    "Campaign Achievement": "Competition",
    "Achieving": "Competition",

    "Achievement": "Achievement",
    "Career Progression": "Achievement",
    "Participation": "Achievement",

    "Performance": "Performance",
    "Sports Performance": "Performance",
    "Sporting Performance": "Performance",
    "Action": "Performance",
    "Helping": "Performance",
    "Timing": "Performance",

    "Record": "Record",
    "Record Setting": "Record",
    "RecordSetting": "Record",

    "Communication": "Communication",
    "Statement": "Communication",
    "Speaking": "Communication",
    "Explanation": "Communication",
    "Announcement": "Communication",
    "Declaration": "Communication",
    "Interview": "Communication",
    "Opinion": "Communication",
    "Acknowledgment": "Communication",

    "Ceremony": "Ceremony",
    "Opening Ceremony": "Ceremony",
    "Ignition": "Ceremony",

    "Accident": "Incident",
    "Failure": "Incident",
    "Loss of Control": "Incident",
    "Spinning": "Incident",
    "Injury": "Incident",
    "Medical": "Incident",
    "Medical Transport": "Incident",
    "Emergency Response": "Incident",
    "Assistance": "Incident",
    "Evacuation": "Incident",
    "Rescue": "Incident",
    "Interruption": "Incident",
    "Health": "Incident",
    "Recovery": "Incident",

    "Reaction": "Reaction",
    "Celebration": "Reaction",
    "Emotional": "Reaction",
    "Emotional Reaction": "Reaction",
    "Disappointment": "Reaction",
    "Screaming": "Reaction",
    "Adversity": "Reaction",

    "Expectation": "Status",
    "Status": "Status",
    "Preparation": "Status",
    "Selection": "Status",
    "Scheduled Event": "Status",
    "Scheduled": "Status",
    "Occurrence": "Status",
    "Arrival": "Status",
    "Return": "Status",
    "Returning": "Status",
    "Entering": "Status",
    "Withdrawal": "Status",
    "Quitting": "Status",
    "Taking a Break": "Status",
    "ParticipantAddition": "Status",
    "Intention": "Status",
    "Admission": "Status",

    "Social": "Social",
    "Personal Interaction": "Social",
    "Meeting": "Social",
    "Reunion": "Social",
    "Visit": "Social",

    "Creation": "Creation",
    "Invention": "Creation",
    "Innovation": "Creation",

    "Career": "Personal",
    "Career Ending": "Personal",
    "Personal": "Personal",
    "Death": "Personal",

    "Funding": "Policy",
    "Policy": "Policy",
    "Security": "Policy",
    "Searching": "Policy"
}

ROLE_MAP = {
    "Agent": "participant",
    "agent": "participant",
    "Participant": "participant",
    "participant": "participant",
    "Participants": "participant",
    "participants": "participant",
    "person": "participant",
    "athlete": "participant",
    "performer": "participant",
    "winner": "participant",
    "Winner": "participant",
    "scorer": "participant",
    "Achiever": "participant",
    "achiever": "participant",
    "competitor": "participant",
    "competitor1": "participant",
    "competitor2": "participant",
    "competitors": "participant",
    "Country": "participant",
    "country": "participant",
    "Team": "participant",
    "team": "participant",
    "organization": "participant",
    "teammate": "participant",
    "selected": "participant",
    "withdrawn_participant": "participant",
    "runner-up": "participant",
    "runner_up": "participant",
    "Countries": "participant",
    "Innovator": "participant",
    "innovator": "participant",
    "Inventor": "participant",

    "Result": "result",
    "result": "result",
    "outcome": "result",
    "achievement": "result",
    "Achievement": "result",
    "Consequence": "result",
    "previous_award": "result",
    "comparison": "result",
    "Comparison": "result",
    "injury": "result",
    "Invention": "result",
    "innovation": "result",

    "Time": "time",
    "time": "time",
    "Date": "time",
    "date": "time",
    "Duration": "time",
    "duration": "time",

    "Place": "location",
    "place": "location",
    "location": "location",
    "Location": "location",
    "Destination": "location",
    "destination": "location",
    "context": "location",

    "Event": "event",
    "event": "event",
    "events": "event",
    "Current Event": "event",
    "current_event": "event",
    "Initial Event": "event",
    "initial_event": "event",
    "previous_event": "event",
    "competition": "event",
    "program": "event",
    "run": "event",
    "Races": "event",
    "races": "event",
    "stage": "event",
    "title": "event",
    "Title": "event",

    "Quantity": "quantity",
    "quantity": "quantity",
    "amount": "quantity",
    "percentage": "quantity",
    "margin": "quantity",
    "Distance": "quantity",
    "distance": "quantity",
    "number_of_races": "quantity",
    "Medal Count": "quantity",

    "Score": "score",
    "score": "score",

    "Position": "position",
    "position": "position",
    "ranking": "position",

    "Medal": "medal",
    "medal": "medal",
    "Medals": "medal",
    "Gold Medals": "medal",
    "Silver Medals": "medal",
    "Bronze Medals": "medal",
    "award": "medal",
    "Award": "medal",

    "Recipient": "recipient",
    "recipient": "recipient",
    "Beneficiary": "recipient",
    "Target": "recipient",
    "target": "recipient",

    "Opponent": "opponent",
    "opponent": "opponent",
    "Loser": "opponent",
    "loser": "opponent",

    "Reason": "cause",
    "reason": "cause",
    "Cause": "cause",
    "condition": "cause",
    "Condition": "cause",
    "Obstacle": "cause",

    "Action": "action",
    "action": "action",
    "method": "action",
    "Usage": "action",
    "usage": "action",
    "Purpose": "action",
    "purpose": "action",

    "Speaker": "speaker",
    "speaker": "speaker",
    "author": "speaker",

    "Audience": "audience",
    "audience": "audience",

    "record": "record",
    "record_holder": "record",
    "previous_record": "record",
    "previous_record_holder": "record",

    "content": "content",
    "Content": "content",
    "news": "content",
    "Feeling": "content",
    "emotion": "content",
    "anthem": "content",
    "song": "content",

    "Instrument": "instrument",
    "policy": "content",
    "type": "content"
}


def normalize_type(t):
    return TYPE_MAP.get(t, t)


def normalize_role(r):
    return ROLE_MAP.get(r, r)


def normalize_text(text):
    return str(text).lower().strip()


def extract_number(text):
    nums = re.findall(r'\d+', text)
    if nums:
        return nums[0]
    return None


def argument_match(gold_arg, pred_arg):
    g = normalize_text(gold_arg)
    p = normalize_text(pred_arg)

    if g in p or p in g:
        return True

    g_num = extract_number(g)
    p_num = extract_number(p)

    if g_num and p_num and g_num == p_num:
        return True

    emb = model.encode([g, p])
    sim = util.cos_sim(emb[0], emb[1]).item()
    return sim > 0.7


def compute_prf(correct, pred_total, gold_total):
    precision = correct / pred_total if pred_total > 0 else 0
    recall = correct / gold_total if gold_total > 0 else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)


with open(GOLD_FILE, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

with open(PRED_FILE, "r", encoding="utf-8") as f:
    pred_data = json.load(f)

gold_events = gold_data.get("events", [])
pred_events = pred_data.get("events", [])

gold_mentions = [e.get("event_mention", "") for e in gold_events]
pred_mentions = [e.get("event_mention", "") for e in pred_events]

gold_embeddings = model.encode(gold_mentions)
pred_embeddings = model.encode(pred_mentions)

correct_TI = 0
correct_TC = 0
correct_AI = 0
correct_AC = 0

matched_pred_indices = set()
matched_event_pairs = []

total_gold_triggers = len(gold_events)
total_pred_triggers = len(pred_events)

total_gold_args = sum(len(e.get("arguments", [])) for e in gold_events)
total_pred_args = sum(len(e.get("arguments", [])) for e in pred_events)

for g_idx, g_ev in enumerate(gold_events):
    best_similarity = -1
    best_match_idx = -1

    for p_idx, p_ev in enumerate(pred_events):
        if p_idx in matched_pred_indices:
            continue

        similarity = util.cos_sim(gold_embeddings[g_idx], pred_embeddings[p_idx]).item()

        if similarity > best_similarity:
            best_similarity = similarity
            best_match_idx = p_idx

    if best_similarity < SIMILARITY_THRESHOLD:
        continue

    matched_pred_indices.add(best_match_idx)
    matched_p_ev = pred_events[best_match_idx]

    gold_trigger = g_ev.get("trigger", "")
    pred_trigger = matched_p_ev.get("trigger", "")

    gold_type_raw = g_ev.get("event_type", "")
    pred_type_raw = matched_p_ev.get("event_type", "")

    gold_type = normalize_type(gold_type_raw)
    pred_type = normalize_type(pred_type_raw)

    trigger_match = normalize_text(gold_trigger) == normalize_text(pred_trigger)
    type_match = gold_type == pred_type

    if trigger_match:
        correct_TI += 1

    if trigger_match and type_match:
        correct_TC += 1

    g_args = g_ev.get("arguments", [])
    p_args = matched_p_ev.get("arguments", [])

    matched_pred_args = set()
    arg_match_count = 0

    for ga in g_args:
        g_role = normalize_role(ga.get("role", ""))
        g_arg = ga.get("argument", "")

        for p_idx_arg, pa in enumerate(p_args):
            if p_idx_arg in matched_pred_args:
                continue

            p_role = normalize_role(pa.get("role", ""))
            p_arg = pa.get("argument", "")

            if argument_match(g_arg, p_arg):
                correct_AI += 1
                arg_match_count += 1

                if g_role == p_role:
                    correct_AC += 1

                matched_pred_args.add(p_idx_arg)
                break

    matched_event_pairs.append({
        "gold": {
            "event_id": g_ev.get("event_id", ""),
            "event_mention": g_ev.get("event_mention", ""),
            "event_type": gold_type_raw,
            "normalized_event_type": gold_type,
            "trigger": gold_trigger,
            "arguments": g_args,
        },
        "prediction": {
            "event_id": matched_p_ev.get("event_id", ""),
            "event_mention": matched_p_ev.get("event_mention", ""),
            "event_type": pred_type_raw,
            "normalized_event_type": pred_type,
            "trigger": pred_trigger,
            "arguments": p_args,
        },
        "similarity": round(best_similarity, 4),
        "trigger_match": trigger_match,
        "type_match": type_match,
        "argument_matches": arg_match_count,
    })

TI_p, TI_r, TI_f1 = compute_prf(correct_TI, total_pred_triggers, total_gold_triggers)
TC_p, TC_r, TC_f1 = compute_prf(correct_TC, total_pred_triggers, total_gold_triggers)
AI_p, AI_r, AI_f1 = compute_prf(correct_AI, total_pred_args, total_gold_args)
AC_p, AC_r, AC_f1 = compute_prf(correct_AC, total_pred_args, total_gold_args)

final_result = {
    "summary": {
        "gold_events": total_gold_triggers,
        "pred_events": total_pred_triggers,
        "gold_arguments": total_gold_args,
        "pred_arguments": total_pred_args,
    },
    "trigger_identification": {"precision": TI_p, "recall": TI_r, "f1": TI_f1},
    "trigger_classification": {"precision": TC_p, "recall": TC_r, "f1": TC_f1},
    "argument_identification": {"precision": AI_p, "recall": AI_r, "f1": AI_f1},
    "argument_classification": {"precision": AC_p, "recall": AC_r, "f1": AC_f1},
    "matched_events": matched_event_pairs,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_result, f, indent=4, ensure_ascii=False)
