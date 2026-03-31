import os
import json

BASE_DIR = "evaluation"
SIMPLE_DIR = os.path.join(BASE_DIR, "evaluation_simple")
COT_DIR = os.path.join(BASE_DIR, "evaluation_CoT")
OUTPUT_FILE = os.path.join(BASE_DIR, "overall_average_scores.json")

TASK_NAMES = [
    "trigger_identification",
    "trigger_classification",
    "argument_identification",
    "argument_classification"
]

def make_empty_scores():
    result = {}
    for task in TASK_NAMES:
        result[task] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    return result

def add_scores(total_scores, file_data):
    for task in TASK_NAMES:
        total_scores[task]["precision"] += file_data[task]["precision"]
        total_scores[task]["recall"] += file_data[task]["recall"]
        total_scores[task]["f1"] += file_data[task]["f1"]

def divide_scores(total_scores, count):
    result = make_empty_scores()

    if count == 0:
        return result

    for task in TASK_NAMES:
        result[task]["precision"] = round(total_scores[task]["precision"] / count, 4)
        result[task]["recall"] = round(total_scores[task]["recall"] / count, 4)
        result[task]["f1"] = round(total_scores[task]["f1"] / count, 4)

    return result

def process_folder(folder_path):
    total_scores = make_empty_scores()
    file_count = 0

    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        add_scores(total_scores, data)
        file_count += 1

    average_scores = divide_scores(total_scores, file_count)

    return {
        "file_count": file_count,
        "average_scores": average_scores
    }

def main():
    simple_result = process_folder(SIMPLE_DIR)
    cot_result = process_folder(COT_DIR)

    final_result = {
        "simple_prompt": simple_result,
        "CoT_prompt": cot_result
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    print("saved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()