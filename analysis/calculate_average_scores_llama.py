import os
import json

BASE_DIR = "evaluation_llama"
SIMPLE_DIR = os.path.join(BASE_DIR, "evaluation_simple")
COT_DIR = os.path.join(BASE_DIR, "evaluation_CoT")
OUTPUT_FILE = os.path.join(BASE_DIR, "llama_average_scores_summary.json")

TASK_NAMES = [
    "trigger_identification",
    "trigger_classification",
    "argument_identification",
    "argument_classification"
]

CONTEXT_TYPES = ["flash", "standard", "deep"]


def make_empty_scores():
    result = {}
    for task in TASK_NAMES:
        result[task] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    return result


def add_one_file_scores(total_scores, file_data):
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


def make_group_result():
    return {
        "count": 0,
        "total": make_empty_scores()
    }


def get_context_type(file_name):
    for context in CONTEXT_TYPES:
        if "_" + context + "_" in file_name:
            return context
    return None


def process_folder(folder_path):
    overall = make_group_result()

    by_context = {}
    for context in CONTEXT_TYPES:
        by_context[context] = make_group_result()

    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        add_one_file_scores(overall["total"], data)
        overall["count"] += 1

        context = get_context_type(file_name)
        if context is not None:
            add_one_file_scores(by_context[context]["total"], data)
            by_context[context]["count"] += 1

    overall_avg = divide_scores(overall["total"], overall["count"])

    context_avg = {}
    for context in CONTEXT_TYPES:
        context_avg[context] = divide_scores(
            by_context[context]["total"],
            by_context[context]["count"]
        )

    return overall, overall_avg, by_context, context_avg


def print_scores(title, count, scores):
    print("=" * 70)
    print(title)
    print("file count:", count)
    print()

    for task in TASK_NAMES:
        print(task)
        print("  precision:", scores[task]["precision"])
        print("  recall   :", scores[task]["recall"])
        print("  f1       :", scores[task]["f1"])
        print()


def main():
    simple_overall, simple_overall_avg, simple_by_context, simple_context_avg = process_folder(SIMPLE_DIR)
    cot_overall, cot_overall_avg, cot_by_context, cot_context_avg = process_folder(COT_DIR)

    print_scores("Simple Prompt Overall Average", simple_overall["count"], simple_overall_avg)
    print_scores("CoT Prompt Overall Average", cot_overall["count"], cot_overall_avg)

    for context in CONTEXT_TYPES:
        print_scores(
            "Simple Prompt - " + context,
            simple_by_context[context]["count"],
            simple_context_avg[context]
        )

    for context in CONTEXT_TYPES:
        print_scores(
            "CoT Prompt - " + context,
            cot_by_context[context]["count"],
            cot_context_avg[context]
        )

    final_result = {
        "simple_prompt": {
            "overall": {
                "file_count": simple_overall["count"],
                "average_scores": simple_overall_avg
            },
            "by_context": {}
        },
        "CoT_prompt": {
            "overall": {
                "file_count": cot_overall["count"],
                "average_scores": cot_overall_avg
            },
            "by_context": {}
        }
    }

    for context in CONTEXT_TYPES:
        final_result["simple_prompt"]["by_context"][context] = {
            "file_count": simple_by_context[context]["count"],
            "average_scores": simple_context_avg[context]
        }

    for context in CONTEXT_TYPES:
        final_result["CoT_prompt"]["by_context"][context] = {
            "file_count": cot_by_context[context]["count"],
            "average_scores": cot_context_avg[context]
        }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    print("=" * 70)
    print("Summary saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()