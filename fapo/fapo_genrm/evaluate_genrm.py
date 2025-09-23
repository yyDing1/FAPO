import argparse
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset
from tabulate import tabulate
from transformers import AutoTokenizer

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_answer(solution_text: str):
    boxed_answer = last_boxed_only_string(solution_text)
    if boxed_answer:
        answer = remove_boxed(boxed_answer).strip()
        if answer.startswith("\\text{") and answer.endswith("}"):
            answer = answer[len("\\text{") : -1]
        return answer
    return None


def calculate_metric(data_source, samples):
    results = {}

    # ProcessBench Metrics
    error_data = [e for e in samples if e["extra_info"]["raw_info"]["label"] != -1]
    correct_data = [e for e in samples if e["extra_info"]["raw_info"]["label"] == -1]

    if "PRM" in data_source:
        err_acc = np.mean([e["extracted_final_answer"] != -1 for e in error_data])
        err_loc_acc = np.mean([e["extracted_final_answer"] == e["extra_info"]["raw_info"]["label"] for e in error_data])
        corr_acc = np.mean([e["extracted_final_answer"] == -1 for e in correct_data])
        pb_f1 = 2 * err_loc_acc * corr_acc / (err_loc_acc + corr_acc)
        metric = {}
        for subset in ["gsm8k", "math", "olympiadbench", "omnimath"]:
            subset_err_data = [e for e in error_data if e["data_source"].endswith(subset)]
            subset_corr_data = [e for e in correct_data if e["data_source"].endswith(subset)]
            subset_err_loc_acc = np.mean([e["extracted_final_answer"] == e["extra_info"]["raw_info"]["label"] for e in subset_err_data])
            subset_corr_acc = np.mean([e["extracted_final_answer"] == -1 for e in subset_corr_data])
            subset_pb_f1 = 2 * subset_err_loc_acc * subset_corr_acc / (subset_err_loc_acc + subset_corr_acc)
            metric[subset] = {
                "err_loc_acc": subset_err_loc_acc,
                "corr_acc": subset_corr_acc,
                "f1": subset_pb_f1,
            }
        metric["avg f1"] = np.mean([v["f1"] for v in metric.values()])
        from pprint import pprint
        pprint(metric)
    else:
        err_acc = np.mean([e["extracted_final_answer"] == False for e in error_data])
        err_loc_acc = None
        corr_acc = np.mean([e["extracted_final_answer"] == True for e in correct_data])
        pb_f1 = None
    results.update(
        {
            "Err (PB)": err_acc,
            "Err_Loc (PB)": err_loc_acc,
            "Corr (PB)": corr_acc,
            "F1 (PB)": pb_f1,
        }
    )

    # FalsePositiveBench Metrics
    positive_data = [e for e in samples if e["extra_info"]["raw_info"]["final_answer_correct"]]
    if "PRM" in data_source:
        pred_true_fp_cnt = len(
            [
                e
                for e in positive_data
                if e["extracted_final_answer"] != -1 and e["extra_info"]["raw_info"]["label"] != -1
            ]
        )
        pred_fp_cnt = len([e for e in positive_data if e["extracted_final_answer"] != -1])
        true_fp_cnt = len([e for e in positive_data if e["extra_info"]["raw_info"]["label"] != -1])
        precision = pred_true_fp_cnt / pred_fp_cnt
        recall = pred_true_fp_cnt / true_fp_cnt
        f1_score = 2 * precision * recall / (precision + recall)
        acc = np.mean([(e["extracted_final_answer"] == -1) == (e["extra_info"]["raw_info"]["label"] == -1) for e in positive_data])
    else:
        pred_true_fp_cnt = len(
            [
                e
                for e in positive_data
                if e["extracted_final_answer"] == False and e["extra_info"]["raw_info"]["label"] != -1
            ]
        )
        pred_fp_cnt = len([e for e in positive_data if e["extracted_final_answer"] == False])
        true_fp_cnt = len([e for e in positive_data if e["extra_info"]["raw_info"]["label"] != -1])
        precision = pred_true_fp_cnt / pred_fp_cnt
        recall = pred_true_fp_cnt / true_fp_cnt
        f1_score = 2 * precision * recall / (precision + recall)
        acc = np.mean([(e["extracted_final_answer"] == True) == (e["extra_info"]["raw_info"]["label"] == -1) for e in positive_data])
    results.update(
        {
            "Precision (FPB)": precision,
            "Recall (FPB)": recall,
            "F1 (FPB)": f1_score,
            "Acc": acc,
        }
    )

    avg_response_length = np.mean([e["total_response_length"] for e in samples])
    results.update(
        {
            "Avg Tokens": avg_response_length,
        }
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", default="~/data/processbench_orm_format")
    parser.add_argument("--tokenizer_path", default="Qwen/Qwen3-8B")
    args = parser.parse_args()

    eval_samples = load_dataset("parquet", data_files=args.parquet_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    def process_line(item):
        responses = [generation["response"] for generation in item["responses"]]
        # responses = [item["responses"]]
        extracted_answers = [extract_answer(response) for response in responses]
        valid_answers = [answer for answer in extracted_answers if answer is not None]
        maj_answers = Counter(valid_answers).most_common(1)[0][0] if len(valid_answers) > 0 else None
        try:
            maj_answers = int(eval(maj_answers))
        except:
            maj_answers = None
        item["extracted_final_answer"] = maj_answers
        responses_tot_length = sum([len(tokenizer.encode(response)) for response in responses])
        item["total_response_length"] = responses_tot_length
        return item

    eval_samples = eval_samples.map(process_line, num_proc=64)
    samples_bucket = defaultdict(list)
    for item in eval_samples:
        source_type = item["data_source"].split("/")[0]
        samples_bucket[source_type].append(item)

    table_headers = [
        "Prompt Type",
        "Count",
        "Err (PB)",
        "Err_Loc (PB)",
        "Corr (PB)",
        "F1 (PB)",
        "Precision (FPB)",
        "Recall (FPB)",
        "F1 (FPB)",
        "Acc",
        "Avg Tokens",
    ]
    float_fmt = (None, None) + (".2%",) * 8 + (".1f",)
    table_rows = []
    for source_type, samples in samples_bucket.items():
        metric = calculate_metric(source_type, samples)
        table_rows.append([source_type, len(samples)] + [metric[k] for k in table_headers[2:]])
    print(tabulate(table_rows, headers=table_headers, tablefmt="grid", floatfmt=float_fmt))
