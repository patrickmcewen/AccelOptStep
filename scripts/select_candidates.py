import json
import pandas as pd
import os
from accelopt.utils import init_service_name

def get_branch_id(plan_name):
    return plan_name.split("_")[1]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor_results_path", type=str, required=True)
    parser.add_argument("--output_base_path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=1)
    args = parser.parse_args()
    with open(args.executor_results_path, "r") as f:
        executor_results = json.load(f)

    candidates = {}
    for record in executor_results["executor_results"]:
        service_name = record["service_name"]
        case_name = record["case_name"]
        for k, v in record.items():
            if k in ["service_name", "case_name"]: 
                continue
            if not "baseline" in v.keys(): # "error": "No plan found"
                continue
            branch_id = get_branch_id(k)
            if not "speedup" in v.keys() or v["speedup"] is None:
                baseline_meta = json.loads(v.get("baseline_metadata", "{}"))
                baseline_latency = baseline_meta.get("cycles", float("inf"))
                candidates.setdefault(case_name, {}).setdefault((service_name, branch_id), []).append({
                    "body": v["baseline"],
                    "spec_code": v["spec_code"],
                    "latency": baseline_latency,
                    "profile": v.get("baseline_metadata", "{}"),
                    "problem": v["problem"],
                    "values": v["values"],
                    "old_service_name": service_name,
                    "plan_id": k,
                    "priority": float("inf"),
                })
            else:
                cycles = v["cycles"]
                candidates.setdefault(case_name, {}).setdefault((service_name, branch_id), []).append({
                    "body": v["body"],
                    "spec_code": v["spec_code"],
                    "latency": cycles,
                    "profile": v.get("kernel_metadata", "{}"),
                    "problem": v["problem"],
                    "values": v["values"],
                    "old_service_name": service_name,
                    "plan_id": k,
                    "priority": cycles,
                })
            
    # First select the best representative for each (service_name, branch_id)
    unique_candidates = {}
    for case_name, service_name_branch_id_candidates in candidates.items():
        for (service_name, branch_id), candidates_items in service_name_branch_id_candidates.items():
            best_candidate = min(candidates_items, key=lambda x: x["priority"])
            unique_candidates.setdefault(case_name, {})[(service_name, branch_id)] = best_candidate
    # Then select the topk candidates for each case
    topk_candidates = {}
    for case_name, service_name_branch_id_candidates in unique_candidates.items():
        sorted_keys = sorted(service_name_branch_id_candidates.keys(), key=lambda x: service_name_branch_id_candidates[x]["priority"])
        topk_candidates[case_name] = [service_name_branch_id_candidates[k] for k in sorted_keys[:args.topk]]
    # For candidates, store the body and spec_code into new .py files and store the service_name,task,kernel to a new csv file
    output_base_path = args.output_base_path
    os.makedirs(output_base_path, exist_ok=True)
    output_dict = []
    for case_name, candidates_items in topk_candidates.items():
        for item in candidates_items:
            new_service_name = init_service_name(case_name)
            body_path = f"{output_base_path}/{new_service_name}_{item['old_service_name']}_{item['plan_id']}_body.py"
            numpy_path = f"{output_base_path}/{new_service_name}_{item['old_service_name']}_{item['plan_id']}_numpy.py"
            with open(body_path, "w") as f:
                body_code = item["body"]
                f.write(body_code)
            with open(numpy_path, "w") as f:
                f.write(item["spec_code"])
            output_dict.append({
                "service_name": new_service_name,
                "task": numpy_path,
                "kernel": body_path,
                "problem": item["problem"],
                "values": item["values"],
                "plan_id": item["plan_id"],
                "case_name": case_name,
                "profile": item["profile"],
            })
    output_path = f"{output_base_path}/candidates.csv"
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(output_path, index=False)