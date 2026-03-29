import json
import argparse
import logging

logger = logging.getLogger(__name__)

args = argparse.ArgumentParser()
args.add_argument("--original_base_prompt_path", type=str, required=True)
args.add_argument("--summarizer_output_list_path", type=str, required=True)
args.add_argument("--new_base_prompt_path", type=str, required=True)
args.add_argument("--machine_config_path", type=str, default=None, help="Path to machine_config.yaml")
args.add_argument("--machine_config_preset", type=str, default="default", help="Preset name in machine_config.yaml")
args.add_argument("--log_file", type=str, default=None, help="Path to per-problem debug log file")
args = args.parse_args()

if args.log_file:
    from pathlib import Path
    _root = logging.getLogger()
    for _h in _root.handlers[:]:
        if isinstance(_h, logging.FileHandler):
            _h.close()
            _root.removeHandler(_h)
    _handler = logging.FileHandler(Path(args.log_file))
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    _root.addHandler(_handler)
    _root.setLevel(logging.INFO)

original_base_prompt_path = args.original_base_prompt_path
summarizer_output_list_path = args.summarizer_output_list_path
new_base_prompt_path = args.new_base_prompt_path

from accelopt.step_kernel_wrapper import load_machine_config, apply_prompt_substitutions
mc = load_machine_config(path=args.machine_config_path, preset=args.machine_config_preset)

with open(original_base_prompt_path, "r") as f:
    base_prompt = apply_prompt_substitutions(f.read(), mc)

with open(summarizer_output_list_path, "r") as f:
    summarizer_output_list = json.load(f)

additional_prompt = ""
for item in summarizer_output_list:
    if item["title"] == "**No optimization found**":
        continue
    additional_prompt += f"{item['summary']}\n\n"

if additional_prompt != "":
    base_prompt += "\n\n#Experiences\n"
    base_prompt += additional_prompt

with open(new_base_prompt_path, "w") as f:
    f.write(base_prompt)