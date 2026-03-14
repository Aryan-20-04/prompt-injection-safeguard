# bench/runner.py — needs to move here (not scripts/) and be completed:
import argparse
from tqdm import tqdm
from bench.evaluator import Evaluator
from bench.utils.reproducibility import capture_snapshot
from bench.models.registry import load_from_yaml, get as get_model
import yaml, json
from pathlib import Path

class BenchmarkRunner:
    def __init__(self):
        self.evaluator = Evaluator()
    def run_from_config_path(self, config_path: str, output_dir: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        output_dir = output_dir or config["run"]["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        snapshot = capture_snapshot(config, seed=config["run"].get("seed", 42))
        (Path(output_dir)/"run_metadata.json").write_text(json.dumps(snapshot, indent=2))
        # ... dataset loading and inference loop

def cli_entry():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    BenchmarkRunner().run_from_config_path(args.config, args.output)

if __name__ == "__main__":
    cli_entry()