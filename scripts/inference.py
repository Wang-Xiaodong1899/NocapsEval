import argparse
import json
from typing import List
import os
import sys
sys.path.append('/workspace/NocapsEval')

import numpy as np
from tqdm import tqdm

from updown.config import Config
from updown.data.datasets import EvaluationDataset, EvaluationDatasetWithConstraints
from updown.models import UpDownCaptioner
from updown.types import Prediction
from updown.utils.evalai import NocapsEvaluator
from updown.utils.constraints import add_constraint_words_to_vocabulary


if __name__ == "__main__":
    with open("/workspace/nocap_eval_all_tk128_clean.json", "r") as file:  
        predictions = json.load(file)
    print(f'loaded data')
    evaluator = NocapsEvaluator("val")
    evaluation_metrics = evaluator.evaluate(predictions)

    print(f"Evaluation metrics for checkpoint {_A.checkpoint_path}:")
    for metric_name in evaluation_metrics:
        print(f"\t{metric_name}:")
        for domain in evaluation_metrics[metric_name]:
            print(f"\t\t{domain}:", evaluation_metrics[metric_name][domain])
