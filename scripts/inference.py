import argparse
import json
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from allennlp.data import Vocabulary

from updown.config import Config
from updown.data.datasets import EvaluationDataset, EvaluationDatasetWithConstraints
from updown.models import UpDownCaptioner
from updown.types import Prediction
from updown.utils.evalai import NocapsEvaluator
from updown.utils.constraints import add_constraint_words_to_vocabulary


if __name__ == "__main__":
    evaluator = NocapsEvaluator("val")
    evaluation_metrics = evaluator.evaluate(predictions)

    print(f"Evaluation metrics for checkpoint {_A.checkpoint_path}:")
    for metric_name in evaluation_metrics:
        print(f"\t{metric_name}:")
        for domain in evaluation_metrics[metric_name]:
            print(f"\t\t{domain}:", evaluation_metrics[metric_name][domain])