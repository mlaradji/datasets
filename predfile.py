#! /usr/bin/env python

# example run (the only one tested)
# python predfile.py "textattack/albert-base-v2-rotten-tomatoes" "rotten_tomatoes" 2>err.log | less

import json
import logging
import os
import pickle
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from hashlib import sha256
from typing import Dict

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import datasets


logger = logging.getLogger(__name__)


def build_metadata(model, dataset: datasets.DatasetDict) -> Dict:
    model_md = dict(
        name=model.name_or_path,
        config=vars(model.config),
        labelmap=model.config.id2label,
    )

    # XXX: no info on DatasetDict, have to fetch one of the slice
    dslice = list(dataset.values())[0]
    dataset_md = dict(info=asdict(dslice.info), labelmap=dslice.features["label"]._str2int)

    env_md = dict(osname=os.name, command=sys.argv)

    return dict(model=model_md, dataset=dataset_md, env=env_md)


def run():
    ap = ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("dataset_id")
    args = ap.parse_args()

    dataset = datasets.load_dataset(args.dataset_id)

    if "test" not in dataset:
        raise ValueError(f"{dataset} has no 'test' split")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id)
    label2id = {v: k for k, v in model.config.id2label.items()}  # ouch

    # Hopefully we don't manage this and go through the `api-inference` endpoint in the future.
    pp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    predictions = []
    logger.debug(f"iterating over test set, {len(dataset['test'])} examples")
    for example in dataset["test"]:
        ex_input = example["text"]  # this is going to be __fun__
        res_dict, *_ = pp(ex_input)
        result = res_dict["label"]  # <-- this is manageable, just need per-task/pipeline adaptations
        prediction = label2id[result]

        serialized = pickle.dumps(example)
        hashed = sha256(serialized).hexdigest()
        predictions.append((hashed, prediction))
        break

    output = dict(metadata=build_metadata(model, dataset), predictions=predictions)
    serialized_output = json.dumps(output, indent=2)
    print(serialized_output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    run()
