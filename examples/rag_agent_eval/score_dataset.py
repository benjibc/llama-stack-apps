# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
import pandas as pd

import fire

from llama_stack_client import LlamaStackClient
from termcolor import cprint
from .util import data_url_from_file
from tqdm import tqdm

async def run_main(host: str, port: int, file_path: str):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    providers = client.providers.list()
    dataset_url = data_url_from_file(file_path)
    filename_id = os.path.basename(file_path)
    print(filename_id)

    client.datasets.register(
        dataset_def={
            "identifier": filename_id,
            "provider_id": providers["datasetio"][0].provider_id,
            "url": {"uri": dataset_url},
            "dataset_schema": {
                "generated_answer": {"type": "string"},
                "expected_answer": {"type": "string"},
                "input_query": {"type": "string"},
            },
        }
    )

    datasets_list_response = client.datasets.list()
    cprint([x.identifier for x in datasets_list_response], "cyan")

    # test scoring with individual rows
    rows_paginated = client.datasetio.get_rows_paginated(
        dataset_id=filename_id,
        rows_in_page=-1,
        page_token=None,
        filter_condition=None,
    )

    # check scoring functions available
    score_fn_list = client.scoring_functions.list()
    cprint([x.identifier for x in score_fn_list], "green")

    # We use 2 LLM As Judge scoring functions for the RAG agent evaluation:
    # - braintrust::answer-correctness using Braintrust's answer-correctness scoring function
    # - meta-reference::llm_as_judge_405b_correctness using Meta's LLM as Judge scoring function with 405B model
    for i in tqdm(range(len(rows_paginated.rows))):
        row = rows_paginated.rows[i]
        score_rows = client.scoring.score(
            input_rows=[row],
            scoring_functions=[
                "braintrust::answer-correctness",
                "meta-reference::llm_as_judge_405b_correctness",
            ],
        )
        cprint(f"Score Rows: {score_rows}", "red")
        break


def main(host: str, port: int, file_path: str):
    asyncio.run(run_main(host, port, file_path))


if __name__ == "__main__":
    fire.Fire(main)
