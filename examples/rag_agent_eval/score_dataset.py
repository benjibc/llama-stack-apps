# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
import pandas as pd
import json
import fire

from llama_stack_client import LlamaStackClient
from termcolor import cprint
from .util import data_url_from_file
from tqdm import tqdm
from pathlib import Path


async def run_main(host: str, port: int, file_path: str, include_original_score: bool):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )
    original_df = pd.read_csv(file_path)

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
    scoring_functions = [
        "braintrust::answer-correctness",
        "meta-reference::llm_as_judge_405b_correctness",
    ]
    output_res = {
        "input_query": [],
        "generated_answer": [],
        "expected_answer": [],
    }
    for x in scoring_functions:
        output_res[x] = []
    
    if include_original_score:
        output_res["original_correctness_llm"] = []
        output_res["original_correctness_human"] = []

    for i in tqdm(range(len(rows_paginated.rows))):
        row = rows_paginated.rows[i]
        score_rows = client.scoring.score(
            input_rows=[row],
            scoring_functions=[
                "braintrust::answer-correctness",
                "meta-reference::llm_as_judge_405b_correctness",
            ],
        )
        # cprint(f"Score Rows: {score_rows}", "red")
        output_res["input_query"].append(row["input_query"])
        output_res["expected_answer"].append(row["expected_answer"])
        output_res["generated_answer"].append(row["generated_answer"])
        for scoring_fn in scoring_functions:
            output_res[scoring_fn].append(score_rows.results[scoring_fn].score_rows[0])

        if include_original_score:
            output_res["original_correctness_llm"].append(original_df.iloc[i]["correctness_llm"])
            output_res["original_correctness_human"].append(original_df.iloc[i]["correctness_human"])

    # dump results
    save_path = "./rag_scored/" + os.path.basename(file_path).replace(".csv", "-scored.xlsx")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pd.DataFrame(output_res).to_excel(f, index=False)
    cprint(f"Saved to {save_path}", "green")

def main(host: str, port: int, file_path: str, include_original_score: bool = True):
    asyncio.run(run_main(host, port, file_path, include_original_score))


if __name__ == "__main__":
    fire.Fire(main)
