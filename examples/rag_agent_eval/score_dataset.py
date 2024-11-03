# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os

import fire

from llama_stack_client import LlamaStackClient
from termcolor import cprint
from .util import data_url_from_file


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
        rows_in_page=3,
        page_token=None,
        filter_condition=None,
    )
    cprint(rows_paginated, "yellow")

    # # check scoring functions available
    score_fn_list = client.scoring_functions.list()
    cprint([x.identifier for x in score_fn_list], "green")

    score_rows = client.scoring.score(
        input_rows=rows_paginated.rows,
        scoring_functions=["braintrust::answer-correctness"],
    )
    cprint(f"Score Rows: {score_rows}", "red")



def main(host: str, port: int, file_path: str):
    asyncio.run(run_main(host, port, file_path))


if __name__ == "__main__":
    fire.Fire(main)
