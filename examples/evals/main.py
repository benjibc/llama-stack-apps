# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree
import json
from pathlib import Path

import fire
import pandas


def main(file_dir=""):
    # for p in Path(file_dir).rglob("*.xlsx"):
    #     with open(p, "rb") as f:
    #         df = pandas.read_excel(f)
    #         df = df.rename(
    #             columns={
    #                 "query": "input_query",
    #             }
    #         )
    #         df = df[
    #             [
    #                 "input_query",
    #                 "generated_answer",
    #                 "expected_answer",
    #                 "correctness_llm",
    #             ]
    #         ]
    #         # save as jsonl
    #         jsonl_data = df.to_json(orient="records", lines=True)
    #         print(len(jsonl_data), type(jsonl_data))
    #         output_path = p.with_suffix(".jsonl")
    #         print(f"Saving to {output_path}")
    #         with open(output_path, "w") as out_f:
    #             out_f.write(jsonl_data)
    #     break

    # read in jsonl format
    for p in Path(file_dir).rglob("*.jsonl"):
        with open(p, "r") as f:
            data = [json.loads(line) for line in f]


if __name__ == "__main__":
    fire.Fire(main)
