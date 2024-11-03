# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pandas as pd
import os
import fire
from termcolor import cprint


def transform_input_query_to_message_list(row):
    msg = {
        "role": "user",
        "content": row.input_query,
    }
    return [msg]


def preprocess_dataset(input_file_path, output_dir):
    """
    Preprocess Excel file and save as CSV

    Args:
        input_file_path (str): Path to the input Excel file
        output_file_path (str): Path where the CSV will be saved
    """
    try:
        # Read the Excel file
        df = pd.read_excel(input_file_path)
        
        # Basic preprocessing steps
        # Remove any empty rows
        df = df.dropna(how='all')
        
        # Remove any duplicate rows
        df = df.drop_duplicates()
        
        # Reset the index after dropping rows
        df = df.reset_index(drop=True)
        
        # Create output directory if it doesn't exist
        input_file_name = os.path.basename(input_file_path)
        output_file_path = os.path.join(output_dir, f"processed_{input_file_name}.csv")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Save to CSV
        cprint(df.columns, "green")
        df.rename(columns={"query": "input_query"}, inplace=True)
        df["chat_completion_input"] = df.apply(
            transform_input_query_to_message_list,
            axis=1,
        )

        # Keep only the relevant columns
        df = df[["input_query", "chat_completion_input", "expected_answer", "generated_answer", "correctness_llm", "correctness_human"]]
        cprint(df.columns, "blue")

        df.to_csv(output_file_path, index=False)
        print(f"Successfully processed and saved to {output_file_path}")

    except Exception as e:
        print(f"Error processing file: {str(e)}")


def main(input_file: str, output_dir: str):
    preprocess_dataset(input_file, output_dir)

if __name__ == "__main__":
    fire.Fire(main)
