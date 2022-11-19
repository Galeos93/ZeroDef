import argparse

import os

my_parser = argparse.ArgumentParser(description="List the content of a folder")

parser_help = """
The following command can be executed: train_solution and obtain_predictions

"""

my_parser.add_argument(
    "Command",
    metavar="command",
    type=str,
    help=parser_help,
    choices={"train_solution", "obtain_predictions"},
)

args = my_parser.parse_args()

command = args.Command

if command == "train_solution":
    os.system("make download-data")
    os.system("make create-extended-dataset")
    os.system(
        "PYTHONPATH=. python zero_deforestation/train.py "
        "--c zero_deforestation/final_solution_config.json"
    )
elif command == "obtain_predictions":
    os.system("make download-data")
    os.system("make create-extended-dataset")
    os.system(
        "PYTHONPATH=. python zero_deforestation/test.py "
        "--c zero_deforestation/inference_config.json "
        "-r saved/models/FinalSolution/1119_205842/checkpoint-epoch22.pth"
    )
else:
    ValueError("This command is not available.")
