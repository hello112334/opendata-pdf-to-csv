import pandas as pd
import os


def main(i):
    opendata_files = os.listdir(f"./data_files/yakkyoku_{i}")
    opendata_file = opendata_files[0]

    file_path = f"./data_files/yakkyoku_{i}/{opendata_file}"
    df = pd.read_excel(file_path)

    output_file_path = f"./output_files/yakkyoku_{i}.csv"
    df.to_csv(output_file_path)


main(1)
