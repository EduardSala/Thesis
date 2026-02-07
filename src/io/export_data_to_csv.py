import pandas as pd
import os
from pathlib import Path

def export_dataframe_to_file(dataframe_insitu,field,dir_output):

    dir_output = Path(dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)

    if (len(dataframe_insitu) > 0):
        mooring_name = dataframe_insitu['platfID'][0]
        output_file = dir_output / f"{mooring_name}_{field}.csv"
        dataframe_insitu.to_csv(output_file)
    else:
        print("No file has been generated!")














