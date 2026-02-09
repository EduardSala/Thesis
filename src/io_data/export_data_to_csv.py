from pathlib import Path
import pandas as pd


def export_dataframe_to_file(dataframe_insitu: pd.DataFrame, field: str, dir_output: str | Path) -> None:
    """
    Export the in-situ DataFrame to a CSV file in the specified output directory.
    The filename is constructed using the mooring name and the field name.
    Args:
        dataframe_insitu: DataFrame containing the in-situ data to be exported.
        field: Field name to be included in the output filename. Wave or wind.
        dir_output: Directory where the output file will be saved. Can be a string or a Path object.

    Returns:
        None
    """
    dir_output = Path(dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)

    # Guard against None or empty DataFrame
    if dataframe_insitu is None or len(dataframe_insitu) == 0:
        print("No file has been generated!")
        return

    mooring_name = dataframe_insitu['platfID'].iloc[0]
    output_file = dir_output / f"{mooring_name}_{field}.csv"
    dataframe_insitu.to_csv(output_file)
