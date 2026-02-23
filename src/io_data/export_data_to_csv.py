from pathlib import Path
import pandas as pd
from utils.logger_setup import logger


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
        logger.info("No file has been generated!")
    else:
        mooring_name = dataframe_insitu['platfID'].iloc[0]
        output_file = dir_output / f"{mooring_name}_{field}.csv"
        dataframe_insitu.to_csv(output_file)
        logger.info(f"{mooring_name}file has been exported!")


def export_dict_to_file(results_dict: dict, cfg):
    """Export 'df_val_sat' DataFrames from results_dict to CSV files.
    Parameters:
        results_dict: Nested dict containing DataFrames under 'df_val_sat'.
        cfg: Config dict with 'bias_correction_techniques' -> 'output_dir'.
    """
    output_dir = Path(cfg['bias_correction_techniques']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    counter = 0
    for method_name, method_data in results_dict.items():
        if "df_sat_val" not in method_data:
            logger.error("Error in dict extraction!")
            continue

        df_sat = method_data["df_sat_val"]
        sat_name = str(df_sat['platfID'].iloc[0])
        filepath = output_dir / f"{sat_name}_{method_name}.csv"
        df_sat.to_csv(filepath, index=False)
        if counter == 0:
            logger.info(
                "After applying the bias correction technique, satellite dataframe has been exported as .csv file!")
        counter = counter + 1
