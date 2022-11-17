from pathlib import Path
from typing import Union

import pandas as pd

from sklearn.model_selection import train_test_split


def create_dataframe_with_image_and_mask_filepaths(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    img_prefix: str,
    mask_prefix: str,
    file_extension: str,
) -> pd.DataFrame:
    """Create a dataframe with image and mask filepaths.

    Args:
        image_dir (Union[str, Path]): Path to the directory containing the images.
        mask_dir (Union[str, Path]): Path to the directory containing the masks.
        img_prefix (str): Prefix of the images.
        mask_prefix (str): Prefix of the masks.
        file_extension (str): File extension of the images and masks.

    Returns:
        pd.DataFrame: Dataframe with image and mask filepaths.
    """

    image_paths = [str(path) for path in Path(image_dir).glob(f"{img_prefix}*{file_extension}")]
    mask_paths = [str(path) for path in Path(mask_dir).glob(f"{mask_prefix}*{file_extension}")]

    # Sort the paths to ensure the image and mask pairs are in the correct order
    image_paths = sorted(image_paths)
    mask_paths = sorted(mask_paths)

    # Create a DataFrame.
    df = pd.DataFrame(
        {
            "image_path": image_paths,
            "mask_path": mask_paths,
        }
    )

    return df


def split_dataframe_into_train_val_test(
    df: pd.DataFrame,
    output_dir: Path,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> None:
    """Split a dataframe into train, validation, and test dataframes.

    Args:
        df (pd.DataFrame): Dataframe to split.
        output_dir (Path): Path to the directory where the csv files will be saved.
        train_size (float): Size of the train dataframe.
        val_size (float): Size of the validation dataframe.
        test_size (float): Size of the test dataframe.
        random_state (int): Random state for reproducibility.
    """

    # Split df into train, validation, and test dataframes
    train_df, val_test_df = train_test_split(df, train_size=train_size, random_state=random_state)
    val_df, test_df = train_test_split(
        val_test_df, train_size=val_size / (val_size + test_size), random_state=random_state
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)


if __name__ == "__main__":
    # Create a DataFrame with image and mask filepaths
    df = create_dataframe_with_image_and_mask_filepaths(
        image_dir="data/images",
        mask_dir="data/masks",
        img_prefix="img_",
        mask_prefix="seg_",
        file_extension=".jpeg",
    )

    # Split the DataFrame into train, validation, and test DataFrames
    split_dataframe_into_train_val_test(
        df=df,
        output_dir="data/index",
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )
