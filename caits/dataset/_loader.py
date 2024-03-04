from typing import Optional
import os
from caits.loading import audio_loader, csv_loader
from ._dataset import Dataset


class DataLoader:

    @staticmethod
    def _get_file_types(path: str) -> set:
        formats = set()
        # Iterate over each item in the given directory
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            # Check if the item is a directory (i.e., a class folder)
            if os.path.isdir(item_path):
                # Iterate over files within the class directory
                for filename in os.listdir(item_path):
                    _, ext = os.path.splitext(filename)
                    if ext:  # Ensure there is an extension
                        # Remove the dot and convert to lower case
                        formats.add(ext.lstrip('.').lower())

        return formats

    @classmethod
    def load_from(
            cls,
            path: str,
            classes: Optional[list[str]] = None
    ) -> Dataset:
        # check dataset's dir types
        formats = cls._get_file_types(path)

        # check if uniform type and class assosiated loading function
        if len(formats) == 1:
            _format = list(formats)[0]
            if _format == "wav":
                dict_data = audio_loader(path, classes=classes)
            elif _format == "csv":
                dict_data = csv_loader(path, classes=classes)
            else:
                # No loader implemented for this file format
                raise NotImplementedError(f"Loading for the {_format} format \
                                           is not implemented.")

            # Return loaded data using Dataset Object
            return Dataset(
                X=dict_data["X"],
                y=dict_data["y"],
                id=dict_data["id"]
            )

        else:
            # Dataset contains mixed file formats
            raise ValueError("Directory files must have the same format.")
