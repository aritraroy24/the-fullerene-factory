# tools/file_deletion_tool.py

import os
import shutil
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class FolderDeletionTool(BaseTool):
    """A tool to delete an entire folder and its contents recursively."""

    name: str = "Folder Deletion Tool"
    description: str = "Deletes 'temporary' folder and all its contents recursively."

    def _run(self) -> str:
        """
        Deletes the entire folder and its contents.
        """
        try:
            directory = "temporary"
            if not os.path.isdir(directory):
                return f"Error: Directory not found at '{directory}'."

            shutil.rmtree(directory)

            return f"Successfully deleted the entire folder at '{directory}'."

        except Exception as e:
            return f"An error occurred while trying to delete the folder: {e}"
