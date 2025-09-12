"""
download_fullerene_tool.py
"""

import os
import json
import httpx
from pathlib import Path
from typing import Type, Any, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# Define the input schema for the tool
class DownloadFullereneInput(BaseModel):
    """Input schema for DownloadFullereneTool."""

    compound_name: str = Field(
        ...,
        description="The name of the fullerene compound to download, in C{n} format (e.g., 'C60', 'C70').",
    )


class DownloadFullereneTool(BaseTool):
    """Tool for downloading base fullerene structures."""

    name: str = "Download Fullerene Tool"
    description: str = (
        "A tool that downloads the base fullerene structure in .xyz format. "
        "The input should be in the 'C{n}' format, such as 'C60'. The output folder name is 'temporary'."
    )
    args_schema: Type[BaseModel] = DownloadFullereneInput

    fullerene_data_file: str = Field(
        "resources/fullerene_molecules.json",
        description="The path to the fullerene molecules JSON file.",
    )
    fullerene_data: Optional[dict] = Field(
        None, description="The loaded fullerene data."
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.fullerene_data = self._load_fullerene_data()

    def _load_fullerene_data(self) -> dict:
        """Helper function to load fullerene data from JSON file."""
        try:
            # Construct a path relative to the current script's location
            script_dir = Path(__file__).parent
            file_path = script_dir / self.fullerene_data_file
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading fullerene data from {self.fullerene_data_file}: {e}")
            return {}

    def _run(self, compound_name: str) -> str:
        """Execute the fullerene structure download."""
        if not self.fullerene_data:
            return "Error: Fullerene data not loaded."

        if not compound_name.startswith("C") or not compound_name[1:].isdigit():
            return "Error: Compound name must be in 'C{n}' format, e.g., 'C60'."

        if compound_name not in self.fullerene_data:
            return f"Error: Compound '{compound_name}' not found in the database."

        download_url = self.fullerene_data[compound_name].get("download_link", "")
        if not download_url:
            return f"Error: No download link found for '{compound_name}'."

        output_folder = "temporary"
        os.makedirs(output_folder, exist_ok=True)
        file_path = os.path.join(output_folder, f"{compound_name.lower()}.xyz")
        headers = {"User-Agent": "data-collection-agent/1.0"}

        try:
            response = httpx.get(download_url, headers=headers, timeout=60.0)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            return f"Successfully downloaded '{compound_name}' to '{file_path}'"
        except httpx.HTTPStatusError as e:
            return f"HTTP error occurred while downloading '{compound_name}': {e}"
        except Exception as e:
            return f"Error downloading fullerene structure: {e}"
