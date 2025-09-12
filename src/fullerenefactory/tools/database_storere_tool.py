from typing import Type, List, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
import glob
from datasets import Dataset
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()


# Define the input schema for the tool
class DatabaseStorerInput(BaseModel):
    """Input schema for DatabaseStorerTool."""

    dataset_name: str = Field(
        "fullerene-conformers",
        description="The name of the Hugging Face dataset to create or update.",
    )


class DatabaseStorerTool(BaseTool):
    """Tool for storing molecular conformers as a Hugging Face dataset."""

    name: str = "Store Hugging Face Dataset Tool"
    description: str = (
        "A tool that uploads all optimized molecular structure files from the "
        "`temporary` folder to a Hugging Face Hub dataset. It requires a "
        "Hugging Face token from the HF_API_KEY environment variable for authentication."
    )
    args_schema: Type[BaseModel] = DatabaseStorerInput

    def _run(self, dataset_name: str = "fullerene-conformers") -> str:
        """
        Stores all `optimized_conformer_*.xyz` files in a Hugging Face dataset.
        The Hugging Face token must be available in the HF_API_KEY environment variable.
        """
        hf_token = os.getenv("HF_API_KEY")
        if not hf_token:
            return "Error: HF_API_KEY environment variable not set. Cannot authenticate with Hugging Face Hub."

        try:
            login(token=hf_token, add_to_git_credential=False)
            print("Successfully authenticated with Hugging Face Hub.")
        except Exception as e:
            return f"Failed to authenticate with Hugging Face Hub: {e}"

        input_folder_path = "temporary"
        xyz_pattern = os.path.join(input_folder_path, "optimized_conformer_*.xyz")
        xyz_files = glob.glob(xyz_pattern)

        if not xyz_files:
            return f"No .xyz files found to store in the specified folder: {input_folder_path}"

        dataset_data = {"filename": [], "content": []}
        failed_reads = []

        for xyz_file_path in xyz_files:
            try:
                with open(xyz_file_path, "r") as f:
                    file_content = f.read()

                filename = os.path.basename(xyz_file_path)

                dataset_data["filename"].append(filename)
                dataset_data["content"].append(file_content)
                print(f"Successfully read {filename}.")

            except Exception as e:
                failed_reads.append(f"Failed to read {xyz_file_path}: {e}")

        if not dataset_data["filename"]:
            summary_message = "No files were successfully read. Dataset not created."
            if failed_reads:
                summary_message += f"\n\nDetails of failed reads:\n"
                summary_message += "\n".join(failed_reads)
            return summary_message

        try:
            # Create a Dataset object from the collected data
            dataset = Dataset.from_dict(dataset_data)

            # Push the dataset to the Hugging Face Hub
            dataset.push_to_hub(repo_id=dataset_name)

            summary_message = f"Successfully uploaded {len(dataset)} files to the dataset '{dataset_name}' on Hugging Face Hub."
            if failed_reads:
                summary_message += (
                    f"\n\nFailed to read {len(failed_reads)} files. Details:\n"
                )
                summary_message += "\n".join(failed_reads)

            return summary_message

        except Exception as e:
            return f"Failed to create or push dataset to Hugging Face Hub: {e}"
