import os
import glob
import shutil
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase.io import read


# Define the input schema for the tool
class StoreBestStructuresInput(BaseModel):
    """Input schema for StoreBestStructuresTool."""

    num_results: int = Field(
        ...,
        description="The number of lowest-energy structures to store. Must be a positive integer.",
    )


class StoreBestStructuresTool(BaseTool):
    """Tool for copying a specified number of the lowest-energy conformer files."""

    name: str = "Store Best Structures Tool"
    description: str = (
        "A tool that reads multiple .xyz molecular structure files (starting with optimized_conformer_) "
        "from the 'temporary' folder, identifies the top N structures with the lowest energy, "
        "and copies these files to the 'best_results' folder."
    )
    args_schema: Type[BaseModel] = StoreBestStructuresInput

    def _run(self, num_results: int) -> str:
        """
        Identifies and copies the N lowest-energy structures from a specified folder.
        """
        input_folder_path = "temporary"
        output_folder_path = "best_results"

        if not os.path.exists(input_folder_path):
            return f"Error: Input folder not found at '{input_folder_path}'."

        if not isinstance(num_results, int) or num_results <= 0:
            return "Error: 'num_results' must be a positive integer."

        xyz_files = glob.glob(
            os.path.join(input_folder_path, "optimized_conformer_*.xyz")
        )
        if not xyz_files:
            return f"No .xyz files found in the specified folder: {input_folder_path}."

        try:
            # Set up the calculator outside the loop for efficiency
            predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
            calculator = FAIRChemCalculator(predictor, task_name="omol")
        except Exception as e:
            return f"An error occurred while initializing the MLIP model: {e}"

        energy_results = []
        for xyz_file_path in xyz_files:
            try:
                # Check if the file is empty and skip it if it is
                if os.path.getsize(xyz_file_path) == 0:
                    print(f"Skipping empty file: {os.path.basename(xyz_file_path)}")
                    continue

                atoms_from_xyz = read(xyz_file_path)
                atoms_from_xyz.calc = calculator
                atoms_from_xyz.info["charge"] = 0
                atoms_from_xyz.info["spin"] = 0
                energy = atoms_from_xyz.get_potential_energy()

                energy_results.append(
                    {
                        "filename": os.path.basename(xyz_file_path),
                        "energy": energy,
                        "path": xyz_file_path,
                    }
                )
            except Exception as e:
                print(f"Skipping file {xyz_file_path} due to an error: {e}")
                continue

        if not energy_results:
            return "No valid XYZ files were processed."

        # Sort the results by energy in ascending order
        energy_results.sort(key=lambda x: x["energy"])

        # Select the top N results
        num_to_copy = min(num_results, len(energy_results))
        top_results = energy_results[:num_to_copy]

        # Ensure the output directory exists
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path, exist_ok=True)

        copied_files = []
        for result in top_results:
            source_path = result["path"]
            destination_path = os.path.join(output_folder_path, result["filename"])
            try:
                shutil.copy(source_path, destination_path)
                copied_files.append(result["filename"])
            except Exception as e:
                return f"Error copying file {result['filename']}: {e}"

        if not copied_files:
            return "No files were copied."

        output_string = (
            f"Successfully identified and copied {len(copied_files)} lowest-energy "
            f"structures to the '{output_folder_path}' folder:\n"
        )
        for filename in copied_files:
            output_string += f"- {filename}\n"

        return output_string
