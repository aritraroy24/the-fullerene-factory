import os
import glob
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase.io import read
import torch


# Define an empty input schema, as this tool requires no input parameters.
class GetBestResultsInput(BaseModel):
    """Input schema for GetBestResultsTool."""

    pass


class GetBestResultsTool(BaseTool):
    """
    A tool that analyzes all molecular structures in the 'best_results' folder and provides a detailed report.
    """

    name: str = "Get Best Results Tool"
    description: str = (
        "A tool that reads all .xyz molecular structure files from the 'best_results' folder, "
        "calculates their energy, and returns a detailed text report of their properties, "
        "sorted by energy from lowest to highest."
    )
    args_schema: Type[BaseModel] = GetBestResultsInput

    def _run(self) -> str:
        """
        Calculates the energy of all .xyz files in the 'best_results' folder and returns a report.
        """
        input_folder_path = "best_results"

        if not os.path.exists(input_folder_path):
            return f"Error: Input folder not found at '{input_folder_path}'. Cannot process structures."

        xyz_files = glob.glob(
            os.path.join(input_folder_path, "optimized_conformer_*.xyz")
        )
        if not xyz_files:
            return f"No .xyz files found in the specified folder: {input_folder_path}"

        try:
            # Set up the calculator outside the loop for efficiency
            predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cpu")
            calculator = FAIRChemCalculator(predictor, task_name="omol")
        except Exception as e:
            return f"An error occurred while initializing the MLIP model: {e}"

        energy_results = []
        for xyz_file_path in xyz_files:
            try:
                atoms_from_xyz = read(xyz_file_path)
                atoms_from_xyz.calc = calculator
                atoms_from_xyz.info["charge"] = 0
                atoms_from_xyz.info["spin"] = 0
                energy = atoms_from_xyz.get_potential_energy()

                energy_results.append(
                    {
                        "filename": os.path.basename(xyz_file_path),
                        "energy_ev": energy,
                    }
                )
            except Exception as e:
                print(f"Skipping file {xyz_file_path} due to error: {e}")
                continue

        if not energy_results:
            return "No valid optimized structures were processed."

        # Sort results by energy for the report
        energy_results.sort(key=lambda x: x["energy_ev"])

        # Prepare the final text report
        text_report = f"Successfully processed {len(energy_results)} optimized structures. The results are:\n"
        for i, res in enumerate(energy_results):
            text_report += f"{i+1}. Filename: {res['filename']}, Energy: {res['energy_ev']:.6f} eV\n"

        return text_report
