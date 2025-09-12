from typing import Type, List, Dict, Any, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import os
import glob
import torch
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase.io import read


# Define the input schema for the tool
class LowestEnergyStructuresInput(BaseModel):
    """Input schema for LowestEnergyStructuresTool."""


class LowestEnergyStructuresTool(BaseTool):
    """Tool for finding the lowest-energy conformers from a set of molecular structures."""

    name: str = "Find Lowest Energy Structures Tool"
    description: str = (
        "A tool that reads multiple .xyz molecular structure files (starting with optimized_conformer_) from a specified folder, "
        "calculates their energy using a Machine Learning Interatomic Potential (MLIP) model, "
        "and identifies the top N structures with the lowest energy."
    )
    args_schema: Type[BaseModel] = LowestEnergyStructuresInput

    def _run(self) -> str:
        """
        Calculates the energy of all .xyz files in a folder and returns the
        details of the lowest-energy structures.
        """
        input_folder_path = "temporary"
        if not os.path.exists(input_folder_path):
            return f"Error: Input folder not found at '{input_folder_path}'"

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

                # Calculate the energy
                energy = atoms_from_xyz.get_potential_energy()

                # Store the filename and energy
                energy_results.append(
                    {
                        "filename": os.path.basename(xyz_file_path),
                        "energy": energy,
                        "atoms": atoms_from_xyz,
                    }
                )
                print(
                    f"Processed {os.path.basename(xyz_file_path)}: Energy = {energy:.6f} eV"
                )

            except FileNotFoundError:
                print(f"Error: XYZ file not found at '{xyz_file_path}'")
                continue
            except Exception as e:
                print(f"An error occurred while processing '{xyz_file_path}': {e}")
                continue

        if not energy_results:
            return "No valid XYZ files were processed."

        # Sort the results by energy in ascending order
        energy_results.sort(key=lambda x: x["energy"])

        # Get the best structure energetically
        lowest_energy_structure = energy_results[0]

        # Extract just the filename without the .xyz extension
        filename_without_extension = os.path.splitext(
            lowest_energy_structure["filename"]
        )[0]

        output = f"\nSuccessfully processed {len(energy_results)} files. The lowest energy structure is: {filename_without_extension}\n"

        print(output)

        # Return only the name of the best structure file, e.g., "optimized_conformer_2"
        return filename_without_extension
