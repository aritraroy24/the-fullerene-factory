import os
import json
import glob
from pathlib import Path
from typing import Type, List, Dict, Any, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import torch
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase.io import read, write
from ase.optimize import BFGS
from ase import Atoms
from tqdm import tqdm
from huggingface_hub import login

login(token=os.getenv("HF_API_KEY"))


# Define the input schema for the tool
class RunMLIPInput(BaseModel):
    """Input schema for RunMLIPTool."""

    charge: int = Field(0, description="The charge of the molecule for optimization.")
    spin: int = Field(
        0, description="The spin multiplicity of the molecule for optimization."
    )


class RunMLIPTool(BaseTool):
    """Tool for running geometry optimization using a Machine Learning Interatomic Potential (MLIP)."""

    name: str = "Run MLIP Optimization Tool"
    description: str = (
        "A tool that performs geometry optimization on multiple molecular structures using the FairChem UMA MLIP model. "
        "It takes a folder of .xyz files, optimizes them, and saves a specified number of the lowest-energy structures to an output folder."
    )
    args_schema: Type[BaseModel] = RunMLIPInput

    def _run(
        self,
        charge: int = 0,
        spin: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Implementation of multi-file MLIP optimization using UMA model.
        """
        folder_path = "temporary"
        if not os.path.exists(folder_path):
            print(f"Input folder not found: { folder_path}")
            return []

        # Find all XYZ files in the folder starting with generated_conformer_
        xyz_pattern = os.path.join(folder_path, "generated_conformer_*.xyz")
        xyz_files = glob.glob(xyz_pattern)

        if not xyz_files:
            print(f"No .xyz files found in { folder_path}")
            return []

        try:
            # Initialize FairChem UMA calculator
            print("⚠️ Loading FairChem UMA model...")
            predictor = pretrained_mlip.get_predict_unit(
                "uma-s-1", device="cuda" if torch.cuda.is_available() else "cpu"
            )
            calculator = FAIRChemCalculator(predictor, task_name="omol")
            print("✅ FairChem UMA model loaded successfully")
        except Exception as e:
            print(f"Failed to load FairChem UMA model: {str(e)}")
            return []

        optimized_results = []

        # Wrap the loop with tqdm for a progress bar
        for xyz_file in tqdm(xyz_files, desc="Optimizing molecules"):
            try:
                filename = os.path.basename(xyz_file)

                try:
                    atoms = read(xyz_file)
                except Exception as e:
                    print(f"Failed to read {filename}: {str(e)}")
                    continue

                try:
                    atoms.info["charge"] = charge
                    atoms.info["spin"] = spin
                    atoms.calc = calculator

                    # Perform geometry optimization using BFGS for better convergence
                    dyn = BFGS(atoms)
                    dyn.run(fmax=0.05)

                    energy_ev = atoms.get_potential_energy()
                    energy_kcal = energy_ev * 23.0609

                    optimized_results.append(
                        {
                            "atoms": atoms.copy(),
                            "energy_ev": energy_ev,
                            "energy_kcal": energy_kcal,
                            "original_file": filename,
                            "original_path": xyz_file,
                        }
                    )

                except Exception as e:
                    print(f"FairChem UMA optimization failed for {filename}: {str(e)}")
                    continue

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

        if not optimized_results:
            print("No successful optimizations to save.")
            return []

        for i, result in enumerate(optimized_results):
            xyz_filename = os.path.join(folder_path, f"optimized_conformer_{i}.xyz")
            write(xyz_filename, result["atoms"])

        print(f"\n✅ FairChem UMA optimization complete!")
        print(f"Saved {len(optimized_results)} structures in '{ folder_path}'")

        return optimized_results
