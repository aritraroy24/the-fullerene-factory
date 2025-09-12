import os
import json
from pathlib import Path
from typing import Type, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem


# Define the input schema for the tool
class SearchPubChemInput(BaseModel):
    """Input schema for SearchPubChemTool."""

    compound_names: List[str] = Field(
        ...,
        description="A list of compound names to search for in PubChem. These are addends for fullerene derivatization. Also, they can be passed as SMILES strings.",
    )


class SearchPubChemTool(BaseTool):
    """Tool for searching PubChem for molecular structures."""

    name: str = "Search PubChem Tool"
    description: str = (
        "A tool that searches the PubChem database for molecular structures of one or more compounds. "
        "It is useful for finding information about addends for fullerene derivatization and saves the structure as an .xyz file. "
        "The tool attempts to save from SMILES."
    )
    args_schema: Type[BaseModel] = SearchPubChemInput

    def _smiles_to_xyz_and_save(
        self,
        smiles_string: str,
        compound_name: str,
    ) -> str:
        """
        Convert a SMILES string to XYZ format and save to an RDKit-readable file.
        """
        output_folder = "temporary"
        try:
            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None:
                return f"Error: Invalid SMILES string '{smiles_string}'."

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=1)

            AllChem.MMFFOptimizeMolecule(mol)

            # Get the XYZ block.
            xyz_string = Chem.MolToXYZBlock(mol)

            output_dir = Path(output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            full_file_path = output_dir / f"{compound_name}.xyz"

            with open(full_file_path, "w") as f:
                f.write(xyz_string)

            return f"Successfully saved XYZ file from SMILES: {full_file_path}"
        except Exception as e:
            return f"Error converting SMILES to XYZ and saving: {e}"

    def _run(self, compound_names: List[str]) -> str:
        """Execute the PubChem search for a list of compounds using pubchempy."""
        results = {}
        for i, compound_name in enumerate(compound_names):
            try:
                # Step 1: Check if the input is a valid SMILES string.
                # RDKit's MolFromSmiles returns None for invalid SMILES strings.
                rdkit_mol = Chem.MolFromSmiles(compound_name, sanitize=True)

                if rdkit_mol is not None:
                    # Case 1: The input is a valid SMILES.
                    smiles = compound_name

                    save_result = self._smiles_to_xyz_and_save(smiles, f"addend_{i}")

                    results[compound_name] = {
                        "query": compound_name,
                        "structure": {
                            "smiles": smiles,
                            "pubchem_cid": "N/A",  # Not applicable for a direct SMILES input
                        },
                        "file_save_status": save_result,
                        "source": "Direct SMILES input",
                    }
                    continue  # Move to the next compound

                # Case 2: The input is not a valid SMILES, so proceed with PubChem search.
                # Use pubchempy to get compounds by name.
                compounds = pcp.get_compounds(compound_name, "name")

                if not compounds:
                    results[compound_name] = {"error": "No compound found."}
                    continue

                # Get the first compound from the search results.
                compound = compounds[0]
                cid = compound.cid
                smiles = compound.canonical_smiles

                # Perform the SMILES to XYZ conversion and save.
                save_result = self._smiles_to_xyz_and_save(smiles, f"addend_{i}")

                # Create a structure dictionary for the result.
                structure = {
                    "formula": compound.molecular_formula,
                    "smiles": smiles,
                    "inchi": compound.inchi,
                    "pubchem_cid": cid,
                }

                results[compound_name] = {
                    "query": compound_name,
                    "structure": structure,
                    "file_save_status": save_result,
                    "source": "PubChem API",
                }

            except pcp.PubChemHTTPError as e:
                results[compound_name] = {"error": f"PubChem API error: {e}"}
            except Exception as e:
                results[compound_name] = {"error": str(e)}

        return json.dumps(results, indent=2)
