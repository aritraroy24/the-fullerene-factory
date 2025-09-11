"""
search_pubchem_tool.py
"""

import os
import json
import httpx
from pathlib import Path
from typing import Type, List, Dict, Any, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem


# Define the input schema for the tool
class SearchPubChemInput(BaseModel):
    """Input schema for SearchPubChemTool."""

    compound_names: List[str] = Field(
        ...,
        description="A list of compound names to search for in PubChem. These are addends for fullerene derivatization.",
    )


class SearchPubChemTool(BaseTool):
    """Tool for searching PubChem for molecular structures."""

    name: str = "Search PubChem Tool"
    description: str = (
        "A tool that searches the PubChem database for molecular structures of one or more compounds. "
        "It is useful for finding information about addends for fullerene derivatization and saves the structure as an .xyz file. "
        "The tool attempts to save from SMILES first and falls back to InChI if necessary."
    )
    args_schema: Type[BaseModel] = SearchPubChemInput

    def _make_pubchem_request(self, url: str) -> Optional[Dict[str, Any]]:
        """Make a request to PubChem API with proper error handling."""
        headers = {
            "User-Agent": "data-collection-agent/1.0",
            "Accept": "application/json",
        }
        try:
            response = httpx.get(url, headers=headers, timeout=60.0)
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def _get_pubchem_structure(self, cid: int) -> Optional[Dict]:
        """Get detailed structure from PubChem CID."""
        PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        try:
            props_url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/MolecularFormula,CanonicalSMILES,InChI/JSON"
            props_data = self._make_pubchem_request(props_url)
            if not props_data:
                return None
            compound = props_data["PropertyTable"]["Properties"][0]
            return {
                "formula": compound.get("MolecularFormula", ""),
                "smiles": compound.get("CanonicalSMILES", ""),
                "inchi": compound.get("InChI", ""),
                "pubchem_cid": cid,
            }
        except Exception as e:
            print(f"Error getting structure for CID {cid}: {e}")
            return None

    def _smiles_to_xyz_and_save(
        self,
        smiles_string: str,
        compound_name: str,
        output_folder: str = "temporary/downloaded_structures",
    ) -> str:
        """
        Convert a SMILES string to XYZ format and save to file.
        """
        try:
            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None:
                return f"Error: Invalid SMILES string '{smiles_string}'."

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=1)
            AllChem.MMFFOptimizeMolecule(mol)

            xyz_block = Chem.MolToXYZBlock(mol)

            output_dir = Path(output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            full_file_path = output_dir / f"{compound_name}.xyz"

            with open(full_file_path, "w") as f:
                f.write(xyz_block)

            return f"Successfully saved XYZ file from SMILES: {full_file_path}"
        except Exception as e:
            return f"Error converting SMILES to XYZ and saving: {e}"

    def _inchi_to_xyz_and_save(
        self,
        inchi_string: str,
        compound_name: str,
        output_folder: str = "temporary/downloaded_structures",
    ) -> str:
        """
        Convert an InChI string to XYZ format and save to file.
        """
        try:
            mol = Chem.MolFromInchi(inchi_string)
            if mol is None:
                return f"Error: Invalid InChI string '{inchi_string}'."

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=1)
            AllChem.MMFFOptimizeMolecule(mol)

            xyz_block = Chem.MolToXYZBlock(mol)

            output_dir = Path(output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            full_file_path = output_dir / f"{compound_name}.xyz"

            with open(full_file_path, "w") as f:
                f.write(xyz_block)

            return f"Successfully saved XYZ file from InChI: {full_file_path}"
        except Exception as e:
            return f"Error converting InChI to XYZ and saving: {e}"

    def _run(self, compound_names: List[str]) -> str:
        """Execute the PubChem search for a list of compounds."""
        PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        results = {}
        for compound_name in compound_names:
            try:
                search_url = f"{PUBCHEM_BASE}/compound/name/{compound_name}/cids/JSON"
                data = self._make_pubchem_request(search_url)

                if not data or not data.get("IdentifierList", {}).get("CID"):
                    results[compound_name] = {"error": "No CIDs found"}
                    continue

                cid = data["IdentifierList"]["CID"][0]
                structure = self._get_pubchem_structure(cid)
                if structure:
                    # Attempt to save from SMILES first
                    save_result = self._smiles_to_xyz_and_save(
                        structure["smiles"], compound_name
                    )

                    # If SMILES conversion fails, fall back to InChI
                    if save_result.startswith("Error"):
                        save_result = self._inchi_to_xyz_and_save(
                            structure["inchi"], compound_name
                        )

                    results[compound_name] = {
                        "query": compound_name,
                        "structure": structure,
                        "file_save_status": save_result,
                    }
                else:
                    results[compound_name] = {"error": "No structure found"}
            except Exception as e:
                results[compound_name] = {"error": str(e)}

        return json.dumps(results, indent=2)
