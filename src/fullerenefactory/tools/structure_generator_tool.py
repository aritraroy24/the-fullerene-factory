import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem.rdMolTransforms import TransformConformer
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from typing import Type
import os


# Define the input schema for the tool
class StructureGenerationInput(BaseModel):
    """Input schema for StructureGenerationTool."""

    current_fullerene: str = Field(
        ...,
        description="Name of the fullerene structure to load from the path. The name should be passed in lowercase, e.g., 'c{n}', where n is the number of carbon atoms.",
    )
    current_addend: str = Field(
        "uma_optimized_outputs",
        description="Name of the addend structure to load from the path. The name should be passed in lowercase, e.g., 'addend_{n}', where n is an integer.",
    )
    current_step: int = Field(
        1,
        description="Current step in the conformer generation process.",
    )
    num_angles: int = Field(
        5,
        description="Number of angles to to generate for addend and fullerene (each angle will be considered in X, Y, Z directions).",
    )


class StructureGenerationTool(BaseTool):
    """Tool for generating molecular structures using RDKit transformations."""

    name: str = "Run Structure Generation Tool"
    description: str = (
        "A tool that generates molecular structures using RDKit transformations. "
        "It takes a set of input parameters and produces molecular conformers and saves them in a folder."
    )
    args_schema: Type[BaseModel] = StructureGenerationInput

    def _run(
        self,
        current_fullerene: str,
        current_addend: str,
        num_angles: int,
        current_step: int,
    ) -> str:
        """
        The main execution method of the tool.
        """

        def clean_xyz_data(xyz_block: str) -> str:
            """
            Cleans an XYZ file block by removing any extra data from the coordinate
            lines, ensuring it can be parsed by RDKit's strict parser.
            """
            lines = xyz_block.strip().split("\n")
            if len(lines) < 2:
                return ""
            cleaned_lines = [lines[0], lines[1]]
            for line in lines[2:]:
                parts = line.split()
                if len(parts) >= 4:
                    cleaned_line = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}"
                    cleaned_lines.append(cleaned_line)
            return "\n".join(cleaned_lines)

        def combine_molecules_with_separation(
            fullerene_mol, addend_mol, min_separation_distance=2.5
        ):
            """
            Translates an addend molecule relative to a fullerene molecule to achieve a
            desired separation distance between their surfaces, then combines them.
            """
            if fullerene_mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(fullerene_mol, randomSeed=1)
            if addend_mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(addend_mol, randomSeed=1)

            if (
                fullerene_mol.GetNumConformers() == 0
                or addend_mol.GetNumConformers() == 0
            ):
                print(
                    "Error: Could not generate 3D coordinates for one or both molecules."
                )
                return None

            fullerene_conf = fullerene_mol.GetConformer(0)
            addend_conf = addend_mol.GetConformer(0)

            def get_center_of_mass(mol, conf):
                coords = conf.GetPositions()
                return np.mean(coords, axis=0)

            addend_com = get_center_of_mass(addend_mol, addend_conf)
            fullerene_com = get_center_of_mass(fullerene_mol, fullerene_conf)

            fullerene_coords = fullerene_conf.GetPositions()
            distances_from_com = np.linalg.norm(
                fullerene_coords - fullerene_com, axis=1
            )
            fullerene_radius = np.max(distances_from_com)

            desired_separation_distance = fullerene_radius + min_separation_distance

            vector_com_to_com = addend_com - fullerene_com
            current_separation = np.linalg.norm(vector_com_to_com)

            if current_separation < desired_separation_distance:
                unit_vector = vector_com_to_com / current_separation
                translation_magnitude = desired_separation_distance - current_separation
                translation_vector = unit_vector * translation_magnitude
                translation_matrix = np.eye(4)
                translation_matrix[:3, 3] = translation_vector
                rdMolTransforms.TransformConformer(addend_conf, translation_matrix)

            combined_mol = Chem.CombineMols(fullerene_mol, addend_mol)
            return combined_mol

        def save_molecules_as_xyz(
            molecules_list,
            folder_path="temporary",
            filename_prefix="generated_conformer_",
        ):
            """
            Saves a list of RDKit molecule objects as individual .xyz files.
            """
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"\nCreated folder: {folder_path}")

            for i, mol in enumerate(molecules_list):
                filename = os.path.join(folder_path, f"{filename_prefix}{i}.xyz")
                try:
                    Chem.MolToXYZFile(mol, filename)
                except Exception as e:
                    print(f"Error saving {filename}: {e}")
            print("Saving process complete.")

        # Updated logic to read from the specified paths
        fullerene_path = os.path.join("temporary", f"{current_fullerene}.xyz")
        addend_path = os.path.join("temporary", f"{current_addend}.xyz")

        try:
            with open(fullerene_path, "r") as f:
                fullerene_xyz_data = f.read()
            cleaned_fullerene_xyz = clean_xyz_data(fullerene_xyz_data)
            fullerene_mol = Chem.MolFromXYZBlock(cleaned_fullerene_xyz)
            if fullerene_mol is None:
                return f"Error: Could not create RDKit molecule from '{fullerene_path}'. Check file format."
        except FileNotFoundError:
            return f"Error: Fullerene file not found at '{fullerene_path}'"
        except Exception as e:
            return f"An error occurred while processing '{fullerene_path}': {e}"

        try:
            with open(addend_path, "r") as f:
                addend_xyz_data = f.read()
            cleaned_addend_xyz = clean_xyz_data(addend_xyz_data)
            addend_mol = Chem.MolFromXYZBlock(cleaned_addend_xyz)
            if addend_mol is None:
                return f"Error: Could not create RDKit molecule from '{addend_path}'. Check file format."
        except FileNotFoundError:
            return f"Error: Addend file not found at '{addend_path}'"
        except Exception as e:
            return f"An error occurred while processing '{addend_path}': {e}"

        # Create an array of equally spaced angles in radians from 0 to 2*pi
        angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
        rotation_matrices = []

        for angle_rad in angles:
            rot_matrix_x = np.array(
                [
                    [1, 0, 0, 0],
                    [0, np.cos(angle_rad), -np.sin(angle_rad), 0],
                    [0, np.sin(angle_rad), np.cos(angle_rad), 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.double,
            )
            rotation_matrices.append(rot_matrix_x)

            rot_matrix_y = np.array(
                [
                    [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
                    [0, 1, 0, 0],
                    [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.double,
            )
            rotation_matrices.append(rot_matrix_y)

            rot_matrix_z = np.array(
                [
                    [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
                    [np.sin(angle_rad), np.cos(angle_rad), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.double,
            )
            rotation_matrices.append(rot_matrix_z)

        transformed_fullerenes = []
        transformed_addends = []

        for rot_matrix in rotation_matrices:
            new_fullerene_mol = Chem.Mol(fullerene_mol)
            conf_fullerene = new_fullerene_mol.GetConformer(0)
            rdMolTransforms.TransformConformer(conf_fullerene, rot_matrix)
            transformed_fullerenes.append(new_fullerene_mol)

            new_addend_mol = Chem.Mol(addend_mol)
            conf_addend = new_addend_mol.GetConformer(0)
            rdMolTransforms.TransformConformer(conf_addend, rot_matrix)
            transformed_addends.append(new_addend_mol)

        combined_molecules = []
        for fullerene in transformed_fullerenes:
            for addend in transformed_addends:
                combined_mol = combine_molecules_with_separation(fullerene, addend)
                if combined_mol is not None:
                    combined_molecules.append(combined_mol)

        if combined_molecules:
            save_molecules_as_xyz(combined_molecules)
            print(
                f"✅ Successfully generated and saved {len(combined_molecules)} molecular conformers for optimization."
            )
            return f"Successfully generated and saved {len(combined_molecules)} molecular conformers in the 'temporary' folder."
        else:
            return "Failed to generate any combined molecular structures."
