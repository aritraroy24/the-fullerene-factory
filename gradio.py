import gradio as gr
import subprocess
from PIL import Image
import os

def xyz_to_pdb(xyz_path):
    """Convert an XYZ file to PDB using Open Babel."""
    pdb_path = xyz_path.replace(".xyz", ".pdb")
    try:
        subprocess.run(["obabel", xyz_path, "-O", pdb_path], check=True)
        print(f"PDB generated: {pdb_path}")
        return pdb_path
    except subprocess.CalledProcessError as e:
        print("Open Babel conversion failed:", e)
        return None

def generate_image(file):
    """Generate a static PNG image of the molecule with PyMOL."""
    xyz_path = file.name
    pdb_path = xyz_to_pdb(xyz_path)
    if pdb_path is None:
        raise FileNotFoundError("PDB file was not created.")

    img_path = os.path.abspath("molecule.png")

    # PyMOL script
    pymol_script = f"""
reinitialize
load {pdb_path}
show sticks
bg_color white
png {img_path}, width=800, height=600, dpi=300, ray=1
quit
"""
    try:
        subprocess.run(["pymol", "-cq"], input=pymol_script.encode(), check=True)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"PNG not generated: {img_path}")
        print(f"PNG generated successfully: {img_path}")
    except subprocess.CalledProcessError as e:
        print("PyMOL execution failed:", e)
        raise RuntimeError("PyMOL failed to generate image.")

    # Read image with Pillow
    img = Image.open(img_path)
    return img

# Gradio Interface
demo = gr.Interface(
    fn=generate_image,
    inputs=gr.File(label="Upload XYZ File", type="filepath"),
    outputs=gr.Image(label="Molecule Image", type="pil"),
    title="Static Molecular Visualization",
    description="Upload an XYZ file → convert to PDB → generate 3D image with PyMOL."
)

demo.launch(share=False)
