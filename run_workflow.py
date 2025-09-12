from fullerenefactory import run_workflow

if __name__ == "__main__":
    query_to_run = "Generate a C42 fullerene structure with an addend with smiles 'COC1=CC2=C(C=C1)C=CC(=O)O2'. Get single steps addition products where the total number of angles to make conformers is 3. Also, store all the optimized structures in the database. Get the 5 best structures and return the energy report as text data."

    print("🚀 Starting the Fullerene Factory workflow from an external script...")
    final_output = run_workflow(query_to_run)

    print("\n\n✅ Final Workflow Output:")
    print(final_output)
