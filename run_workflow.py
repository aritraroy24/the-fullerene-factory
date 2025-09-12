from fullerenefactory import run_workflow

if __name__ == "__main__":
    query_to_run = "Generate a C60 fullerene structure with the addend as diethyl malonate. Get single steps addition products where the total number of angles to make conformers is 4. Also, store all the optimized structures in the database. Get the 5 best structures and return the energy report as text data."

    print("🚀 Starting the Fullerene Factory workflow from an external script...")
    final_output = run_workflow(query_to_run)

    print("\n\n✅ Final Workflow Output:")
    print(final_output)
