import requests
import pandas as pd
from time import sleep

# Define the SMARTS patterns for the specified molecules/substructures
smarts_patterns = {
    "Toluene": "c1ccccc1C",
    "Anthracene": "c1ccc2c(c1)ccc3c2ccc(c3)",
    "2-Cyclohexanone": "O=C1CCCCCC1",
    "Enol Ether Substructure": "C=COC",
    "Phenyl Ester Substructure": "C=CC(=O)Oc1ccccc1",
}

num_molecules = 100
all_results = []

for mol_type, smarts_pattern in smarts_patterns.items():
    print(f"Searching for {mol_type} with SMARTS pattern: {smarts_pattern}")

    # URL-encode the SMARTS pattern
    encoded_smarts = requests.utils.quote(smarts_pattern)

    # Perform the substructure search to get CIDs
    search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsubstructure/smarts/{encoded_smarts}/cids/txt"

    try:
        response = requests.get(search_url)
        response.raise_for_status()
        cids_str = response.text.strip()
        cids = cids_str.split("\n") if cids_str else []

        if not cids:
            print(f"No compounds found for SMARTS pattern: {smarts_pattern}")
            continue

        cids_to_process = cids[:num_molecules]
        print(
            f"Found {len(cids)} total CIDs. Retrieving data for the first {len(cids_to_process)}."
        )

        # Retrieve Canonical SMILES for each CID
        for cid in cids_to_process:
            smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"
            smiles_response = requests.get(smiles_url)
            smiles_response.raise_for_status()
            smiles = smiles_response.text.strip()

            all_results.append(
                {"CID": cid, "SMILES": smiles, "Molecule Type": mol_type}
            )
            # Introduce a small delay to respect PubChem's rate limits
            sleep(0.2)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while searching for {mol_type}: {e}")
        continue

# Save the results to a CSV file
if all_results:
    try:
        df = pd.DataFrame(all_results)
        df.to_csv("molecules_with_smarts.csv", index=False)
        print("\nSuccessfully saved the data to 'molecules_with_smarts.csv'")
    except ImportError:
        print(
            "pandas library not found. Please install with 'pip install pandas' to save to a CSV."
        )
        print("Printing results instead:")
        for result in all_results:
            print(result)
    except Exception as e:
        print(f"An error occurred while writing the CSV file: {e}")
else:
    print("\nNo data was retrieved. The CSV file was not created.")
