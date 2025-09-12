import warnings
import json
from pydantic import PydanticDeprecatedSince20

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

# Import CrewAI components
from .initialization_crew.crew import InitializationCrew
from .conformer_crew.crew import ConformerCrew
from .post_processing_crew.crew import PostProcessingCrew


def run_workflow(query: str):
    """
    Runs the Fullerene Factory workflow from a given query.
    """
    try:
        print("🚩 Starting Initialization Crew")
        parsed_query_output = (
            InitializationCrew().crew().kickoff(inputs={"query": query})
        )
        parsed_query_str = parsed_query_output.raw
        parsed_query_dict = json.loads(parsed_query_str)

        current_fullerene = parsed_query_dict.get("base_fullerene").lower()
        addends = parsed_query_dict.get("addends", [])
        num_steps = parsed_query_dict.get("num_steps", 1)
        num_angles = parsed_query_dict.get("num_angles", 5)

        if not current_fullerene:
            return "Base fullerene not found. Exiting..."

        if not addends:
            return "Addend not found. Exiting..."

        for i in range(num_steps):
            current_addend = f"addend_{i}" if i < len(addends) else "addend_0"

            print(
                f"\n\n🚩 Starting Conformer Crew for step {i+1} with fullerene {current_fullerene} and addend {current_addend}"
            )

            conformer_crew = ConformerCrew(
                parsed_query=parsed_query_dict,
                current_fullerene=current_fullerene,
                current_addend=current_addend,
                current_step=i + 1,
            ).crew()

            crew_output = conformer_crew.kickoff(
                inputs={
                    "parsed_query": parsed_query_dict,
                    "current_fullerene": current_fullerene,
                    "current_addend": current_addend,
                    "num_angles": num_angles,
                    "current_step": i + 1,
                }
            )

            current_fullerene = crew_output.raw

        print("\n\n🚩 Starting Post-Processing Crew")

        post_processing_crew = PostProcessingCrew(parsed_query=parsed_query_dict).crew()
        crew_result = post_processing_crew.kickoff(
            inputs={
                "parsed_query": parsed_query_dict,
                "num_results": parsed_query_dict.get("num_results", 3),
                "is_store_database": parsed_query_dict.get("is_store_database", False),
                "is_delete_intermediate_files": parsed_query_dict.get(
                    "is_delete_intermediate_files", True
                ),
            }
        )

        return crew_result

    except Exception as e:
        return f"Error occurred: {e}"


if __name__ == "__main__":
    # Example usage for direct execution
    query = "Generate a C42 fullerene structure with an addend with smiles 'COC1=CC2=C(C=C1)C=CC(=O)O2'. Get single steps addition products where the total number of angles to make conformers is 2. Also, store all the optimized structures in the database and and return the energy report as text data."
    final_output = run_workflow(query)
    print(f"\n\n🎯🎯 Final Results: \n{final_output}")
