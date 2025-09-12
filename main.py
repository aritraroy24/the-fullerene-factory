import warnings
import json
from pydantic import PydanticDeprecatedSince20

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)


from fullerenefactory.initialization_crew.crew import InitializationCrew
from fullerenefactory.conformer_crew.crew import ConformerCrew


# main.py
def main():
    query = "Generate a C42 fullerene structure with an addend with smiles 'C1C(CC2=CC=CC=C2C1C3=C(C4=CC=CC=C4OC3=O)O)C5=CC=C(C=C5)OCC6=CC=C(C=C6)C(F)(F)F'. Get 2 steps addition products."
    try:
        parsed_query_output = (
            InitializationCrew().crew().kickoff(inputs={"query": query})
        )
        parsed_query_str = parsed_query_output.raw
        parsed_query_dict = json.loads(parsed_query_str)

        current_fullerene = parsed_query_dict.get("base_fullerene").lower()
        addends = parsed_query_dict.get("addends", [])
        num_steps = parsed_query_dict.get("num_steps", 1)
        num_angles = parsed_query_dict.get(
            "num_angles", 5
        )  # Get num_angles from the parsed query

        if not current_fullerene:
            print("Base fullerene not found. Exiting...")
            return

        if not addends:
            print("Addend not found. Exiting...")
            return

        for i in range(num_steps):
            if i < len(addends):
                current_addend = f"addend_{i}"
            else:
                current_addend = "addend_0"

            print(
                f"\n\n🚩 Starting step {i+1} with fullerene {current_fullerene} and addend {current_addend}"
            )

            # Create a new crew for each step with updated inputs
            conformer_crew = ConformerCrew(
                parsed_query=parsed_query_dict,
                current_fullerene=current_fullerene,
                current_addend=current_addend,
                current_step=i + 1,
            ).crew()

            # Pass the dynamic values as inputs to the crew's kickoff method
            # This is the crucial change that provides the context to the LLM
            crew_output = conformer_crew.kickoff(
                inputs={
                    "current_fullerene": current_fullerene,
                    "current_addend": current_addend,
                    "num_angles": num_angles,
                    "current_step": i + 1,
                }
            )

            current_fullerene = crew_output.raw

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
