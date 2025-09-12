import warnings
import json
from pydantic import PydanticDeprecatedSince20

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)


from fullerenefactory.initialization_crew.crew import InitializationCrew
from fullerenefactory.isomer_crew.crew import IsomerCrew


def main():
    query = "Generate a C80 fullerene structure with anthracene molecule as an addend. Get only one step addition product."
    try:
        parsed_query_str = InitializationCrew().crew().kickoff(inputs={"query": query})
        parsed_query_dict = json.loads(parsed_query_str)
        for i in range(parsed_query_dict.get("num_steps", 1)):
            # get the addend list
            addends = parsed_query_dict.get("addends", [])
            if addends:
                if i < len(addends):
                    current_addend = addends[i]
                    IsomerCrew(
                        parsed_query=parsed_query_dict, addend=current_addend
                    ).crew().kickoff()
                elif len(addends) == 1:
                    current_addend = addends[0]
                    IsomerCrew(
                        parsed_query=parsed_query_dict, addend=current_addend
                    ).crew().kickoff()
            else:
                print("Addend not found. Exiting...")
                break
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
