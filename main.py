import warnings
from pydantic import PydanticDeprecatedSince20

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)


from fullerenefactory.initialization_crew.crew import InitializationCrew


def main():
    query = "Generate a C60 fullerene structure with anthracene molecule as an addend. Get only one step addition product."
    try:
        InitializationCrew().crew().kickoff(inputs={"query": query})
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
