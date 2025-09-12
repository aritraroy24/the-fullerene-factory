import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

# Import the new CrewAI tools
from ..tools.database_storere_tool import DatabaseStorerTool
from ..tools.store_best_structure_tool import StoreBestStructuresTool
from ..tools.get_best_results_tool import GetBestResultsTool
from ..tools.delete_temporary_files import FolderDeletionTool

api_key = os.getenv("DEEPSEEK_API_KEY")


@CrewBase
class PostProcessingCrew:
    """Post-Processing Crew"""

    def __init__(self, parsed_query: dict = None):
        self.parsed_query = parsed_query

    # AGENTS
    @agent
    def database_storer(self) -> Agent:
        dataset_storer_tool = DatabaseStorerTool()
        return Agent(
            config=self.agents_config["database_storer"],
            verbose=True,
            llm=LLM(model="deepseek/deepseek-chat", api_key=api_key),
            tools=[dataset_storer_tool],
            is_store_database=self.parsed_query.get("is_store_database", False),
        )

    @agent
    def best_conformers_finder(self) -> Agent:
        store_best_structures_tool = StoreBestStructuresTool()
        return Agent(
            config=self.agents_config["best_conformers_finder"],
            verbose=True,
            llm=LLM(model="deepseek/deepseek-chat", api_key=api_key),
            tools=[store_best_structures_tool],
            num_results=self.parsed_query.get("num_results", 3),
        )

    @agent
    def temporary_files_remover(self) -> Agent:
        folder_deletion_tool = FolderDeletionTool()
        return Agent(
            config=self.agents_config["temporary_files_remover"],
            verbose=True,
            llm=LLM(model="deepseek/deepseek-chat", api_key=api_key),
            is_delete_intermediate_files=self.parsed_query.get(
                "is_delete_intermediate_files", True
            ),
            tools=[folder_deletion_tool],
        )

    @agent
    def best_result_shower(self) -> Agent:
        get_best_results_tool = GetBestResultsTool()
        return Agent(
            config=self.agents_config["best_result_shower"],
            verbose=True,
            llm=LLM(model="deepseek/deepseek-chat", api_key=api_key),
            tools=[get_best_results_tool],
        )

    # TASKS
    @task
    def store_structures_in_database(self) -> Task:
        return Task(
            config=self.tasks_config["store_structures_in_database"],
            agent=self.database_storer(),
        )

    @task
    def find_best_conformers(self) -> Task:
        return Task(
            config=self.tasks_config["find_best_conformers"],
            agent=self.best_conformers_finder(),
        )

    @task
    def remove_temporary_files(self) -> Task:
        return Task(
            config=self.tasks_config["remove_temporary_files"],
            agent=self.temporary_files_remover(),
        )

    @task
    def get_best_results(self) -> Task:
        return Task(
            config=self.tasks_config["get_best_results"],
            agent=self.best_result_shower(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Post-Processing Crew"""

        # Build the dynamic list of tasks
        tasks = []
        tasks.append(self.find_best_conformers())

        if self.parsed_query.get("is_store_database", False):
            tasks.append(self.store_structures_in_database())
        if self.parsed_query.get("is_delete_intermediate_files", True):
            tasks.append(self.remove_temporary_files())

        tasks.append(self.get_best_results())

        return Crew(
            agents=self.agents,
            tasks=tasks,  # Pass the dynamic task list
            process=Process.sequential,
            verbose=True,
        )
