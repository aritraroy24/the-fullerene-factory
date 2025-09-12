import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

# Import the new CrewAI tools
from ..tools.structure_generator_tool import StructureGenerationTool
from ..tools.run_mlip_tool import RunMLIPTool
from ..tools.best_conformation_finder_tool import LowestEnergyStructuresTool

api_key = os.getenv("DEEPSEEK_API_KEY")


@CrewBase
class ConformerCrew:
    """Conformer Crew"""

    def __init__(
        self,
        parsed_query: dict = None,
        current_addend: str = None,
        current_fullerene: str = None,
        current_step: int = 1,
    ):
        self.parsed_query = parsed_query
        self.current_addend = current_addend
        self.current_fullerene = current_fullerene
        self.current_step = current_step

    # AGENTS
    @agent
    def conformer_generator(self) -> Agent:
        structure_generator_tool = StructureGenerationTool()
        print(f"Current Fullerene in Agent: {self.current_fullerene}")
        print(f"Current Addend in Agent: {self.current_addend}")
        print(f"Current Step in Agent: {self.current_step}")
        print(f"Number of Angles in Agent: {self.parsed_query.get('num_angles', 5)}")
        print("\n")
        return Agent(
            config=self.agents_config["conformer_generator"],
            verbose=True,
            llm=LLM(model="deepseek/deepseek-chat", api_key=api_key),
            num_angles=self.parsed_query.get("num_angles", 5),
            current_addend=self.current_addend,
            current_fullerene=self.current_fullerene,
            current_step=self.current_step,
            tools=[structure_generator_tool],
        )

    @agent
    def optimization_runner(self) -> Agent:
        optimization_runner_tool = RunMLIPTool()
        return Agent(
            config=self.agents_config["optimization_runner"],
            verbose=True,
            llm=LLM(model="deepseek/deepseek-chat", api_key=api_key),
            tools=[optimization_runner_tool],
        )

    @agent
    def best_conformer_finder(self) -> Agent:
        best_conformer_finder_tool = LowestEnergyStructuresTool()
        return Agent(
            config=self.agents_config["best_conformer_finder"],
            verbose=True,
            llm=LLM(model="deepseek/deepseek-chat", api_key=api_key),
            tools=[best_conformer_finder_tool],
        )

    # TASKS
    @task
    def generate_conformers(self) -> Task:
        return Task(
            config=self.tasks_config["generate_conformers"],
            agent=self.conformer_generator(),
        )

    @task
    def run_optimization(self) -> Task:
        return Task(
            config=self.tasks_config["run_optimization"],
            agent=self.optimization_runner(),
        )

    @task
    def find_best_conformer(self) -> Task:
        return Task(
            config=self.tasks_config["find_best_conformer"],
            agent=self.best_conformer_finder(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Initialization Crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
