import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from mcp import StdioServerParameters
from typing import List


@CrewBase
class FullereneFactoryCrew:
    """FullereneFactory Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    mcp_server_params = [
        StdioServerParameters(
            command="python",
            args=[
                "../mcp_servers/cheminformatics_mcp_server.py",
            ],
            env={"GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")},
        ),
        StdioServerParameters(
            command="python",
            args=[
                "../mcp_servers/computational_mcp_server.py",
            ],
            env={"GEMINI_API_KEY": os.getenv("GEMINI_API_KEY")},
        ),
    ]

    mcp_connect_timeout = 90  # 90 seconds timeout for all MCP connections

    # AGENTS
    @agent
    def query_parser(self) -> Agent:
        return Agent(config=self.agents_config["query_parser"], verbose=True)

    @agent
    def base_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config["base_retrieval"],
            tools=self.get_mcp_tools(),
            verbose=True,
        )

    @agent
    def addend_selector(self) -> Agent:
        return Agent(
            config=self.agents_config["addend_selector"],
            tools=self.get_mcp_tools(),
            verbose=True,
        )

    @agent
    def structure_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["structure_generator"],
            tools=self.get_mcp_tools(),
            verbose=True,
        )

    @agent
    def structure_validator(self) -> Agent:
        return Agent(
            config=self.agents_config["structure_validator"],
            tools=self.get_mcp_tools(),
            verbose=True,
        )

    @agent
    def structure_storer(self) -> Agent:
        return Agent(
            config=self.agents_config["structure_storer"],
            tools=self.get_mcp_tools(),
            verbose=True,
        )

    @agent
    def structure_visualizer(self) -> Agent:
        return Agent(
            config=self.agents_config["structure_visualizer"],
            tools=self.get_mcp_tools(),
            verbose=True,
        )

    # TASKS
    @task
    def parse_query(self) -> Task:
        return Task(
            config=self.tasks_config["parse_query"],
        )

    @task
    def retrieve_base_structure(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_base_structure"],
            context=[self.parse_query()],
        )

    @task
    def select_addend(self) -> Task:
        return Task(
            config=self.tasks_config["select_addend"],
            context=[self.parse_query(), self.retrieve_base_structure()],
        )

    @task
    def generate_structure(self) -> Task:
        return Task(
            config=self.tasks_config["generate_structure"],
            context=[
                self.parse_query(),
                self.retrieve_base_structure(),
                self.select_addend(),
            ],
        )

    @task
    def validate_structure_with_MMFF(self) -> Task:
        return Task(
            config=self.tasks_config["validate_structure_mmff"],
            context=[self.generate_structure()],
        )

    @task
    def validate_structure_with_MLIP(self) -> Task:
        return Task(
            config=self.tasks_config["validate_structure"],
            context=[self.validate_structure_with_MMFF()],
        )

    @task
    def validate_structure_with_DFT(self) -> Task:
        return Task(
            config=self.tasks_config["validate_structure_dft"],
            context=[self.validate_structure_with_MLIP()],
        )

    @task
    def store_structure(self) -> Task:
        return Task(
            config=self.tasks_config["store_structure"],
            context=[self.validate_structure_with_DFT()],
        )

    @task
    def visualize_structure(self) -> Task:
        return Task(
            config=self.tasks_config["visualize_structure"],
            context=[self.validate_structure_with_DFT()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the FullereneFactory Crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
