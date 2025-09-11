import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from typing import List

# Import the new CrewAI tools
from ..tools.search_pubchem_tool import SearchPubChemTool
from ..tools.download_fullerene_tool import DownloadFullereneTool


@CrewBase
class InitializationCrew:

    def __init__(self, query: str = ""):
        """Initialization Crew"""
        self.query = query

    # AGENTS
    @agent
    def query_parser(self) -> Agent:
        return Agent(
            config=self.agents_config["query_parser"],
            verbose=True,
            llm=LLM(
                model="deepseek/deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY")
            ),
            query=self.query,
        )

    @agent
    def base_retriever(self) -> Agent:
        download_fullerene_tool = DownloadFullereneTool()
        return Agent(
            config=self.agents_config["base_retriever"],
            tools=[download_fullerene_tool],
            verbose=True,
            llm=LLM(
                model="deepseek/deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY")
            ),
        )

    @agent
    def addend_selector(self) -> Agent:
        search_pubchem_tool = SearchPubChemTool()
        return Agent(
            config=self.agents_config["addend_selector"],
            tools=[search_pubchem_tool],
            verbose=True,
            llm=LLM(
                model="deepseek/deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY")
            ),
        )

    # TASKS
    @task
    def parse_query(self) -> Task:
        return Task(
            config=self.tasks_config["parse_query"],
            agent=self.query_parser(),
        )

    @task
    def retrieve_base_structure(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_base_structure"],
            agent=self.base_retriever(),
            context=[self.parse_query()],
        )

    @task
    def select_addend(self) -> Task:
        return Task(
            config=self.tasks_config["select_addend"],
            agent=self.addend_selector(),
            context=[self.parse_query(), self.retrieve_base_structure()],
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
