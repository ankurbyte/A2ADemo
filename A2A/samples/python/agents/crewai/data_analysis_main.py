"""Entry point for the A2A + CrewAI Data Analysis sample."""
# A2A agent for data analysis using BytePlys ModelArk DeepSeek v3 LLM (instead of default LLM used by A2A)
# This agent is designed to analyze data using natural language queries. Agent reads product, order, sales and customer data from shopify store for analysius
# Agent implementation class data_analysis_agent.py
# Agent running on localhost:10007
from data_analysis_agent import DataAnalysisAgent
import click
from common.server import A2AServer
from common.types import AgentCapabilities, AgentCard, AgentSkill, MissingAPIKeyError
import logging
import os
from data_analysis_task_manager import AgentTaskManager
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10007)
def main(host, port):
    """Entry point for the A2A + CrewAI Data Analysis sample."""
    try:
        if not os.getenv("ARK_API_KEY"):
            raise MissingAPIKeyError("ARK_API_KEY environment variable not set.")

        capabilities = AgentCapabilities(streaming=False)
        skill = AgentSkill(
            id="data_analyzer",
            name="Data Analyzer",
            description=(
                "Analyze e-commerce data to provide insights on trends, "
                "performance metrics, and recommendations for improvement."
            ),
            tags=["data analysis", "e-commerce", "business intelligence"],
            examples=[
                "Show me the top-selling products",
                "What are the current sales trends?",
                "Analyze customer purchase patterns",
                "Top 5 customers based on total sales"
            ]
        )

        agent_card = AgentCard(
            name="Data Analysis Agent - ModelArk crewai",
            description=(
                "An intelligent agent that analyzes e-commerce data "
                "and provides actionable insights and recommendations."
            ),
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=DataAnalysisAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=DataAnalysisAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill]
        )

        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=DataAnalysisAgent()),
            host=host,
            port=port
        )
        logger.info(f"Starting server on {host}:{port}")
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)

if __name__ == "__main__":
    main()