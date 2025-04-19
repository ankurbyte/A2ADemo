#Custom A2A Server built using BytePlus HiAgent APIs
# A workflow agent that retrieves customer details using HiAgent APIs
# agent implementation class CustomerDetails_agent.py
#agent running on port 10005, access url localhost:10005
import sys
import os
import click
import logging
from dotenv import load_dotenv

# Add the parent directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Update these imports to match directory structure
from common.server.server import A2AServer  # server is in server subdirectory
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError  # types.py is direct
from agents.hiagent.task_manager import AgentTaskManager
from agents.hiagent.CustomerDetails_agent import HiAgentCustomerAgent

load_dotenv()

# Change logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10005)
def main(host, port):
    """Starts the HiAgent Customer Details Agent server."""
    try:
        if not os.getenv("HIAGENT_API_KEY"):
            raise MissingAPIKeyError("HIAGENT_API_KEY environment variable not set.")

        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id="get_customer_details",
            name="Customer Details Tool",
            description="Retrieves customer details using customer ID",
            tags=["customer", "details", "hiagent"],
            examples=[
                "6130165088455",
                "Find customer details for ID 6130165088455"
            ]
        )

        agent_card = AgentCard(
            name="Customer Details Agent - BytePlus HiAgent",
            description="Retrieves customer details using HiAgent APIs",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=HiAgentCustomerAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=HiAgentCustomerAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill]
        )

        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=HiAgentCustomerAgent()),
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