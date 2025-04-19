from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from common.utils.push_notification_auth import PushNotificationSenderAuth
from agents.langgraph.task_manager import AgentTaskManager
from agents.langgraph.customerID_agent import ShopifyCustomerAgent
import click
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10005)
def main(host, port):
    """Starts the Shopify Customer Agent server."""
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")
        if not os.getenv("SHOPIFY_ACCESS_TOKEN"):
            raise MissingAPIKeyError("SHOPIFY_ACCESS_TOKEN environment variable not set.")

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id="get_customer_info",
            name="Saral Customer Information Tool",
            description="Retrieves customer information from saral store",
            tags=["saral", "customer", "name", "email", "address"],
            examples=["Find email of the customer with id 1234567890", "Get customer name with id 1234567890", "Find address of the customer with id 1234567890 "],
        )
        agent_card = AgentCard(
            name="Saral Store Customer Agent",
            description="Helps retrieve Shopify customer information",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=ShopifyCustomerAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=ShopifyCustomerAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=ShopifyCustomerAgent(), 
                notification_sender_auth=notification_sender_auth
            ),
            host=host,
            port=port,
        )

        # Add debug logging
        logger.info(f"Agent Card: {agent_card.dict()}")
        logger.info(f"Server routes: {[route.path for route in server.app.routes]}")

       

        server.app.add_route(
            "/.well-known/jwks.json", 
            notification_sender_auth.handle_jwks_endpoint, 
            methods=["GET"]
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