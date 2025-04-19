#Custom A2A server to demonstrate the langgraph agent with order details read from shopify store
#         name="Order Details Agent - langgraph",
#         description="Helps retrieve order information",
#         name="Order Details Agent - langgraph",
# Agent implementation class getorder_agent.py
# Agent running on localhost:10006
from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from common.utils.push_notification_auth import PushNotificationSenderAuth
from agents.langgraph.task_manager import AgentTaskManager
from agents.langgraph.getorder_agent import SaralOrderAgent
import click
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10006)
def main(host, port):
    """Starts the Saral Order Agent server."""
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")
        if not os.getenv("SHOPIFY_ACCESS_TOKEN"):
            raise MissingAPIKeyError("SHOPIFY_ACCESS_TOKEN environment variable not set.")

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id="get_order_info",
            name="Saral Store Order Information Tool",
            description="Retrieves order information from saral online store",
            tags=["saral", "order", "orderID", "Amount", "items", "customerID", "orderID", "totalAmount", "totalItems", "lastOrders", "last7DaysOrders", "last7DaysAmount", "last7DaysItems"],
            examples=["Find the number of orders placed by the customer with id 1234567890", "Find the last two orders placed by the customer with id 1234567890", " What items customer with id 1234567890 has ordered in the last 7 days?", "What is the total amount spent by the customer with id 1234567890?", "What is the total amount spent by the customer with id 1234567890 in the last 7 days?"],
        )
        agent_card = AgentCard(
            name="Order Details Agent - langgraph",
            description="Helps retrieve order information",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=SaralOrderAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=SaralOrderAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=SaralOrderAgent(), 
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