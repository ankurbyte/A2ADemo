from langchain_core.tools import tool, StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import httpx
from typing import Any, Dict, AsyncIterable, Literal
from pydantic import BaseModel
import os
import logging

memory = MemorySaver()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@tool
def get_shopify_customer(customer_id: str = None) -> dict:
    """Use this to get customer information from Shopify using GraphQL.

    Args:
        customer_id:  Customer ID of the customer.

    Returns:
        A dictionary containing customer data, or an error message if the request fails.
    """    
    try:
        logger.debug("Received customer_id: {customer_id}")
        print(f"---------- Step 1 Received customer_id: {customer_id}")
        formatted_customer_id = f"gid://shopify/Customer/{customer_id}"
        logger.debug(f"--------- Formatted customer_id: {formatted_customer_id}")
        print(f"--------- Step 1 Received formatted customer_id: {formatted_customer_id}")
        # GraphQL query for customer search
        query = """
        query getCustomer($id: ID!) {
          customer(id: $id) {
            id
            firstName
            lastName
            email
            phone
            numberOfOrders
            amountSpent {
              amount
              currencyCode
            }
            createdAt
            defaultAddress {
              formattedArea
              address1
            }
            verifiedEmail
            validEmailAddress
          }
        }
        """
        
        variables = {
            "id": formatted_customer_id
        }

        headers = {
            "X-Shopify-Access-Token": os.getenv("SHOPIFY_ACCESS_TOKEN"),
            "Content-Type": "application/json",
        }

        logger.debug(f"Sending GraphQL query with variables: {variables}")
        print(f"Step 2 Sending GraphQL query with variables: {variables}")


        with httpx.Client() as client:
            response = client.post(
                f"https://{os.getenv('SHOPIFY_SHOP_URL')}/admin/api/2024-01/graphql.json",
                json={"query": query, "variables": variables},
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            logger.debug(f"Received API response: {data}")
            print(f"Received API response: {data}")
            
            if "errors" in data:
                return {"error": data["errors"][0]["message"]}
            
            customer = data.get("data", {}).get("customer")
            if not customer:
                return {"error": "No customer found with the provided ID"}
                
            return {
                "id": customer["id"],
                "email": customer["email"],
                "firstName": customer["firstName"],
                "lastName": customer["lastName"],
                "phone": customer["phone"],
                "numberOfOrders": customer["numberOfOrders"],
                "amountSpent": customer["amountSpent"],
                "address": customer["defaultAddress"],
                "verifiedEmail": customer["verifiedEmail"],
                "validEmailAddress": customer["validEmailAddress"]
            }

    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str

class ShopifyCustomerAgent:
    SYSTEM_INSTRUCTION = (
        "You are a specialized assistant for retrieving Shopify customer information. "
        "Your sole purpose is to use the 'get_shopify_customer' tool to find customer details using the customer id shared by user. "
        "If the user asks about anything other than Shopify customer information, "
        "politely state that you cannot help with that topic. "
        "Always ask for the customer's customer ID if not provided "
        "When presenting customer information, format it clearly including: "
        "- Customer ID\n"
        "- Email (and verification status)\n"
        "- Full Name\n"
        "- Phone\n"
        "- Number of Orders\n"
        "- Amount Spent (with currency)\n"
        "- Default Address\n"
        "Set response status to input_required if you need the customer ID. "
        "Set response status to error if there is an error. "
        "Set response status to completed if the request is complete."
    )
     
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        print(f"Step 1 : reached in customerID_agent.py init")
        self.tools = [get_shopify_customer]

        self.graph = create_react_agent(
            self.model, 
            tools=self.tools, 
            checkpointer=memory, 
            prompt=self.SYSTEM_INSTRUCTION, 
            response_format=ResponseFormat
        )

    def invoke(self, query, sessionId) -> str:
        config = {"configurable": {"thread_id": sessionId}}
        self.graph.invoke({"messages": [("user", query)]}, config)        
        return self.get_agent_response(config)

    async def stream(self, query, sessionId) -> AsyncIterable[Dict[str, Any]]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": sessionId}}

        for item in self.graph.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Looking up customer information...",
                }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing customer data...",
                }            
        
        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config) 
        print(f"-------------- current_state {current_state}")       
        structured_response = current_state.values.get('structured_response')
        print(f"-------------- structured_response {structured_response}")
        if structured_response and isinstance(structured_response, ResponseFormat): 
            if structured_response.status == "input_required":
                print(f"-------------- input_required")
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "error":
                print(f"-------------- error")
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]