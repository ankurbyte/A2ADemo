from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage
import httpx
from typing import Any, Dict, AsyncIterable, Literal
from pydantic import BaseModel
from agents.langgraph.deepseek_wrapper import DeepSeekChat
import os

memory = MemorySaver()

@tool
async def get_shopify_customer(email: str = None):
    """Use this to get customer information from Shopify using GraphQL.

    Args:
        email: Customer's email address

    Returns:
        A dictionary containing customer data, or an error message if the request fails.
    """    
    try:
        # GraphQL query for customer search
        query = """
        query getCustomerByEmail($query: String!) {
          customers(first: 1, query: $query) {
            edges {
              node {
                id
                email
                firstName
                lastName
                phone
                ordersCount
                totalSpent
              }
            }
          }
        }
        """
        
        variables = {
            "query": f"email:{email}" if email else ""
        }

        headers = {
            "X-Shopify-Access-Token": os.getenv("SHOPIFY_ACCESS_TOKEN"),
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://{os.getenv('SHOPIFY_SHOP_URL')}/admin/api/2024-01/graphql.json",
                json={"query": query, "variables": variables},
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            if "errors" in data:
                return {"error": data["errors"][0]["message"]}
            
            customers = data.get("data", {}).get("customers", {}).get("edges", [])
            if not customers:
                return {"error": "No customer found with the provided email"}
                
            customer = customers[0]["node"]
            return {
                "id": customer["id"],
                "email": customer["email"],
                "firstName": customer["firstName"],
                "lastName": customer["lastName"],
                "phone": customer["phone"],
                "ordersCount": customer["ordersCount"],
                "totalSpent": customer["totalSpent"]
            }

    except Exception as e:
        return {"error": f"API request failed: {str(e)}"}

@tool
async def get_shopify_customer(customer_id: str = None):
    """Use this to get customer information from Shopify using GraphQL.

    Args:
        customer_id: Shopify Customer ID (gid://shopify/Customer/XXXXX format)

    Returns:
        A dictionary containing customer data, or an error message if the request fails.
    """    
    try:
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
            "id": customer_id
        }

        headers = {
            "X-Shopify-Access-Token": os.getenv("SHOPIFY_ACCESS_TOKEN"),
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://{os.getenv('SHOPIFY_SHOP_URL')}/admin/api/2024-01/graphql.json",
                json={"query": query, "variables": variables},
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
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

# Update the system instruction to match the new query parameters
class ShopifyCustomerAgent:
    SYSTEM_INSTRUCTION = (
        "You are a specialized assistant for retrieving Shopify customer information. "
        "Your sole purpose is to use the 'get_shopify_customer' tool to find customer details using their Shopify ID. "
        "If the user asks about anything other than Shopify customer information, "
        "politely state that you cannot help with that topic. "
        "Always ask for the customer's Shopify ID if not provided (format: gid://shopify/Customer/XXXXX). "
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

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str

class ShopifyCustomerAgent:
    SYSTEM_INSTRUCTION = (
        "You are a specialized assistant for retrieving Shopify customer information. "
        "Your sole purpose is to use the 'get_shopify_customer' tool to find customer details using their email. "
        "If the user asks about anything other than Shopify customer information, "
        "politely state that you cannot help with that topic. "
        "Always ask for the customer's email if not provided. "
        "When presenting customer information, format it clearly including: "
        "- Customer ID\n"
        "- Email\n"
        "- Full Name (if available)\n"
        "- Phone (if available)\n"
        "- Number of Orders\n"
        "- Total Spent\n"
        "Set response status to input_required if you need the email address. "
        "Set response status to error if there is an error. "
        "Set response status to completed if the request is complete."
    )
     
    def __init__(self):
        self.model = DeepSeekChat(api_key=os.getenv("DEEPSEEK_API_KEY"))
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
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(structured_response, ResponseFormat): 
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "error":
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