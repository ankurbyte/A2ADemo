import httpx
from typing import Any, Dict, AsyncIterable
import os
import json
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from common.types import (
    TaskStatus,
    TaskState,
    Message,
    Artifact
)

logger = logging.getLogger(__name__)

class HiAgentCustomerAgent:
    """Agent that retrieves customer details using HiAgent APIs"""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
    
    def __init__(self):
        self.api_key = os.getenv("HIAGENT_API_KEY")
        if not self.api_key:
            raise ValueError("HIAGENT_API_KEY environment variable not set")
        self.base_url = "https://hiagent-byteplus.volcenginepaas.com/api/proxy/api/v1"
        self.app_id = "d01928p4n9qpbu7sdv80"
        self.user_id = "123"

    async def _run_workflow(self, customer_id: str) -> Dict:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/run_app_workflow",
                    headers={
                        "Apikey": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "UserID": self.user_id,
                        "AppID": self.app_id,
                        "InputData": json.dumps({
                            "user_input": customer_id  # Remove the data wrapper, keep only user_input
                        })
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Workflow API Response: {data}")
                return data
        except httpx.TimeoutError:
            logger.error("Workflow API request timed out")
            raise
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during workflow: {e.response.text if hasattr(e, 'response') else str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error running workflow: {e}")
            raise

    def _extract_customer_details(self, process_data: Dict) -> Dict:
        """Extracts customer details from the process response"""
        try:
            if not process_data or not isinstance(process_data, dict):
                logger.error(f"Invalid process data format: {process_data}")
                return None

            nodes = process_data.get("nodes", {})
            if not nodes:
                logger.error("No nodes found in process data")
                return None

            # Log the available nodes for debugging
            logger.debug(f"Available nodes: {list(nodes.keys())}")
            
            for node_id, node in nodes.items():
                logger.debug(f"Processing node {node_id}: {node.get('nodeType')}")
                
                if node.get("nodeType") == "end":
                    output_str = node.get("output", "{}")
                    output_data = json.loads(output_str)
                    if "output" in output_data:
                        return output_data["output"]
                    
                elif node.get("nodeType") == "http_request":
                    output_str = node.get("output", "{}")
                    output_data = json.loads(output_str)
                    if "data" in output_data and "customer" in output_data["data"]:
                        return output_data["data"]["customer"]
            
            logger.error("No valid customer data found in nodes")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting customer details: {e}")
            return None

    async def _query_process(self, run_id: str) -> Dict:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/query_run_app_process",
                    headers={
                        "Apikey": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "UserID": self.user_id,
                        "AppID": self.app_id,
                        "RunID": run_id
                    },
                    timeout=30.0  # Add timeout
                )
                response.raise_for_status()
                process_data = response.json()
                
                # Validate response structure
                if not isinstance(process_data, dict):
                    raise ValueError(f"Invalid response format: {process_data}")
                
                if "status" not in process_data:
                    raise ValueError(f"Missing status in response: {process_data}")
                    
                logger.debug(f"Process API Response: {process_data}")  # Change to debug level
                return process_data
            except (httpx.TimeoutError, httpx.HTTPError) as e:
                error_detail = e.response.text if hasattr(e, 'response') else str(e)
                logger.error(f"API error during process query: {error_detail}")
                raise

    async def _wait_for_process(self, run_id: str, max_retries: int = 5) -> Dict:
        """Wait for process to complete with retries"""
        import asyncio
        
        for attempt in range(max_retries):
            process_data = await self._query_process(run_id)
            if process_data["status"] == "success":
                return process_data
            elif process_data["status"] == "processing":
                logger.info(f"Process still running, attempt {attempt + 1}/{max_retries}")
                await asyncio.sleep(2)  # Wait 2 seconds before retry
            else:
                raise ValueError(f"Process failed with status: {process_data['status']}")
        
        raise TimeoutError("Process did not complete within the expected time")

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        try:
            # Extract customer ID from query - improved parsing
            query = query.strip()
            # Extract numbers from the query
            customer_id = ''.join(filter(str.isdigit, query))
            
            logger.info(f"Received query: '{query}'")
            logger.info(f"Extracted customer ID: '{customer_id}'")

            if not customer_id:
                yield {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": "Please provide a valid customer ID (numbers only)."
                }
                return

            yield {
                "is_task_complete": False,
                "content": f"Looking up details for customer ID: {customer_id}..."
            }

            # Step 1: Run workflow
            workflow_response = await self._run_workflow(customer_id)
            logger.info(f"Workflow response: {workflow_response}")
            
            # Fix: Change RunID to runId to match API response
            run_id = workflow_response.get('runId')  # Changed from 'RunID' to 'runId'
            
            if not run_id:
                logger.error(f"Invalid workflow response: {workflow_response}")
                yield {
                    "is_task_complete": True,
                    "content": "Failed to initiate workflow. Invalid response from server."
                }
                return

            yield {
                "is_task_complete": False,
                "content": "Processing customer information..."
            }

            # Step 2: Query process with retry
            try:
                process_data = await self._wait_for_process(run_id)
            except (TimeoutError, ValueError) as e:
                logger.error(f"Process error: {e}")
                yield {
                    "is_task_complete": True,
                    "content": f"Failed to retrieve customer details: {str(e)}"
                }
                return

            # Extract customer details
            customer_details = self._extract_customer_details(process_data)
            if not customer_details:
                yield {
                    "is_task_complete": True,
                    "content": "No customer details found."
                }
                return

            # Format response
            response = (
                f"Customer Details:\n"
                f"First Name: {customer_details['firstName']}\n"
                f"Last Name: {customer_details['lastName']}\n"
                f"Email: {customer_details['email']}"
            )

            yield {
                "is_task_complete": True,
                "content": response
            }

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            yield {
                "is_task_complete": True,
                "content": "An error occurred while processing your request. Please try again."
            }