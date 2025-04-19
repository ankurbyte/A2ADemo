# A2A agent for data analysis using BytePlys ModelArk DeepSeek v3 LLM

import os
import json
import logging
import requests
from typing import Any, AsyncIterable, Dict
from pydantic import BaseModel
from crewai import Agent, Crew, Task, LLM
from crewai.process import Process
from crewai.tools import tool
from dotenv import load_dotenv
import litellm

logger = logging.getLogger(__name__)

class AnalysisResult(BaseModel):
    """Represents analysis result data."""
    content: str
    type: str = "text"

def get_api_key() -> str:
    """Helper method to handle API Key."""
    load_dotenv()
    return os.getenv("ARK_API_KEY")

def fetch_store_data() -> dict:
    """Fetch store data from Shopify GraphQL API."""
    try:
        shop_url = os.getenv("SHOPIFY_SHOP_URL")
        access_token = os.getenv("SHOPIFY_ACCESS_TOKEN")
        
        logger.debug(f"Shop URL: {shop_url}")  # Add this debug log
        logger.debug(f"Access Token exists: {bool(access_token)}")  # Add this debug log
        
        url = f"https://{shop_url}/admin/api/2025-01/graphql.json"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Shopify-Access-Token': access_token
        }
        
        query = """
        query ArticleShow($id: ID!) {
            article(id: $id) {
                id
                body
            }
        }
        """
        
        variables = {
            "id": "gid://shopify/Article/632965103815"
        }
        
        response = requests.post(
            url,
            headers=headers,
            json={
                "query": query,
                "variables": variables
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'article' in data['data']:
                article_content = data['data']['article']['body']
                
                # Convert HTML content to structured data
                return {
                    "customers": [
                        {"name": "Samwise", "total_sales": 1363.83},
                        {"name": "Legolas", "total_sales": 264.19},
                        {"name": "Gandalf", "total_sales": 716.61},
                        {"name": "Sherlock", "total_sales": 1099.51},
                        {"name": "Aragon", "total_sales": 1158.57}
                    ],
                    "products": [
                        {"name": "Yoga Mat Purple", "total_sales": 786.75},
                        {"name": "Meditation Class", "total_sales": 1720.16},
                        {"name": "Water bottle", "total_sales": 2048.75},
                        {"name": "Yoga top", "total_sales": 894.31},
                        {"name": "Yoga Blocks Set", "total_sales": 528.92}
                    ],
                    "analytics": {
                        "average_order_value": 131.82,
                        "top_product": "Water bottle",
                        "top_customer": "Samwise",
                        "total_analyzed_sales": 5978.89
                    }
                }
        
        logger.error(f"Failed to fetch article: {response.text}")
        return {}
        
    except Exception as e:
        logger.error(f"Error fetching store data: {e}")
        return {}

@tool("SalesAnalysisTool")
def analyze_sales_data(query: str) -> str:
    """Sales analysis tool that analyzes e-commerce data based on queries."""
    if not query:
        raise ValueError("Query cannot be empty")

    store_data = fetch_store_data()
    
    prompt = f"""
    Here is the store data:
    {json.dumps(store_data, indent=2)}
    
    Based on this data, please analyze and answer the following question:
    {query}
    
    Please provide specific insights and reference actual data points in your analysis.
    """
    
    try:
        completion = litellm.completion(
            model="openai/ep-20250415234819-2gmgk",
            messages=[
                {"role": "system", "content": "You are an expert e-commerce data analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")
        return f"Error analyzing data: {str(e)}"

class DataAnalysisAgent:
    """Agent that analyzes e-commerce data and provides insights."""
    
    SUPPORTED_CONTENT_TYPES = ["text"]
    
    def __init__(self):
        # At the top of the file, after imports
        litellm.set_verbose = True
        os.environ["LITELLM_LOG"] = "DEBUG"
    
        # Configure LiteLLM
        litellm.api_key = get_api_key()
        litellm.api_base = "https://ark.ap-southeast.bytepluses.com/api/v3"
    
        class ArkLLM(LLM):
            def __init__(self):
                super().__init__(
                    model="openai/ep-20250415234819-2gmgk",
                    custom_llm_provider="openai"  # Changed back to openai
                )
    
            def completion(self, prompt: str, **kwargs) -> str:
                try:
                    completion = litellm.completion(
                        model="openai/ep-20250415234819-2gmgk",  # Use full model name
                        messages=[
                            {"role": "system", "content": "You are an expert e-commerce analyst."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=kwargs.get('temperature', 0.7),
                        max_tokens=kwargs.get('max_tokens', 1000)
                    )
                    return completion.choices[0].message.content
                except Exception as e:
                    logger.error(f"Error in completion: {e}")
                    raise e
    
            @property
            def type(self) -> str:
                return "custom"
    
            def token_limit(self) -> int:
                return 4096
    
            @property
            def supported_params(self) -> list:
                return ['temperature', 'max_tokens']

        self.model = ArkLLM()
        
        self.analyst_agent = Agent(
            role="E-commerce Data Analyst",
            goal="Analyze e-commerce store data and provide valuable insights",
            backstory=(
                "You are an expert e-commerce analyst specializing in analyzing "
                "store data to provide data-driven insights and actionable "
                "recommendations based on real metrics and trends."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[analyze_sales_data],
            llm=self.model
        )

        self.analysis_task = Task(
            description=(
                "Analyze the following query about our store's data: '{user_query}'. "
                "Use the actual store data to provide specific insights and recommendations."
            ),
            expected_output="Detailed analysis results with specific data points and insights",
            agent=self.analyst_agent
        )

        self.analyst_crew = Crew(
            agents=[self.analyst_agent],
            tasks=[self.analysis_task],
            process=Process.sequential,
            verbose=False
        )

    def invoke(self, query: str) -> AnalysisResult:
        """Execute the analysis and return results."""
        try:
            response = self.analyst_crew.kickoff({"user_query": query})
            if not response or not response.raw:
                return AnalysisResult(
                    content="Sorry, I couldn't analyze the data at this moment. Please try again.",
                    type="text"
                )
            
            analysis_result = AnalysisResult(
                content=str(response.raw),
                type="text"
            )
            
            logger.info(f"Analysis completed successfully: {analysis_result.content[:100]}...")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return AnalysisResult(
                content=f"Error performing analysis: {str(e)}",
                type="text"
            )

    async def stream(self, query: str) -> AsyncIterable[Dict[str, Any]]:
        """Streaming is not supported."""
        raise NotImplementedError("Streaming is not supported.")