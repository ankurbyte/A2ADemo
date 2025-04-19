# A2ADemo
New agents added in Google A2A :
1. CustomerDetails_agent.py : Agent and A2A server built using BytePlus HiAgent APIs. This agent will get customer name and email from e-commerce store based on customer id
2. getorder_agent.py : langgraph agent with call to external e-commerce store to get order details based on customer id
3. data_analysis_agent.py : crewai agent with classes modified to use BytePlus ModelArk LLM APIs instead of using default LLM in A2A package. 
