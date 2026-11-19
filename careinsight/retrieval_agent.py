
from pathlib import Path
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from mcp import StdioServerParameters
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from careinsight.retrieval import search_medication_reviews

# Locate the CareInsight project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Connect to the local FDA medication information MCP server.
medication_mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=[
                "run",
                "python",
                "-m",
                "careinsight.medication_mcp_server",
            ],
            cwd=str(PROJECT_ROOT),
        ),
    ),
)


retrieval_agent = Agent(
    name="retrieval_agent",
    model="gemini-3.8-flash",

    description=(
        "Retrieves patient medication reviews, rating statistics, "
        "and FDA medication labeling information."
    ),

    instruction="""
    You are the CareInsight Retrieval Agent.

    Your responsibility is to retrieve patient medication
    reviews and relevant FDA medication labeling information.

    TOOLS:

    1. search_medication_reviews:
       Use this tool to retrieve patient reviews, review counts,
       average ratings, and rating distributions.

    2. lookup_medication_info:
       Use this MCP tool to retrieve FDA medication labeling
       information, including warnings and adverse reactions.

    RULES:
    - Never invent review counts or statistics.
    - Report the number of reviews returned by the review tool.
    - Include average ratings and rating distributions when available.
    - Provide supporting patient review examples.
    - Do not interpret ratings as sentiment classifications.
    - Do not provide medical advice or treatment recommendations.
    - If a medication is not found, report that clearly.
    - Distinguish FDA labeling information from patient-reported
      experiences.
    - Never claim that FDA labeling proves a medication caused
      a patient's reported experience.
    """,

    tools=[
        search_medication_reviews,
        medication_mcp_toolset,
    ],
)
