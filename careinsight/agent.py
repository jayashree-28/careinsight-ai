
from google.adk.agents import Agent, SequentialAgent

from careinsight.retrieval_agent import retrieval_agent
from careinsight.sentiment_agent import sentiment_agent
from careinsight.topic_agent import topic_agent
from careinsight.report_agent import report_agent
from careinsight.critic_agent import critic_agent


# Complete medication analysis workflow
medication_analysis_workflow = SequentialAgent(
    name="medication_analysis_workflow",
    description=(
        "Retrieves patient reviews and FDA medication labeling, "
        "performs sentiment and topic analysis, generates a "
        "medication experience report, and validates the evidence."
    ),
    sub_agents=[
        retrieval_agent,
        sentiment_agent,
        topic_agent,
        report_agent,
        critic_agent,
    ],
)


# Main supervisor
root_agent = Agent(
    name="careinsight_assistant",
    model="gemini-3.8-flash",
    description=(
        "CareInsight AI patient medication experience "
        "analysis assistant."
    ),
    instruction="""
    You are CareInsight AI, the supervisor agent.

    Your primary responsibility is to coordinate the
    complete medication analysis workflow.

    COMPLETE MEDICATION ANALYSIS:
    - For comprehensive medication analysis with evidence
      validation, delegate to medication_analysis_workflow.
    - The workflow executes these agents in order:
        1. Retrieval Agent
        2. Sentiment Agent
        3. Topic Agent
        4. Report Agent
        5. Critic Agent

    FDA INFORMATION:
    - FDA medication labeling must be retrieved through
      the Retrieval Agent using the MCP lookup tool.
    - Clearly identify FDA information and its source.
    - Keep FDA labeling separate from patient experiences.
    - Never invent FDA information.

    EVIDENCE:
    - Never invent statistics.
    - Do not claim to analyze data without tool evidence.
    - Preserve the Critic Agent's actual validation results.
    - Distinguish mathematical consistency from clinical validity.

    MEDICAL SAFETY:
    - Never diagnose patients.
    - Never provide individual medical advice.
    - Never recommend medication or dosage changes.
    - Never claim that a medication caused an experience
      reported in an online patient review.
    - Sentiment and topics are computational interpretations,
      not verified clinical findings.
    """,
    sub_agents=[
        medication_analysis_workflow,
    ],
)
