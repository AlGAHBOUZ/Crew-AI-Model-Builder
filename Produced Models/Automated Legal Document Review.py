"""
This Model was build entirely using the model builder, the only edits made were removing any redundant tools

A multi-agent system that automates the review of legal documents, ensuring compliance, identifying risks, and suggesting improvements
"""

import os
from crewai import Agent, Task, Crew
from utils import get_openai_api_key, get_serper_api_key

from crewai_tools import (
              ScrapeWebsiteTool,
              SerperDevTool
          )


openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["SERPER_API_KEY"] = get_serper_api_key()

# Agent 1: Legal Document Analyzer
legal_analyzer = Agent(
    role="Legal Document Analyzer",
    goal="Extract and categorize clauses, assess potential legal risks, and check for regulatory compliance, ensuring the document adheres to all relevant laws and regulations.",
    tools=[
        ScrapeWebsiteTool(
            url="https://www.law.cornell.edu/wex/contract",
            extract="""
            [A contract is] an agreement between private parties creating mutual obligations enforceable by law.
            """
        ),
        SerperDevTool(
            query="What are the elements of a contract",
            extract="""
            - Offer
            - Acceptance
            - Consideration
            - Capacity
            - Legality
            """
        ),
    ],
    verbose=True,
    backstory=(
        "You are a seasoned Legal Document Analyst, with unmatched skills "
        "in navigating and extracting critical information on the topics currently being researched at Valeo ('https://www.law.cornell.edu/wex/contract')"
    ),
    allow_delegation=False
)

# Task 1: Extract Document Clauses
extract_clauses = Task(
    description=(
        "1. Extract all clauses, articles, or sections from the input document.\n"
        "2. Categorize each clause according to pre-defined legal categories (e.g., contract terms, warranties, disclaimers, etc.).\n"
    ),
    expected_output="A structured representation of the document's clauses, categorized and annotated with identified legal risks and compliance assessments.",
    agent=legal_analyzer
)

# Agent 2: Risk Assessor
risk_assessor = Agent(
    role="Risk Assessor",
    goal="Identify and evaluate potential legal risks associated with a document, considering both internal and external factors, providing actionable recommendations to mitigate risks and enhance its legal standing.",
    tools=[
        ScrapeWebsiteTool(
            url="https://www.investopedia.com/terms/l/legalrisk.asp",
            extract="""
            Legal risk refers to the possibility that a company or individual could be held legally liable for its actions or decisions.
            """
        ),
        SerperDevTool(
            query="What is the difference between tort and contract",
            extract="""
            - A tort is a civil wrong that results in injury to another person or their property.
            - A contract is a legally binding agreement between two or more parties.
            """
        ),
    ],
    verbose=True,
    backstory=(
        "You are a highly skilled Risk Assessor, your ability to anticipate and evaluate potential legal challenges is exceptional. You possess a comprehensive understanding of legal principles and industry best practices, enabling you to assess documents with a critical eye. Your keen insights allow you to identify potential vulnerabilities and develop strategic recommendations to minimize risks and strengthen the document's legal position."
    ),
    allow_delegation=False
)

# Task 2: Assess Legal Risks
assess_risks = Task(
    description=(
        "1. Analyze the extracted clauses and identified legal risks from Agent 1.\n"
        "2. Identify and evaluate potential internal and external factors that may impact the legal risks of the document.\n"
        "3. Determine the severity and likelihood of each legal risk and assess its potential impact on the document's legal standing.\n"
    ),
    expected_output="A comprehensive report detailing the risk assessment, including the identified legal risks, their severity and likelihood, and their potential impact on the document's legal standing.",
    agent=risk_assessor,
    context=[extract_clauses],
)

# Agent 3: Recommendation Generator
recommendation_generator = Agent(
    role="Recommendation Generator",
    goal="Generate specific, actionable recommendations to address identified legal risks and improve a document's legal standing, providing clear and concise guidance to users on how to enhance its legal effectiveness.",
    tools=[
        ScrapeWebsiteTool(
            url="https://www.wikihow.com/Write-a-Legal-Recommendation",
            extract="""
            1. Identify the legal issue.
            2. Research the law.
            3. Analyze the facts.
            4. Develop a legal argument.
            5. Write a recommendation.
            """
        ),
        SerperDevTool(
            query="What are the elements of a legal recommendation",
            extract="""
            - A statement of the legal issue
            - A summary of the relevant law
            - An analysis of the facts
            - A legal argument
            - A recommendation
            """
        ),
    ],
    verbose=True,
    backstory=(
        "Your exceptional talent as a Recommendation Generator lies in your ability to craft tailored solutions to complex legal challenges. You possess a deep understanding of legal frameworks and best practices, which empowers you to develop practical recommendations that effectively mitigate risks and improve the document's legal standing. Your clear and concise guidance enables users to confidently implement your suggestions, enhancing the document's overall legal effectiveness."
    ),
    allow_delegation=False
)

# Task 3: Generate Recommendations
generate_recommendations = Task(
    description=(
        "1. Review the legal risk report generated by Agent 2.\n"
        "2. Develop specific and actionable recommendations to mitigate the identified legal risks.\n"
        "3. Ensure that the recommendations align with the legal requirements of the document and its intended purpose.\n"
    ),
    expected_output="A list of clear and concise recommendations on how to address identified legal risks and improve the document's legal effectiveness.",
    agent=recommendation_generator,
    context=[assess_risks],
)

# Agent 4: Document Finalizer
document_finalizer = Agent(
    role="Document Finalizer",
    goal="Incorporate recommendations from other agents into the document, ensuring that the final output is legally sound and compliant, producing a thoroughly reviewed legal document that is ready for use with minimized risks and full compliance with relevant laws.",
    tools=[
        ScrapeWebsiteTool(
            url="https://www.law.cornell.edu/wex/legal_document",
            extract="""
            A legal document is a written record of a legal transaction or agreement.
            """
        ),
        SerperDevTool(
            query="What are the essential elements of a legal document",
            extract="""
            - The names of the parties involved
            - The date of the agreement
            - The terms of the agreement
            - The signatures of the parties
            """
        ),
    ],
    verbose=True,
    backstory=(
        "You are an experienced Document Finalizer with a keen eye for detail and a deep understanding of legal documentation. Your role is crucial in ensuring that all the recommendations from the previous agents are accurately integrated into the document, resulting in a finalized version that is both legally sound and ready for use. Your expertise ensures that the document adheres to all legal requirements and is free from risks."
    ),
    allow_delegation=False
)

# Task 4: Finalize Document
finalize_document = Task(
    description=(
        "1. Review and analyze the input document, the legal document analysis report generated by Agent 1, the risk assessment report generated by Agent 2, and the recommendations provided by Agent 3.\n"
        "2. Incorporate the recommendations into the input document, ensuring accuracy, consistency, and compliance with legal requirements.\n"
        "3. Produce a polished and finalized legal document that minimizes risks and fully complies with relevant laws and regulations.\n"
    ),
    expected_output="A finalized legal document that is legally sound, minimizes risks, and is ready for use.",
    agent=document_finalizer,
    context=[generate_recommendations, assess_risks, extract_clauses],
)

# Create the Crew instance
crew = Crew(
    agents=[legal_analyzer, risk_assessor, recommendation_generator, document_finalizer],
    tasks=[extract_clauses, assess_risks, generate_recommendations, finalize_document]
)
input_document="path/to/input/legal_document.docx"
# Run the Crew instance
extract_clauses.description = f"Here is the legal doucment: {input_document}\n"

result = crew.kickoff()

print(result)
