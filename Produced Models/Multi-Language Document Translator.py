"""
This Model was build entirely using the model builder, the only edits made were removing any redundant tools


A  multi-agent system that translates documents into multiple languages while maintaining context and cultural relevance
"""

import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, FileReadTool

# Set environment variables
openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

file_read_tool = FileReadTool()
serper_dev_tool = SerperDevTool()

# Agent 1: Language Detector and Source Text Preparation Agent
detector = Agent(
    role="Language Detector and Source Text Preparation Agent",
    goal="Analyze the input document to determine the source language. Prepare the source text for translation by removing any formatting or special characters that may interfere with the translation process.",
    tools=[file_read_tool],
    verbose=True,
    backstory="As an AI Language Specialist, your mastery in analyzing and preprocessing text is what sets you apart. You possess an in-depth understanding of different languages and their nuances, enabling you to swiftly determine the source language of any given document. Your expertise extends to preparing the text for translation, meticulously removing any formatting or special characters that could hinder the translation process. With your precision and efficiency, you lay the groundwork for a seamless translation journey.",
    allow_delegation=False
)

# Task: Detect language and prepare source text
detect_and_prepare = Task(
    description=(
        "1. Analyze the input document: {Document} to determine the source language.\n"
        "2. Prepare the source text for translation by removing any formatting or special characters that may interfere with the translation process.\n"
        "3. Apply preprocessing techniques such as text normalization, tokenization, and stop-word removal.\n"
        "4. Identify any potential language-specific challenges or nuances that should be addressed during translation, and flag these for the Machine Translation and Cultural Adaptation Agents to consider while performing their work"
    ),
    expected_output="Source text that has been properly formatted, cleaned of unnecessary elements and is ready to be translated.",
    agent=detector,
)


# Agent 2: Machine Translation Agent
translator = Agent(
    role="Machine Translation Agent",
    goal="Perform the initial machine translation of the source text into the target language using a pre-trained machine translation model. Generate a draft translation that preserves the general meaning and structure of the original text.",
    tools=[serper_dev_tool],
    verbose=True,
    backstory="As an AI Translation Engine, your fluency in multiple languages and your prowess in computational linguistics make you a formidable force. You excel at harnessing the power of pre-trained machine translation models to produce draft translations that accurately convey the essence of the original text. Your translations are not just word-for-word conversions; you capture the nuances and context, ensuring that the target text retains the intended meaning and structure.",
    allow_delegation=False
)

# Task: Machine translate source text
machine_translate = Task(
    description=(
        "1. Translate the source text into the target language: {target_language} using pre-trained machine translation models.\n"
        "2. Output a 'raw translation' which has the general structure and meaning of the original text, but potentially with some inaccuracies or cultural inappropriacies that need further refinement."
    ),
    expected_output="Draft MT output in the target language.",
    agent=translator,
)


# Agent 3: Cultural Adaptation Agent
adapter = Agent(
    role="Cultural Adaptation Agent",
    goal="Analyze the draft translation to identify any cultural nuances or references that may not be easily understood by the target audience. Adapt the translation to ensure that it is culturally appropriate and relevant to the target culture.",
    tools=[],
    verbose=True,
    backstory="As an AI Cultural Navigator, your deep understanding of diverse cultures and your ability to bridge linguistic and cultural gaps are invaluable. You are adept at identifying cultural nuances and references that may elude others, ensuring that the translated text resonates with the target audience. Your cultural adaptation skills guarantee that the translation is not just accurate but also culturally appropriate, respecting the sensibilities and values of the target culture.",
    allow_delegation=False
)

# Task: Culturally adapt machine translation
cultural_adapt = Task(
    description=(
        "1. Analyze draft translation to identify cultural references or expressions.\n"
        "2. Collaborate closely with Agent 4, the human post-editor to ensure cultural accuracy."
    ),
    expected_output="Culturally adapted and refined draft translation that accurately reflects the intended message.",
    agent=adapter,
    context=[machine_translate],
)


# Agent 4: Human Post-Editor
editor = Agent(
    role="Human Post-Editor",
    goal="Review the culturally adapted translation and make any necessary corrections or improvements to ensure accuracy, fluency, and adherence to the original meaning. Collaborate with the Cultural Adaptation Agent to resolve any remaining cultural or linguistic issues.",
    tools=[],
    verbose=True,
    backstory="As an AI Language Craftsperson, your expertise in linguistics and your keen eye for detail make you the final arbiter of translation quality. You meticulously review the culturally adapted translation, ensuring that it is accurate, fluent, and true to the original intent. Your collaboration with the Cultural Adaptation Agent ensures that any remaining cultural or linguistic issues are resolved, resulting in a translation that is both precise and culturally sensitive.",
    allow_delegation=False,
)

# Task: Post-edit culturally adapted translation
post_edit = Task(
    description=(
        "1. Review and refine the culturally adapted draft translation given by the adapter to check accuracy and fluency, as well as to make sure it conveys the meaning and tone of the original message\n"
        "2. Ensure the translated text adheres to any specific formatting and style requirements. \n"
    ),
    expected_output="A polished and accurate translation in the target language which is fluently written, stylistically appropriate and ready for use.",
    agent=editor,
    context=[detect_and_prepare, machine_translate, cultural_adapt],
)



# Create the Crew
crew = Crew(
    agents=[detector, translator, adapter, editor],
    tasks=[detect_and_prepare, machine_translate, cultural_adapt, post_edit],
    verbose=True,
)


# Define the input document and target language
Document = "upload your file"
target_language = "Arabic"


# Run the Crew with the extracted text
result = crew.kickoff(inputs={"Document": Document, "target_language": target_language})

# Print the result
print(result)
