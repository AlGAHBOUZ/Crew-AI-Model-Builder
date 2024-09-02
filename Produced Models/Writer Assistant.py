"""
This Model was build entirely using the model builder, the only edits made were removing redundant tools and building a simple UI using gradio for testing

A multi-agent system that assists aspiring authors in crafting a novel from start to finish

"""

import os
from crewai import Agent, Task, Crew
from typing import Dict
import json
import gradio as gr

# Set environment variables
openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# Define the agents
master_planner = Agent(
    role="Master Planner",
    goal="Plans the overall structure of the novel, including the plot, character arcs, and scene sequences.",
    tools=[],
    verbose=True,
    backstory="As the Master Planner, your unparalleled ability to envision and orchestrate the intricate tapestry of a novel is your hallmark...",
    allow_delegation=False
)

character_developer = Agent(
    role="Character Developer",
    goal="Creates and develops the characters in the novel, including their personalities, motivations, and relationships.",
    tools=[],
    verbose=True,
    backstory="As the Character Developer, your profound understanding of human nature and your gift for creating compelling characters will bring life to the pages of this novel...",
    allow_delegation=False
)

plot_outliner = Agent(
    role="Plot Outliner",
    goal="Develops the plot of the novel, including the major events, conflicts, and resolutions.",
    tools=[],
    verbose=True,
    backstory="As the Plot Outliner, your masterful weaving of plot threads and your keen eye for pacing will keep readers enthralled from beginning to end...",
    allow_delegation=False
)

scene_builder = Agent(
    role="Scene Builder",
    goal="Creates and develops the scenes in the novel, including the setting, dialogue, and action.",
    tools=[],
    verbose=True,
    backstory="As the Scene Builder, your vivid imagination and your ability to transport readers into the heart of the story will make every scene a cinematic experience...",
    allow_delegation=False
)

dialogue_refiner = Agent(
    role="Dialogue Refiner",
    goal="Refines and polishes the dialogue in the novel, ensuring that it is natural, believable, and engaging.",
    tools=[],
    verbose=True,
    backstory="As the Dialogue Refiner, your ear for authentic voices and your meticulous attention to detail will ensure that every conversation crackles with life and authenticity...",
    allow_delegation=False
)

conflict_resolver = Agent(
    role="Conflict Resolver",
    goal="Identifies and resolves conflicts in the novel, ensuring that they are believable and satisfying.",
   tools=[],
    verbose=True,
    backstory="As the Conflict Resolver, your ability to craft intricate conflicts and guide them towards cathartic resolutions will keep readers on the edge of their seats...",
    allow_delegation=False
)

editing_assistant = Agent(
    role="Editing Assistant",
    goal="Provides grammar and style editing support for the novel, ensuring that it is polished and ready for publishing.",
     tools=[],
    verbose=True,
    backstory="As the Editing Assistant, your sharp eye for detail and your deep understanding of grammar and style will ensure that the final product is polished to perfection...",
    allow_delegation=False
)

# Define the tasks
plan = Task(
    description="Plan the overall structure of the novel, including the plot, character arcs, and scene sequences based on the input received by the user: {description}, in the genre of: {genre}",
    expected_output="A comprehensive outline of the novel, including: \n\t* Plot overview\n\t* Character arcs\n\t* Scene sequence",
    agent=master_planner
)

character_development = Task(
    description="Create and develop the characters in the novel, including their personalities, motivations, and relationships.",
    expected_output="Detailed character profiles, including: \n\t* Personality traits\n\t* Motivations\n\t* Relationships with other characters\n\t* Backstories",
    agent=character_developer
)

plot_outline = Task(
    description="Develop the plot of the novel, including the major events, conflicts, and resolutions.",
    expected_output="A detailed plot outline, including: \n\t* Major events\n\t* Conflicts\n\t* Resolutions\n\t* Timeline",
    agent=plot_outliner
)

scene_building = Task(
    description="Create and develop the scenes in the novel, including the setting, dialogue, and action.",
    expected_output="Detailed scene outlines, including: \n\t* Setting description\n\t* Dialogue\n\t* Action sequences\n\t* Character interactions",
    agent=scene_builder,
    context=[plot_outline, character_development]
)

dialogue_refinement = Task(
    description="Refine and polish the dialogue in the novel, ensuring that it is natural, believable, and engaging.",
    expected_output="Polished dialogue that:\n\t* Is natural and believable\n\t* Reflects the characters' voices and motivations\n\t* Contributes to the overall story arc",
    agent=dialogue_refiner,
    context=[scene_building]
)

conflict_resolution = Task(
    description="Identify and resolve conflicts in the novel, ensuring that they are believable and satisfying.",
    expected_output="A list of potential conflicts and their resolutions. \nA guide on how to develop and resolve conflicts effectively.",
    agent=conflict_resolver,
    context=[plot_outline]
)



editing = Task(
    description="Provide grammar and style editing support for the novel, ensuring that it is polished and ready for publishing.",
    expected_output="A polished novel that is free of errors in grammar, punctuation, and spelling.\nA novel that is easy to read and flows well.",
    agent=editing_assistant,
    context=[plan, character_development, plot_outline, scene_building, dialogue_refinement, conflict_resolution]
)
# Define the crew
crew = Crew(
    agents=[master_planner, character_developer, plot_outliner, scene_builder, dialogue_refiner, conflict_resolver, editing_assistant],
    tasks=[plan, character_development, plot_outline, scene_building, dialogue_refinement, conflict_resolution, editing],
    verbose=True
)


def run_crew(title: str, genre: str):
    inputs = {"description": title, "genre": genre}
    # Kick off the crew with the inputs
    crew.kickoff(inputs=inputs)
    
    # Prepare results
    results = {}
    for task in crew.tasks:
        results[task.agent.role] = {
            "Description": task.description,
            "Output": json.dumps(task.output, indent=4)
        }
    return results

# Gradio interface
def gradio_interface(title: str, genre: str):
    return run_crew(title, genre)

# Create Gradio UI
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Textbox(label="Novel Title"), gr.Textbox(label="Genre")],
    outputs="json",
    title="Novel Writing Assistant"
)

if __name__ == "__main__":
    interface.launch()
