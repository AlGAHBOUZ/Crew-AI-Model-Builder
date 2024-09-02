# Crew-AI-Model-Builder
## Project Description
This repository features an AI model built using Crew AI that is designed to automate the creation of other multi-agent models. By utilizing this framework, you can easily define, plan, and generate AI agents tailored to specific tasks, streamlining the process of multi-agent system development. The model efficiently handles the creation of agents with defined roles, goals, tasks, and necessary tools, ensuring that your multi-agent framework operates smoothly to achieve the user's requirements.

## How to Use?
1) Clone the Repository: Start by cloning the repository to your local machine.
2) Install Dependencies: Install all required dependencies listed in the requirements.txt file.
3) Set Environment Variables: Make sure to set the necessary environment variables, including your OpenAI API key and model name.
4) Run the Model: Execute the main script to start the AI model builder, inputting your project requirements when prompted.
5) Review the Output: The model will generate a multi-agent framework based on your input, which you can find in the specified output directory.

## Code Explanation
### Overview
The code is structured around defining and managing a set of AI agents using the Crew AI framework. Each agent has a specific role, and they collaborate to create a final multi-agent system based on the user's requirements. Here's a breakdown of the key components:

### Agents
#### Agent Planner: 
The mastermind behind determining the number of agents needed and defining their roles. It ensures that only the necessary agents are created, minimizing redundancy.

#### Goal and Backstory Writer: 
This agent writes the goals and backstories for the agents defined by the Agent Planner. It ensures that each agent’s purpose is well-articulated, providing a solid foundation for the tasks ahead.

#### Task Detailer: 
The Task Detailer breaks down the tasks for each agent into detailed steps. It outlines the expected outputs and provides examples to guide each agent in fulfilling its role.

#### Tool Assigner: 
This agent identifies and assigns the necessary tools to each agent, including both predefined tools and custom-coded ones where required. It ensures that agents are equipped with the resources they need without overloading them with unnecessary tools.

#### Framework Combiner: 
The final agent combines all the work done by the previous agents into a cohesive Python script, following Crew AI's syntax. It ensures that the final code meets the user’s requirements and is ready for deployment.

#### Results
The results generated while testing the model can be found in the Produced Models folder.

## Limitations
#### Redundant Tools: 
The model sometimes assigns unnecessary tools to agents, which can result in redundancy. In most cases, these tools can be removed by the user without affecting the model's functionality.

#### Complex Model Building: 
The model is not capable of building highly complex multi-agent systems. It is best suited for creating simpler models with a clearly defined description.

## Contributions
Contributions to the model are welcome. If you encounter any bugs, issues, or have ideas for enhancements, feel free to open an issue or submit a pull request. Collaboration is encouraged to help improve the model's capabilities and address its limitations.
