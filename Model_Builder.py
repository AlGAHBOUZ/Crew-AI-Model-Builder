import warnings
from crewai import Agent, Task, Crew
import gradio as gr
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment variables
openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# Define agents

# Agent 0: Agent Planner
agent_planner = Agent(
    role="Agent Planner",
    goal="Determine the number of agents needed to fulfill the user's requirements and define the role of each agent.",
    tools=[],
    verbose=True,
    backstory=(
        "You are the master planner with the ability to analyze the user's requirements and decide on the optimal number of agents needed to accomplish the tasks. "
        "Your deep understanding of multi-agent systems allows you to define clear roles for each agent, ensuring that the framework operates smoothly and efficiently."
    ),
    allow_delegation=False
)

# Agent 1: Goal and Backstory Writer
goal_writer = Agent(
    role="Agent Goal and Backstory Writer",
    goal="Writing the goal and backstory for the agents provided by the Agent Planner",
    tools=[],
    verbose=True,
    backstory=(
        "You are an expert in creating clear and comprehensive goals and backstories for the AI agents the Agent Planner provides"
        "Your unmatched ability to univel the goals and backstories of the agents provided by the agent planner allows you to craft goals that precisely define each agent's role in the multi-agent framework. "
        "Your work sets the foundation for the entire framework, ensuring that each agent has a well-defined purpose and backstory."
    ),
    allow_delegation=False,
    async_execution = True
)

# Agent 2: Task Detailer
task_detailer = Agent(
    role="Task Detailer",
    goal="Writing detailed tasks for each agent, including expected outputs follwing the provided examples from the original code.",
    tools=[],
    verbose=True,
    backstory=(
        "As a Task Detailer, you are responsible for breaking down complex tasks into detailed, actionable steps. "
        "Your skills in outlining the tasks, expected outputs, and providing relevant examples ensure that each agent knows exactly what is required. "
        "Your work ensures that the agents perform their roles efficiently, with clear guidance and practical examples."
    ),
    allow_delegation=False,
    async_execution = True
)

# Agent 3: Tool Assigner
tool_assigner = Agent(
    role="Tool Assigner",
    goal="Assigning the necessary tools to each agent, including pre-defined tools or writing code for custom tools as needed.",
    tools=[],
    verbose=True,
    backstory=(
        "As the Tool Assigner, you are an expert in identifying the right tools for each agent to perform their tasks. "
        "Your knowledge of available tools and ability to write custom code ensures that each agent is equipped with what they need to meet the user requirements. "
        "You play a crucial role in enabling the agents to execute their tasks with the right resources."
    ),
    allow_delegation=False,
    async_execution = True
)

# Agent 4: Framework Combiner
framework_combiner = Agent(
    role="Framework Combiner",
    goal="Combining the work of the Goal Writer, Task Detailer, and Tool Assigner into a final Python code that meets the user's requirements.",
    tools=[],
    verbose=True,
    backstory=(
        "As the Framework Combiner, you are responsible for bringing together the work of the previous agents into a cohesive Python script. "
        "Your attention to detail and understanding of the overall framework ensures that the final product is aligned with the user's goals and ready for deployment."
    ),
    allow_delegation=False
)


# Define Tasks

# Task for Agent 0: Planning Agents
plan_agents = Task(
    description=(
        "1. Analyze the user's requirements which are : {requirements} to determine the number of agents needed.\n"
        "2. Define the role of each agent based on the tasks that need to be accomplished.\n"
        "3. Try to minmize the number of agents, only include what are absolutely nesceaary to acomplish a given task, which is usally from 3 to 5, very rarely passing 6\n"
        "4. Example format:\n"
        "\t Agent 1: Role - Agent Goal and Backstory Writer\n"
        "\t Agent 2: Role - Task Detailer\n"
    ),
    expected_output="A list of agents with defined roles that will be used to fulfill the user's requirements.",
    agent=agent_planner
)

# Task for Agent 1: Writing Goals and Backstories
write_goals_backstories = Task(
    description=(
        "1. Write the goal for each agent as follwing the input given by the Agent Planner.\n"
        "2. Craft a backstory that provides context and depth to each agent's role.\n"
        "3. Use the format and writing style of simlar to the sample below:\n"
        "\t goal='Search the web looking for novel and impactful ideas in the area of one of the topics provided by the Scraper then provide a project proposal',\n"
        "\t backstory='As an AI Topic Researcher, your prowess in navigating and extracting critical information from the internet is unmatched...'\n"
    ),
    expected_output="Goals and backstories for all agents in the required format.",
    agent=goal_writer,
    context = [plan_agents]
)

# Task for Agent 2: Detailing Tasks
detail_tasks = Task(
    description=(
        "1. Based on the backstroy, write the tasks required for each agent in detail, ensuring clarity and completeness, follwing the input given by the Agent Planner.\n"
        "2. Highlight the expected outputs for each AGENT's task.\n"
        "3. Use the format and writing style of simlar to the sample below:\n"
        """
        description=(
        "1. Write the code to solve the problem using the suggested algorithm(s).\n"
        "2. If the solution is incorrect, revise the code and rewrite it to ensure correctness.\n"
        "3. Make sure you obey all constrains, and beware of the time limit of the problem"
        ),
        expected_output="A code solution that solves the problem.",
        """
    ),
    expected_output="Detailed tasks for each agent, with their expected outputs, combined into one Single Task named after their respective agents",
    agent=task_detailer,
    context = [plan_agents]
)

# Task for Agent 3: Assigning Tools
assign_tools = Task(
    description=(
        "1. Identify and assign only the necessary tools to each agent, following the input given by the Agent Planner.\n"
        "2. Here is a list of all pre-defined tools in the crewai_tools library:\n"
        "   from crewai_tools import (\n"
        "       ScrapeWebsiteTool,\n"
        "       SerperDevTool\n"
        "   )\n\n"
        "   Available tools include:\n"
        "   [\"BaseTool\", \"BrowserbaseLoadTool\", \"CSVSearchTool\", \"CodeDocsSearchTool\", \"CodeInterpreterTool\", \"ComposioTool\", \"DOCXSearchTool\", \"DallETool\", \"DirectoryReadTool\", \"DirectorySearchTool\", \"EXASearchTool\", \"FileReadTool\", \"FileWriterTool\", \"FirecrawlCrawlWebsiteTool\", \"FirecrawlScrapeWebsiteTool\", \"FirecrawlSearchTool\", \"GithubSearchTool\", \"JSONSearchTool\", \"LlamaIndexTool\", \"MDXSearchTool\", \"MultiOnTool\", \"MySQLSearchTool\", \"NL2SQLTool\", \"PDFSearchTool\", \"PGSearchTool\", \"RagTool\", \"ScrapeElementFromWebsiteTool\", \"ScrapeWebsiteTool\", \"ScrapflyScrapeWebsiteTool\", \"SeleniumScrapingTool\", \"SerperDevTool\", \"SerplyJobSearchTool\", \"SerplyNewsSearchTool\", \"SerplyScholarSearchTool\", \"SerplyWebSearchTool\", \"SerplyWebpageToMarkdownTool\", \"TXTSearchTool\", \"Tool\", \"VisionTool\", \"WebsiteSearchTool\", \"XMLSearchTool\", \"YoutubeChannelSearchTool\", \"YoutubeVideoSearchTool\"]\n\n"
        "3. Where pre-defined tools are insufficient, write code to create custom tools to meet the requirements.\n"
        "4. Ensure that all tools are correctly integrated into the agents' tasks, using a format similar to the example below:\n"
        "   tools=[scrape_tool, search_tool],\n"
        "5. Just to remind you, most of the times, agents don't need tools, so only when absolutely necessary, grant an agent a tool"
    ),
    expected_output="Tools assigned to each agent, with custom tools coded where necessary.",
    agent=tool_assigner,
    context=[plan_agents]
)

# Task for Agent 4: Combining the Framework
combine_framework = Task(
    description=(
        "1. Collect the goals, backstories, tasks, and tools created by the previous agents.\n"
        "2. Combine them into a final Python script that meets the user's requirements follwing Crew AI's syntax\n"
        "3. Ensure that the code doesn't have any errors before provding the final code"
        "4. Import all neseccary libraires, like: from crewai import Agent, Task, Crew and other's when needed, and set up the environment variables: "
        """
        openai_api_key = get_openai_api_key()
        os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

        """
        "5. If user input is required, follow this syntax:\n"
        """
        plan = Task(descritpion = "Write the outline of the project proposal based on the user's idea, which is: (idea) (the variable should be passed in curly brackets!!!!) that the writer will fill with the needed content",
        expected_output="A list of strings containing the names of the AI topics scraped from the website.",
        agent=planner
        ")
        idea = input("Eter your project idea so our agents can work on it: ")
        crew.kickoff(inputs = a dictinoary having the name of the variable inside the task and the name of the vairable you want to pass to the task recived from the user's input)
        """
        "6. Ensure that the combined input from the previous agents results in agents and their tasks being formatted similarly to the example below:\n"
        """
        # Agent 1: Web Scraper
        scraper = Agent(
            role="Valeo AI Web Scraper",
            goal="Scraping Valeo'es website"
            "and gathering the titles of the AI topics currently being researched",
            tools = [scrape_tool],
            verbose=True,
            backstory=(
                "You are a professional web scraper, with umatched skills "
                "in navigating and extracting critical "
                "information of the topics currently being researched at Valeo ('https://www.valeo.com/en/valeo-ai/')"
            ),
            allow_delegation=False
        
        )
        #Scrap
        scrap = Task(
            description=(
                "1. Scrape Valeo's website at 'https://www.valeo.com/en/valeo-ai/' to gather the titles having <h2> tags for the AI topics currently being researched.\n"
                "2. Compile the titles into a list of strings.\n"
            ),
            expected_output="A list of strings containing the names of the AI topics scraped from the website.",
            agent=scraper
        )
        # Final Crew:
        job_application_crew = Crew(
            agents=[scraper,
                researcher,
                    planner,
                    writer,
                    editor
                    ],

            tasks=[scrap,
                research,
                plan,
                write, edit],

            verbose=True
        )
        # Running the crew
        result = job_application_crew.kickoff()
        print(result)
        """
        "7.If the output of one or more tasks of an agent(s) is nessecary for the completion of another agent's task, pass the name of these tasks in the context paramter, like the example below:"
        """
        plan = Task(descritpion = "Write the outline of the project proposal that the writer will fill with the needed content"
            expected_output="A list of strings containing the names of the AI topics scraped from the website.",
            agent=planner
        ")

        write = Task(
        descritpion = "Write a detailed project proposal following the outline provided by the planner, where each section is at least 5 lines"
            expected_output="A detailed project proposal.",
            agent=writer
            context = [plan]

        )
        """
        "8. If a couple or more of agents don't depend on each other's work, you can make the async_execution = True when declaring the Agents"
    ),
    expected_output=("A complete Python script for the multi-agent framework, from start (importing libraires) to finish (running the crew), ready for review, where each agent is defined in to suit CrewAI's syntax.") ,
    agent=framework_combiner,
    context = [write_goals_backstories, detail_tasks,assign_tools ]
)

crew = Crew(
    agents=[agent_planner, goal_writer,
            task_detailer,
            tool_assigner,
            framework_combiner],
    tasks=[plan_agents, write_goals_backstories,
           detail_tasks,
           assign_tools,
           combine_framework],
    verbose=True
)




def generate_python_code(user_requirements):
    # Convert user_requirements to a dictionary
    inputs = {"requirements": user_requirements}
    
    result = crew.kickoff(inputs)
    
    final_python_code = crew.final_agent.output 
    
    # Save the final Python code to a file-like object
    code_file = BytesIO()
    code_file.write(final_python_code.encode('utf-8'))
    code_file.seek(0)
    
    return result, final_python_code, code_file

# Create the Gradio UI
def main():
    with gr.Blocks() as ui:
        gr.Markdown("# Multi-Agent Python Code Generator")
        
        with gr.Row():
            user_requirements = gr.Textbox(
                label="Enter Your Requirements",
                placeholder="Describe your project requirements here...",
                lines=5
            )
        
        generate_button = gr.Button("Generate Python Code")
        result_output = gr.Textbox(label="Result")
        output_code = gr.Code(label="Generated Python Code")
        download_button = gr.File(label="Download Python Code", visible=False)
        
        generate_button.click(
            fn=generate_python_code,
            inputs=user_requirements,
            outputs=[result_output, output_code, download_button]
        )
        
    ui.launch()

if __name__ == "__main__":
    main()


