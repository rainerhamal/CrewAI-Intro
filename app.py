import os

import streamlit as st
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain.agents import tools
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

search_tool = DuckDuckGoSearchRun(name="Search")

def stream_data(step_output):
    st.markdown("---")
    for step in step_output:
        if isinstance(step, tuple) and len(step) == 2:
            action, observation = step
            if isinstance(action, dict) and "tool" in action and "tool_input" in action and "log" in action:
                st.markdown(f"# Action")
                st.markdown(f"**Tool:** {action['tool']}")
                st.markdown(f"**Tool Input** {action['tool_input']}")
                st.markdown(f"**Log:** {action['log']}")
                st.markdown(f"**Action:** {action['Action']}")
                st.markdown(f"**Action Input:** ```json\n{action['tool_input']}\n```")
            elif isinstance(action, str):
                st.markdown(f"**Action:** {action}")
            else:
                st.markdown(f"**Action:** {str(action)}")

            st.markdown(f"**Observation**")
            if isinstance(observation, str):
                observation_lines = observation.split('\n')
                for line in observation_lines:
                    if line.startswith('Title: '):
                        st.markdown(f"**Title:** {line[7:]}")
                    elif line.startswith('Link: '):
                        st.markdown(f"**Link:** {line[6:]}")
                    elif line.startswith('Snippet: '):
                        st.markdown(f"**Snippet:** {line[9:]}")
                    elif line.startswith('-'):
                        st.markdown(line)
                    else:
                        st.markdown(line)
            else:
                st.markdown(str(observation))
        else:
            st.markdown(step)

# !Assembling Agents

# Creating a senior researcher agent with memory and verbose mode
researcher = Agent(
    role = 'Senior Researcher',
    goal = 'Uncover groundbreaking technologies in {topic}',
    verbose = True,
    memory = True,
    backstory = (
        "Driven by curiosity, you're at the forefront of innovation, eager to explore and share knowledge that could change the world."
    ),
    tools = [search_tool],
    allow_delegation = True,# type: ignore
    step_callback = stream_data
)

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft engaging narratives that captivate and educate, bringing new discoveries to light in an accessible manner."
  ),
  tools=[search_tool],
  allow_delegation=False, # type: ignore
  step_callback = stream_data,
)

# !Defining Tasks
# Research task
research_task = Task(
    description = (
        "Identify the next big trend in {topic}. Focus on identifying pros and cons and the overall narrative. Your final report should clearly articulate the key points, its market opportunities, and potential risks."
    ),
    expected_output = 'A comprehensive paragraphs long report on the latest AI trends.',
    tools = [search_tool],
    agent = researcher,
) 

# Writing task with language model configuration
write_task = Task(
  description=(
    "Compose an insightful article on {topic}. Focus on the latest trends and how it's impacting the industry. This article should be easy to understand, engaging, and positive."
  ),
  expected_output='A 1 paragraph article on {topic} advancements formatted as markdown.',
  tools=[search_tool],
  agent=writer,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
)

# !Form Crew

# Forming the tech-focused crew with some enhanced configurations
crew = Crew(
    agents = [researcher, writer],
    tasks = [research_task, write_task],
    process = Process.sequential, # Optional: Sequential task execution is default
    memory = True,
    cache = True,
    max_rpm = 5,
    share_crew = False #share information to crewAI team to train models
)

# !Kick it off

# Starting the task execution process with enhanced feedback

#  result = crew.kickoff(inputs={'topic': 'AI in healthcare'})
# print(result)

# Main Streamlit page

if st.button("Begin"):

  result = crew.kickoff(inputs={'topic': 'AI in healthcare'})
  
  st.markdown(result)