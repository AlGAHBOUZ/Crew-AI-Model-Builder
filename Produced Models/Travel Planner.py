"""
This Model was build entirely using the model builder, the only edits made were removing any redundant tools

A multi-agent system that streamlines the process of planning a comprehensive and personalized travel plan.

"""

import os
from crewai import Agent, Task, Crew
import pandas as pd

# Agent 1: Country Data Manager
country_data_manager = Agent(
    role="Country Data Manager",
    goal="Gather and organize comprehensive data on various countries, covering tourist attractions, cultural events, travel regulations, and other pertinent information relevant for trip planning",   
    backstory="As a seasoned Country Data Manager, your meticulous nature and unrivaled expertise in data collection empower you to curate an extensive database on countries worldwide...",
    verbose=True,
    allow_delegation=False
)

# Agent 2: Itinerary Planner
itinerary_planner = Agent(
    role="Itinerary Planner",
    goal="Personalize trip itineraries based on user specifications, considering duration, interests, and budget constraints, and generate a day-to-day plan for the journey",
    backstory="Your exceptional organizational abilities and deep understanding of user preferences make you an indispensable Itinerary Planner...",
    verbose=True,
    allow_delegation=False
)

# Agent 3: Travel Advisor
travel_advisor = Agent(
    role="Travel Advisor",
    goal="Provide expert guidance on transportation options within countries and between cities, including trains, buses, flights, and local modes of travel",
    backstory="As a seasoned Travel Advisor, your mastery of transportation systems allows you to navigate the complexities of travel...",
    verbose=True,
    allow_delegation=False
)

# Agent 4: Accommodation Specialist
accommodation_specialist = Agent(
    role="Accommodation Specialist",
    goal="Secure hotel and accommodation reservations throughout the trip, ensuring availability and alignment with user preferences",
    backstory="With your impeccable attention to detail and extensive network in the hospitality industry, you excel as an Accommodation Specialist...",
    verbose=True,
    allow_delegation=False
)

# Agent 5: Experience Curator
experience_curator = Agent(
    role="Experience Curator",
    goal="Recommend and arrange unique cultural experiences, tours, activities, and dining options to enrich the user's travel experience",
    backstory="As an Experience Curator, your passion for exploration and deep knowledge of local cultures enable you to craft unforgettable experiences...",
    verbose=True,
    allow_delegation=False
)

# Task 1: Country Data Collection
gather_country_data = Task(
    description="""
        1. Research and gather comprehensive data on different countries, covering tourist attractions, cultural events, travel regulations, and relevant information for trip planning.
        2. Organize and categorize the collected data into an easily accessible and searchable database.
        3. Stay updated on changes in travel regulations, visa requirements, and other important information.
    """,
    expected_output="A comprehensive database of country-specific travel information.",
    agent=country_data_manager
)

# Task 2: Itinerary Planning
create_itinerary = Task(
    description="""
        1. Analyze user preferences, including travel duration, interests, and budget constraints.
        2. Create customized daily plans that optimize sightseeing, cultural immersion, and relaxation time.
        3. Ensure that the itinerary aligns with the user's interests and preferences.
        4. Provide alternative options and suggestions to enhance the user's travel experience.
    """,
    expected_output="A detailed and personalized itinerary that meets the user's requirements.",
    agent=itinerary_planner,
    context=[gather_country_data]
)

# Task 3: Transportation Arrangement
book_transportation = Task(
    description="""
        1. Provide expert guidance on transportation options within the country and between cities.
        2. Research and recommend the most suitable and cost-effective modes of travel.
        3. Book transportation arrangements, including trains, buses, flights, and local modes of travel.
        4. Monitor travel schedules and provide real-time updates in case of delays or changes.
    """,
    expected_output="Well-planned and efficient transportation arrangements.",
    agent=travel_advisor,
    context=[create_itinerary]
)

# Task 4: Accommodation Booking
arrange_accommodation = Task(
    description="""
        1. Manage hotel and accommodation reservations throughout the trip.
        2. Research and select hotels that meet the user's preferences and budget constraints.
        3. Negotiate rates and secure reservations to ensure availability and suitability.
        4. Provide detailed information about hotel amenities, location, and facilities.
    """,
    expected_output="Secured and confirmed hotel reservations that meet the user's requirements.",
    agent=accommodation_specialist,
    context=[create_itinerary]
)

# Task 5: Experience Curation
plan_experiences = Task(
    description="""
        1. Recommend and arrange unique cultural experiences, tours, activities, and dining options.
        2. Research and identify hidden gems and off-the-beaten-path attractions.
        3. Make reservations and provide detailed information about tour schedules, activity availability, and dining options.
        4. Collaborate with local vendors and service providers to ensure the quality and authenticity of experiences.
        5. Provide personalized recommendations that align with the user's interests and preferences.
    """,
    expected_output="A curated list of unique and memorable experiences that enhance the user's travel experience.",
    agent=experience_curator,
    context=[create_itinerary]
)

# Crew: Trip Planning Crew
trip_planning_crew = Crew(
    agents=[country_data_manager, itinerary_planner, travel_advisor, accommodation_specialist, experience_curator],
    tasks=[gather_country_data, create_itinerary, book_transportation, arrange_accommodation, plan_experiences],
    verbose=True
)

# Kick off the Crew and run the tasks
result = trip_planning_crew.kickoff()
print(result)
