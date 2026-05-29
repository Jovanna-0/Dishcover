# Dishcover
Dishcover: Ingredient-Based Recipe Recommendation Engine
Overview
Dishcover is an end-to-end classical machine learning pipeline which addresses household food waste by allowing users to input their currently available ingredients and receive highly relevant recipe recommendations.  

The core recommendation logic utilizes a content-based filtering approach, specifically deploying TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization and Cosine Similarity  to evaluate and rank recipes based on ingredient uniqueness and availability. The resulting model is deployed as a fully interactive web application via Streamlit.  

Key Features:
Ingredient-Matching Engine: Computes similarity scores between user inputs and a cleaned dataset of over 13,500 recipes.  

Interactive UI: A streamlined, dropdown-based web interface for error-free ingredient selection.  

Fast Retrieval: Precomputed TF-IDF vectors ensure rapid calculation and highly responsive user interactions.