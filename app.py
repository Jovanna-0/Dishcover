import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter


# --- Page Configuration ---
st.set_page_config(page_title="Dishcover", page_icon="🍳", layout="centered")

# --- Load Models & Data ---
# --- Load Models & Data ---
@st.cache_resource
def load_models():
    vectorizer = joblib.load('data/tfidf_vectorizer.pkl')
    matrix = joblib.load('data/tfidf_matrix.pkl')
    df = pd.read_csv('data/cleaned_recipes.csv') 
    return vectorizer, matrix, df

vectorizer, tfidf_matrix, df = load_models()

# --- THE FIX: Create unique ingredients but BAN single letters (like "a") ---
all_ingredients_text = " ".join(df['Cleaned_Ingredients'].dropna().astype(str).tolist())
# This line grabs the unique words, but ONLY if they are longer than 1 letter!
unique_ingredients = sorted([word for word in list(set(all_ingredients_text.split())) if len(word) > 1])

# --- User Input Cleaning Function ---
measurements = [
    # Articles, Prepositions & Conjunctions (This kills the "a" and "the"!)
    'a', 'an', 'the', 'and', 'or', 'with', 'of', 'in', 'into', 'for', 'to', 'some', 'as', 'at', 'about',
    'of', 'about', 'accompaniment',

    # Volume, Weight & Containers
    'cup', 'cups', 'oz', 'ounce', 'ounces', 'tsp', 'teaspoon', 'teaspoons',
    'tbsp', 'tablespoon', 'tablespoons', 'pound', 'pounds', 'lb', 'lbs',
    'gram', 'grams', 'g', 'kg', 'ml', 'liter', 'liters', 'pint', 'pints',
    'quart', 'quarts', 'gallon', 'gallons', 'fluid', 'fl', 'can', 'cans', 
    'package', 'packages', 'jar', 'jars', 'bottle', 'bottles', 'packet', 
    'packets', 'bag', 'bags', 'box', 'boxes', 'envelope',

    # Amounts & Shapes
    'pinch', 'dash', 'piece', 'pieces', 'clove', 'cloves', 'stick', 'sticks', 
    'bunch', 'bunches', 'sprig', 'sprigs', 'head', 'heads', 'handful', 
    'drop', 'drops', 'whole', 'half', 'quarter',

    # Prep States & Adjectives
    'chopped', 'diced', 'minced', 'sliced', 'peeled', 'fresh', 'large', 
    'small', 'medium', 'big', 'thin', 'thick', 'crushed', 'melted', 
    'beaten', 'cooked', 'raw', 'warm', 'cold', 'hot', 'softened', 
    'grated', 'shredded', 'sifted', 'divided', 'taste', 'dry', 
    'ground', 'roasted', 'fried', 'boiled', 'baked', 'optional', 'to'
]

def clean_input(user_string):
    cleaned_list = []
    for item in user_string.split(','):
        item = item.lower()
        item = re.sub(r'[^a-z\s]', '', item)
        words = item.split()
        filtered_words = [word for word in words if word not in measurements]
        cleaned_item = "_".join(filtered_words).strip('_')
        if cleaned_item:
            cleaned_list.append(cleaned_item)
    return cleaned_list

# --- The User Interface ---
st.title("🍳 Dishcover")
st.markdown("**Your Ingredient-Based Recipe Matcher**")
st.write("Select the ingredients you currently have, and we will find the perfect recipe for you!")

# Join all words together
all_ingredients_text = " ".join(df['Cleaned_Ingredients'].dropna().astype(str).tolist())
all_words = all_ingredients_text.split()

# Count how many times every single word appears in the dataset
word_counts = Counter(all_words)

# Create the dropdown list, but ONLY allow words that appear at least 5 times!
unique_ingredients = sorted([word for word, count in word_counts.items() if count >= 5 and len(word) > 1])


# --- The User Interface (The Dropdown Menu) ---
# Dropdown Pilihan Bahan
user_selection = st.multiselect(
    "What's in your kitchen?", 
    options=unique_ingredients,       
    placeholder="Choose your ingredients...",
    format_func=lambda x: x.replace('_', ' ').title() 
)

# Search Button & ML Logic
if st.button("Find Recipes", type="primary"):
    if not user_selection:
        st.warning("Please select at least one ingredient!")
    else:
        with st.spinner('Searching the pantry...'):
            # 1. Clean user input
            user_string = ", ".join(user_selection)
            cleaned_user_items = clean_input(user_string)
            user_search_string = " ".join(cleaned_user_items)
            
            # 2. Calculate TF-IDF scores (Cosine Similarity)
            user_vector = vectorizer.transform([user_search_string])
            similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
            
            # Grab the top 50 recipes to resort them
            top_50_indices = similarity_scores.argsort()[-50:][::-1]
            top_results = df.iloc[top_50_indices].copy()
            top_results['Cosine_Score'] = similarity_scores[top_50_indices]
            
            # 3. Count how many ingredients actually match (Match Counter)
            def count_matches(recipe_ingredients_string):
                count = 0
                for item in cleaned_user_items:
                    if re.search(rf'\b{item}\b', str(recipe_ingredients_string)):
                        count += 1
                return count
                
            top_results['Match_Count'] = top_results['Cleaned_Ingredients'].apply(count_matches)
            
            # 4. Sort based on the highest number of ingredient matches
            final_results = top_results.sort_values(
                by=['Match_Count', 'Cosine_Score'], 
                ascending=[False, False]
            ).head(5)
            
            st.success("Here are your top matches!")
            
            # 5. Display the results on the screen
            for index, row in final_results.iterrows():
                # Note: Remove `(Matches {row['Match_Count']} ingredients)` if you want to hide it from the user.
                with st.expander(f"🍽️ {row['Title'].title()} (Matches {row['Match_Count']} ingredients)"):
                    st.write("**Ingredients Required:**")
                    # Replaces the underscores with spaces and capitalizes the words nicely
                    clean_display = str(row['Cleaned_Ingredients']).replace('_', ' ').title()
                    st.write(clean_display)
                    
                    # Display instructions if they exist in the dataset
                    if pd.notna(row.get('Instructions')) and 'Instructions' in df.columns:
                        st.write("**Instructions:**")
                        st.write(row['Instructions'])