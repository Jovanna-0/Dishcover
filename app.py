import streamlit as st
import pandas as pd
import joblib
import re
from collections import Counter

# --- Page Configuration ---
st.set_page_config(page_title="Dishcover", page_icon="🍳", layout="centered")

@st.cache_resource
def load_models():
    # We removed the '../' because app.py is already in the main folder!
    vectorizer = joblib.load('data/tfidf_vectorizer.pkl') 
    knn_model = joblib.load('data/knn_model.pkl') 
    df = pd.read_csv('data/cleaned_recipes.csv') 
    return vectorizer, knn_model, df

try:
    vectorizer, knn_model, df = load_models()
except Exception as e:
    st.error(f"Error loading files. Check your folder paths! Details: {e}")
    st.stop()

# --- User Input Cleaning Function ---
measurements = [
    'a', 'an', 'the', 'and', 'or', 'with', 'of', 'in', 'into', 'for', 'to', 'some', 'as', 'at', 'about',
    'accompaniment', 'cup', 'cups', 'oz', 'ounce', 'ounces', 'tsp', 'teaspoon', 'teaspoons',
    'tbsp', 'tablespoon', 'tablespoons', 'pound', 'pounds', 'lb', 'lbs',
    'gram', 'grams', 'g', 'kg', 'ml', 'liter', 'liters', 'pint', 'pints',
    'quart', 'quarts', 'gallon', 'gallons', 'fluid', 'fl', 'can', 'cans', 
    'package', 'packages', 'jar', 'jars', 'bottle', 'bottles', 'packet', 
    'packets', 'bag', 'bags', 'box', 'boxes', 'envelope', 'pinch', 'dash', 
    'piece', 'pieces', 'clove', 'cloves', 'stick', 'sticks', 'bunch', 'bunches', 
    'sprig', 'sprigs', 'head', 'heads', 'handful', 'drop', 'drops', 'whole', 
    'half', 'quarter', 'chopped', 'diced', 'minced', 'sliced', 'peeled', 
    'fresh', 'large', 'small', 'medium', 'big', 'thin', 'thick', 'crushed', 
    'melted', 'beaten', 'cooked', 'raw', 'warm', 'cold', 'hot', 'softened', 
    'grated', 'shredded', 'sifted', 'divided', 'taste', 'dry', 'ground', 
    'roasted', 'fried', 'boiled', 'baked', 'optional'
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

# --- Build the Dropdown Options ---
all_ingredients_text = " ".join(df['Cleaned_Ingredients'].dropna().astype(str).tolist())
all_words = all_ingredients_text.split()
word_counts = Counter(all_words)

# ONLY allow words that appear at least 5 times and are more than 1 letter
unique_ingredients = sorted([word for word, count in word_counts.items() if count >= 5 and len(word) > 1])

# --- The User Interface ---
st.title("🍳 Dishcover")
st.markdown("**Your Ingredient-Based Recipe Matcher**")
st.write("Select the ingredients you currently have, and we will find the perfect recipe for you!")

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
        with st.spinner('Powering up the Machine Learning Radar...'):
            
            # 1. Clean user input
            user_string = ", ".join(user_selection)
            cleaned_user_items = clean_input(user_string)
            user_search_string = " ".join(cleaned_user_items)
            
            # 2. Vectorize the input
            user_vector = vectorizer.transform([user_search_string])
            
            # 3. ASK THE ML MODEL TO FIND THE NEIGHBORS (The Big Upgrade!)
            distances, indices = knn_model.kneighbors(user_vector, n_neighbors=50)
            similarity_scores = 1 - distances.flatten() # Convert distance back to Cosine Score
            top_50_indices = indices.flatten()
            
            top_results = df.iloc[top_50_indices].copy()
            top_results['Cosine_Score'] = similarity_scores
            
            # 4. Count how many ingredients actually match (Match Counter)
            def count_matches(recipe_ingredients_string):
                count = 0
                for item in cleaned_user_items:
                    if re.search(rf'\b{item}\b', str(recipe_ingredients_string)):
                        count += 1
                return count
                
            top_results['Match_Count'] = top_results['Cleaned_Ingredients'].apply(count_matches)
            
            # 5. Sort based on the highest number of ingredient matches
            final_results = top_results.sort_values(
                by=['Match_Count', 'Cosine_Score'], 
                ascending=[False, False]
            ).head(5)
            
            st.success("Here are your top matches!")
            
            # 6. Display the results beautifully
            for index, row in final_results.iterrows():
                with st.expander(f"🍽️ {row['Title'].title()} (Matches {row['Match_Count']} ingredients)"):
                    st.write("**Ingredients Required:**")
                    
                    # Try to parse the ingredients list nicely, otherwise fall back to string
                    try:
                        import ast
                        ing_list = ast.literal_eval(row['Ingredients'])
                        for ing in ing_list:
                            st.markdown(f"- {ing}")
                    except:
                        clean_display = str(row['Cleaned_Ingredients']).replace('_', ' ').title()
                        st.write(clean_display)
                    
                    # Display instructions if they exist
                    if pd.notna(row.get('Instructions')) and 'Instructions' in df.columns:
                        st.write("**Instructions:**")
                        st.write(row['Instructions'])