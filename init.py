import numpy as np
import pandas as pd

np.random.seed(42)  # For reproducibility

# Define distributions and lookup values for categories
education_levels = ["High School", "Associate's", "Bachelor's", "Master's", "PhD", "Professional (MD/JD)"]
edu_probs_women = [0.10, 0.10, 0.40, 0.25, 0.05, 0.10]
edu_probs_men =   [0.12, 0.15, 0.40, 0.20, 0.05, 0.08]

dating_intentions = ["Casual", "Long-term", "Marriage-Oriented", "Open to Either", "Figuring it Out"]
intent_probs_women = [0.15, 0.25, 0.15, 0.25, 0.20]
intent_probs_men   = [0.25, 0.20, 0.10, 0.30, 0.15]

drinking_habits = ["Never", "Socially", "Often"]
drink_probs_women = [0.20, 0.70, 0.10]
drink_probs_men   = [0.15, 0.65, 0.20]

personality_traits = [
    "Adventurous", "Ambitious", "Artistic", "Athletic", "Carefree", "Compassionate",
    "Creative", "Easygoing", "Funny", "Intellectual", "Outgoing", "Spontaneous",
    "Down-to-earth", "Energetic", "Thoughtful", "Charming"
]

attract_desc_women = [
    "Cute girl-next-door", "Charming with a warm smile", "Pretty and down-to-earth",
    "Cute and bubbly personality", "Charming and sweet", "Elegant and graceful",
    "Gorgeous and confident", "Stunning and charismatic"
]

attract_desc_men = [
    "Average-looking guy with a friendly vibe", "Down-to-earth looks, great personality",
    "Cute in a nerdy way", "Boy-next-door charm", "Tall, athletic build, nice smile",
    "Charming and well-groomed", "Rugged and handsome", "Model-like looks"
]

women_attract_scores = np.random.normal(loc=0.5, scale=0.15, size=100)
men_attract_scores = np.random.beta(a=1.0, b=4.0, size=100)

def get_attractiveness_description(score, descriptors_list):
    n = len(descriptors_list)
    rank = int(score * n)
    if rank >= n:
        rank = n - 1
    return descriptors_list[rank]

# Generate women's profiles
women_profiles = []
for i in range(100):
    age = int(np.round(np.random.normal(loc=27.0, scale=1.5)))
    age = np.clip(age, 25, 30)
    height = np.round(np.clip(np.random.normal(loc=65.0, scale=2.5), 60, 72), 1)
    education = np.random.choice(education_levels, p=edu_probs_women)
    intention = np.random.choice(dating_intentions, p=intent_probs_women)
    drinking = np.random.choice(drinking_habits, p=drink_probs_women)
    traits = np.random.choice(personality_traits, size=2, replace=False)
    personality = traits[0] + " and " + traits[1]
    attract_score = women_attract_scores[i]
    attract_desc = get_attractiveness_description(attract_score, attract_desc_women)
    
    women_profiles.append({
        "WomanID": f"W{i+1}",
        "Age": age,
        "Height(inches)": height,
        "Education": education,
        "Dating Intentions": intention,
        "Drinking Habits": drinking,
        "Personality Traits": personality,
        "Physical Attractiveness": attract_desc
    })

women_df = pd.DataFrame(women_profiles)
women_df.to_csv("synthetic_women_profiles.csv", index=False)

# Generate men's profiles
men_profiles = []
for j in range(100):
    age = int(np.round(np.random.normal(loc=27.0, scale=1.5)))
    age = np.clip(age, 25, 30)
    height = np.round(np.clip(np.random.normal(loc=70.0, scale=2.5), 64, 78), 1)
    education = np.random.choice(education_levels, p=edu_probs_men)
    intention = np.random.choice(dating_intentions, p=intent_probs_men)
    drinking = np.random.choice(drinking_habits, p=drink_probs_men)
    traits = np.random.choice(personality_traits, size=2, replace=False)
    personality = traits[0] + " and " + traits[1]
    attract_score = men_attract_scores[j]
    attract_desc = get_attractiveness_description(attract_score, attract_desc_men)

    men_profiles.append({
        "ManID": f"M{j+1}",
        "Age": age,
        "Height(inches)": height,
        "Education": education,
        "Dating Intentions": intention,
        "Drinking Habits": drinking,
        "Personality Traits": personality,
        "Physical Attractiveness": attract_desc
    })

men_df = pd.DataFrame(men_profiles)
men_df.to_csv("synthetic_men_profiles.csv", index=False)

# Initialize probability matrices
prob_women_likes_men = np.zeros((100, 100))
prob_men_likes_women = np.zeros((100, 100))

# Women -> Men probabilities
for i in range(100):
    woman_attr = women_attract_scores[i]
    for j in range(100):
        p = 0.6 * (0.5 / woman_attr) ** 1
        man_attr = men_attract_scores[j]
        p *= p * man_attr
        p_final = p / (p + 1)  # Logistic transformation
        prob_women_likes_men[i, j] = p_final

# Scale to target ~12% mean
avg_prob_women = prob_women_likes_men.mean()
scale_factor = 0.12 / avg_prob_women
prob_women_likes_men *= scale_factor
prob_women_likes_men = prob_women_likes_men / (prob_women_likes_men + 1)

# Men -> Women probabilities
for j in range(100):
    man_attr = men_attract_scores[j]
    for i in range(100):
        woman_attr = women_attract_scores[i]
        p = 0.15 + 0.5 * woman_attr * (1 / man_attr) ** 0.5
        p_final = p / (p + 1)  # Logistic transformation
        prob_men_likes_women[j, i] = p_final

# Scale to target ~43% mean
avg_prob_men = prob_men_likes_women.mean()
scale_factor = 0.43 / avg_prob_men
prob_men_likes_women *= scale_factor
prob_men_likes_women = prob_men_likes_women / (prob_men_likes_women + 1)

# Save probability matrices
women_ids = [f"W{k+1}" for k in range(100)]
men_ids   = [f"M{k+1}" for k in range(100)]

wm_df = pd.DataFrame(prob_women_likes_men, index=women_ids, columns=men_ids)
wm_df.to_csv("probability_matrix_women_likes_men.csv", index_label="Woman")

mw_df = pd.DataFrame(prob_men_likes_women, index=men_ids, columns=women_ids)
mw_df.to_csv("probability_matrix_men_likes_women.csv", index_label="Man")

print("Synthetic profiles and probability matrices saved.")
