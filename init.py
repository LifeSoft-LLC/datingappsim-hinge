import numpy as np
import pandas as pd

np.random.seed(42)  # For reproducibility

# Define distributions and lookup values for categories
education_levels = ["High School", "Associate's", "Bachelor's", "Master's", "PhD", "Professional (MD/JD)"]
# Probabilities for women's education (summing to 1)
edu_probs_women = [0.10, 0.10, 0.40, 0.25, 0.05, 0.10]
# Probabilities for men's education (slightly different)
edu_probs_men =   [0.12, 0.15, 0.40, 0.20, 0.05, 0.08]

dating_intentions = ["Casual", "Long-term", "Marriage-Oriented", "Open to Either", "Figuring it Out"]
# Probabilities for women intentions
intent_probs_women = [0.15, 0.25, 0.15, 0.25, 0.20]
# Probabilities for men intentions (more casual)
intent_probs_men   = [0.25, 0.20, 0.10, 0.30, 0.15]

drinking_habits = ["Never", "Socially", "Often"]
# Probabilities for drinking habits
drink_probs_women = [0.20, 0.70, 0.10]
drink_probs_men   = [0.15, 0.65, 0.20]

# Personality traits list
personality_traits = [
    "Adventurous", "Ambitious", "Artistic", "Athletic", "Carefree", "Compassionate",
    "Creative", "Easygoing", "Funny", "Intellectual", "Outgoing", "Spontaneous",
    "Down-to-earth", "Energetic", "Thoughtful", "Charming"
]

# Attractiveness descriptors for women (ordered roughly from moderate to high attractiveness)
attract_desc_women = [
    "Cute girl-next-door", "Charming with a warm smile", "Pretty and down-to-earth",
    "Cute and bubbly personality", "Charming and sweet", "Elegant and graceful",
    "Gorgeous and confident", "Stunning and charismatic"
]
# Attractiveness descriptors for men (from lower to higher)
attract_desc_men = [
    "Average-looking guy with a friendly vibe", "Down-to-earth looks, great personality",
    "Cute in a nerdy way", "Boy-next-door charm", "Tall, athletic build, nice smile",
    "Charming and well-groomed", "Rugged and handsome", "Model-like looks"
]

# Assign numeric attractiveness scores (0 to 1) for each person
women_attract_scores = np.clip(np.random.normal(loc=0.5, scale=0.15, size=100), 0, 1)
men_attract_scores = np.random.beta(a=1.0, b=4.0, size=100)
men_attract_scores = np.clip(men_attract_scores, 0, 1)

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
    height = np.random.normal(loc=65.0, scale=2.5)
    height = np.clip(height, 60, 72)
    height = round(height, 1)
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
    height = np.random.normal(loc=70.0, scale=2.5)
    height = np.clip(height, 64, 78)
    height = round(height, 1)
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

# Mapping for education levels (for compatibility checks)
edu_level_index = {level: idx for idx, level in enumerate(education_levels)}

# Utility: seriousness level for intentions
intent_rank = {
    "Casual": 1,
    "Figuring it Out": 2,
    "Open to Either": 2,
    "Long-term": 3,
    "Marriage-Oriented": 4
}

# Initialize probability matrices
prob_women_likes_men = np.zeros((100, 100))
prob_men_likes_women = np.zeros((100, 100))

# Calculate Women -> Men probabilities with stronger incompatibility responses
for i, woman in enumerate(women_profiles):
    woman_attr = women_attract_scores[i]
    for j, man in enumerate(men_profiles):

        # baseline: more attractive women are more choosy
        p = 0.6*(0.5/woman_attr)**1
    
        # baseline further influenced by man's attractiveness
        man_attr = men_attract_scores[j]
        p *= p * man_attr

        # Height compatibility: if man is shorter than woman, apply a stronger penalty.
        if man["Height(inches)"] < woman["Height(inches)"]:
            height_diff = woman["Height(inches)"] - man["Height(inches)"]
            # Increase penalty: 10% per inch difference, up to 40%
            penalty = 1 - min(0.10 * height_diff, 0.40)
            p *= penalty

        # Education compatibility: stronger penalties for education mismatches.
        woman_edu_level = edu_level_index[woman["Education"]]
        man_edu_level = edu_level_index[man["Education"]]
        edu_diff = woman_edu_level - man_edu_level
        if edu_diff >= 2:
            p *= 0.3  # larger penalty for a two-level gap
        elif edu_diff == 1:
            p *= 0.6  # penalty for a one-level gap

        # Dating intentions: stronger penalty for mismatches.
        w_intent = woman["Dating Intentions"]
        m_intent = man["Dating Intentions"]
        if w_intent == m_intent:
            p *= 1.1  # slight boost for matching
        else:
            # If woman is serious and man is casual:
            if (w_intent in ["Long-term", "Marriage-Oriented"]) and m_intent == "Casual":
                p *= 0.3
            # If man is serious and woman is casual:
            elif (m_intent in ["Long-term", "Marriage-Oriented"]) and w_intent == "Casual":
                p *= 0.5
            # If one is serious and the other is "Figuring it Out":
            if (w_intent in ["Long-term", "Marriage-Oriented"]) and m_intent == "Figuring it Out":
                p *= 0.7
            elif (m_intent in ["Long-term", "Marriage-Oriented"]) and w_intent == "Figuring it Out":
                p *= 0.8

        # Drinking compatibility: stronger penalty for a major mismatch.
        w_drink = woman["Drinking Habits"]
        m_drink = man["Drinking Habits"]
        drink_scale = {"Never": 0, "Socially": 1, "Often": 2}
        drink_diff = abs(drink_scale[w_drink] - drink_scale[m_drink])
        if drink_diff == 2:
            p *= 0.8  # stronger penalty

        # Removed clamp; assign raw probability directly
        prob_women_likes_men[i, j] = p

# Scale the Women->Men matrix to target an average of ~12%
avg_prob_women = prob_women_likes_men.mean()
if avg_prob_women < 0.10 or avg_prob_women > 0.15:
    scale_factor = 0.12 / avg_prob_women
    prob_women_likes_men *= scale_factor

# Apply logistic transformation: p* = p/(p+1)
prob_women_likes_men = prob_women_likes_men / (prob_women_likes_men + 1)

# Calculate Men -> Women probabilities with stronger incompatibility responses
for j, man in enumerate(men_profiles):
    man_attr = men_attract_scores[j]
    for i, woman in enumerate(women_profiles):
        woman_attr = women_attract_scores[i]
        # Combined baseline based on woman's attractiveness + hotter men are more choosy
        p = 0.15 + 0.5 * woman_attr * (1/man_attr)**0.5  # baseline

        # Height: if woman is taller than man, apply a stronger penalty.
        if woman["Height(inches)"] > man["Height(inches)"]:
            p *= 0.90  # increased penalty

        # Dating intentions: stronger penalty for mismatches.
        w_intent = woman["Dating Intentions"]
        m_intent = man["Dating Intentions"]
        if w_intent == m_intent:
            p *= 1.05
        else:
            if (w_intent in ["Long-term", "Marriage-Oriented"]) and m_intent == "Casual":
                p *= 0.6  # stronger penalty than before
            elif (m_intent in ["Long-term", "Marriage-Oriented"]) and w_intent == "Casual":
                p *= 0.8  # stronger penalty than before

        # Drinking compatibility: stronger penalty for major mismatch.
        w_drink = woman["Drinking Habits"]
        m_drink = man["Drinking Habits"]
        drink_scale = {"Never": 0, "Socially": 1, "Often": 2}
        drink_diff = abs(drink_scale[w_drink] - drink_scale[m_drink])
        if drink_diff == 2:
            p *= 0.90  # stronger penalty

        # Removed clamp; assign raw probability directly
        prob_men_likes_women[j, i] = p

# Scale the Men->Women matrix to target an average of ~43%
avg_prob_men = prob_men_likes_women.mean()
if avg_prob_men < 0.40 or avg_prob_men > 0.45:
    scale_factor = 0.43 / avg_prob_men
    prob_men_likes_women *= scale_factor

# Apply logistic transformation: p* = p/(p+1)
prob_men_likes_women = prob_men_likes_women / (prob_men_likes_women + 1)

# Create DataFrames for the matrices and save to CSV files.
women_ids = [f"W{k+1}" for k in range(100)]
men_ids   = [f"M{k+1}" for k in range(100)]

wm_df = pd.DataFrame(prob_women_likes_men, index=women_ids, columns=men_ids)
wm_df.to_csv("probability_matrix_women_likes_men.csv", index_label="Woman")

mw_df = pd.DataFrame(prob_men_likes_women, index=men_ids, columns=women_ids)
mw_df.to_csv("probability_matrix_men_likes_women.csv", index_label="Man")
