import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import openpyxl  # for Excel export

##############################################################################
# 1) PRELOAD THE CSVs (PROFILES & PROBABILITY MATRICES)
##############################################################################
women_df = pd.read_csv("synthetic_women_profiles.csv")
men_df   = pd.read_csv("synthetic_men_profiles.csv")

prob_women_likes_men = pd.read_csv("probability_matrix_women_likes_men.csv", index_col=0)
prob_men_likes_women = pd.read_csv("probability_matrix_men_likes_women.csv", index_col=0)

# Create lookup dictionaries for profile info.
women_info = {row["WomanID"]: row for _, row in women_df.iterrows()}
men_info   = {row["ManID"]: row for _, row in men_df.iterrows()}

all_women_ids = list(women_info.keys())
all_men_ids   = list(men_info.keys())
all_user_ids  = all_women_ids + all_men_ids

##############################################################################
# 1.5) SELECT "JACK" AND "JILL" AS MIDDLE-PERFORMING PROFILES
##############################################################################
man_avgs = prob_women_likes_men.mean(axis=0)
overall_man_avg = man_avgs.mean()
jack_id = (man_avgs - overall_man_avg).abs().idxmin()

woman_avgs = prob_men_likes_women.mean(axis=0)
overall_woman_avg = woman_avgs.mean()
jill_id = (woman_avgs - overall_woman_avg).abs().idxmin()

print(f"Selected Jack: {jack_id}, Selected Jill: {jill_id}")

##############################################################################
# 2) THE SIMULATION FUNCTION
##############################################################################
def run_dating_simulation(
    daily_queue_size=5,
    incoming_order="FIFO",
    weight_reciprocal=1.0,
    weight_queue_penalty=0.5,
    export_trace=False,
    export_jack_jill_trace=False,
    show_match_plots=True,
    show_like_plots=True,
    plot_type="Bar"
):
    np.random.seed(42)
    random.seed(42)
    
    incoming_likes = {uid: [] for uid in all_user_ids}
    matches = {uid: set() for uid in all_user_ids}
    likes_sent = {uid: set() for uid in all_user_ids}
    
    daily_logs = []
    
    for day in range(1, 4):  # Fixed at 3 days
        day_records = []
        login_order = all_user_ids.copy()
        random.shuffle(login_order)
        
        for user in login_order:
            if user.startswith("W"):
                candidate_pool = [cid for cid in all_men_ids if cid not in matches[user]]
                get_prob = lambda cand: prob_women_likes_men.loc[user, cand]
                get_reciprocal = lambda cand: prob_men_likes_women.loc[cand, user]
            else:
                candidate_pool = [cid for cid in all_women_ids if cid not in matches[user]]
                get_prob = lambda cand: prob_men_likes_women.loc[user, cand]
                get_reciprocal = lambda cand: prob_women_likes_men.loc[cand, user]
            
            user_incoming = incoming_likes[user].copy()
            if incoming_order.upper() == "LIFO":
                user_incoming = list(reversed(user_incoming))
            num_incoming = min(len(user_incoming), daily_queue_size)
            incoming_queue = user_incoming[:num_incoming]
            incoming_likes[user] = user_incoming[num_incoming:]
            
            incoming_ids = [sender for (sender, _) in incoming_queue]
            candidate_pool = [cid for cid in candidate_pool if cid not in incoming_ids]
            
            num_fresh = daily_queue_size - num_incoming
            rec_scores = {}
            for cand in candidate_pool:
                base_prob = get_prob(cand)
                score = base_prob
                q = len(incoming_likes[cand])
                score *= 1 / (1 + weight_queue_penalty * q)
                reciprocal_prob = get_reciprocal(cand)
                score *= (reciprocal_prob ** weight_reciprocal)
                rec_scores[cand] = score
            
            if num_fresh > 0 and rec_scores:
                sorted_candidates = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
                fresh_candidates = [cand for cand, _ in sorted_candidates[:num_fresh]]
            else:
                fresh_candidates = []
            
            daily_queue = ([(sender, "incoming", sent_day) for (sender, sent_day) in incoming_queue] + 
                           [(cid, "fresh", day) for cid in fresh_candidates])
            
            for candidate, source, sent_day in daily_queue:
                if candidate in matches[user]:
                    continue
                like_prob = get_prob(candidate)
                final_p = like_prob / (like_prob + 1)  # Logistic transformation
                roll = np.random.rand()
                if roll < final_p:
                    if user in likes_sent[candidate]:
                        matches[user].add(candidate)
                        matches[candidate].add(user)
                    else:
                        likes_sent[user].add(candidate)
                        if source == "fresh":
                            incoming_likes[candidate].append((user, day))
                day_records.append({
                    "Day": day, "UserID": user, "CandidateID": candidate, "Decision": "Like" if roll < final_p else "Pass"
                })
        
        daily_logs.append(pd.DataFrame(day_records))
    
    return daily_logs, matches
