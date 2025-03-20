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
# 2) THE HINGE-LIKE SIMULATION FUNCTION WITH PERSISTENT UPDATING
##############################################################################
def run_dating_simulation(
    num_days=3,
    daily_queue_size=5,
    incoming_order="FIFO",  # "FIFO" or "LIFO"
    weight_reciprocal=1.0,
    weight_queue_penalty=0.5,
    random_seed=42,
    export_trace=False,
    export_jack_jill_trace=False,
    show_match_plots=True,
    show_like_plots=True,
    plot_type="Bar Chart",
    summary_out=None,
    plot_out=None,
    trace_out=None,
    trace_jj_out=None
):
    """
    Runs the Hinge-like dating app simulation (with live, persistent updating).
    The summary text and plots are updated in place using dedicated output widgets.
    
    Simulation details:
      - Each user sees a daily queue composed of incoming likes (taken in either FIFO or LIFO order)
        and fresh recommendations (scored as:
          S₍ᵢⱼ₎ = Pᵢⱼ * 1/(1 + w_queue*Qⱼ) * (Pⱼᵢ)^(w_reciprocal)
        where Qⱼ is the number of pending likes for candidate j).
      - For fresh candidates, likes are added to the recipient's incoming queue (with today's timestamp).
    
    NEW METRICS DEFINITIONS:
      - Unseen Likes: count of likes that were never seen by the recipient.
      - Stale Unseen Likes: count of unseen likes that were not sent on day 3.
    
    Plotting Options:
      - Match Plots: displays matches per man/woman.
      - Like Plots: displays likes sent per man/woman.
      - Plot Type: "Bar Chart" (individual counts) or "Histogram" (aggregated bins).
    """
    # Set seeds for reproducibility.
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # ----- Simulation State Dictionaries -----
    # For incoming likes, store (sender, day_sent)
    incoming_likes = {uid: [] for uid in all_user_ids}
    matches = {uid: set() for uid in all_user_ids}
    likes_sent = {uid: set() for uid in all_user_ids}
    daily_logs = []
    
    # NEW: Track who each user has already seen, so they can't appear again on future days
    already_seen = {uid: set() for uid in all_user_ids}

    # Simulation loop
    for day in range(1, num_days + 1):
        day_records = []
        login_order = all_user_ids.copy()
        random.shuffle(login_order)
        
        for user in login_order:
            # Build candidate pool: opposite sex, not matched, not previously seen
            if user.startswith("W"):
                candidate_pool = [
                    cid for cid in all_men_ids
                    if cid not in matches[user] and cid not in already_seen[user]
                ]
                get_prob = lambda cand: prob_women_likes_men.loc[user, cand]
                get_reciprocal = lambda cand: prob_men_likes_women.loc[cand, user]
            else:
                candidate_pool = [
                    cid for cid in all_women_ids
                    if cid not in matches[user] and cid not in already_seen[user]
                ]
                get_prob = lambda cand: prob_men_likes_women.loc[user, cand]
                get_reciprocal = lambda cand: prob_women_likes_men.loc[cand, user]
            
            # (a) Build incoming likes portion (FIFO or LIFO).
            user_incoming = incoming_likes[user].copy()
            if incoming_order.upper() == "LIFO":
                user_incoming.reverse()
            num_incoming = min(len(user_incoming), daily_queue_size)
            incoming_queue = user_incoming[:num_incoming]
            
            # Remove these processed incoming likes from the user's queue
            incoming_likes[user] = user_incoming[num_incoming:]
            
            # (b) Build fresh recommendations. Exclude any candidate already in incoming_queue.
            incoming_ids = [sender for (sender, _) in incoming_queue]
            candidate_pool = [cid for cid in candidate_pool if cid not in incoming_ids]
            num_fresh = daily_queue_size - num_incoming
            
            rec_scores = {}
            for cand in candidate_pool:
                base_prob = get_prob(cand)
                q = len(incoming_likes[cand])  # number of pending likes for cand
                score = base_prob * (1 / (1 + weight_queue_penalty * q)) \
                        * (get_reciprocal(cand) ** weight_reciprocal)
                rec_scores[cand] = score
            
            if num_fresh > 0 and rec_scores:
                sorted_candidates = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)
                fresh_candidates = [cand for cand, _ in sorted_candidates[:num_fresh]]
            else:
                fresh_candidates = []
            
            # (c) Combine queue: incoming + fresh
            daily_queue = ([(sender, "incoming", sent_day) for (sender, sent_day) in incoming_queue] +
                           [(cid, "fresh", day) for cid in fresh_candidates])
            
            # (d) Process each candidate in daily_queue
            for candidate, source, sent_day in daily_queue:
                # If already matched, skip
                if candidate in matches[user]:
                    continue
                
                like_prob = get_prob(candidate)
                roll = np.random.rand()
                decision = "Pass"
                match_formed = False
                
                # Decide Like or Pass
                if roll < like_prob:
                    decision = "Like"
                    if user in likes_sent[candidate]:
                        # That forms a match
                        match_formed = True
                        matches[user].add(candidate)
                        matches[candidate].add(user)
                    else:
                        likes_sent[user].add(candidate)
                        if source == "fresh":
                            incoming_likes[candidate].append((user, day))
                
                delay = day - sent_day
                day_records.append({
                    "Day": day,
                    "UserID": user,
                    "CandidateID": candidate,
                    "QueueType": source,
                    "LikeProbability": like_prob,
                    "RandomRoll": roll,
                    "Decision": decision,
                    "MatchFormed": match_formed,
                    "Delay": delay
                })

                # Mark this candidate as already seen by this user
                already_seen[user].add(candidate)
        
        daily_logs.append(pd.DataFrame(day_records))
    
    return daily_logs, matches, incoming_likes