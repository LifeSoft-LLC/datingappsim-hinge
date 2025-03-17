from flask import Flask, request, render_template_string, url_for
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import os

if not os.path.exists("probability_matrix_women_likes_men.csv"):
    print("detected first run. Attempting to generate csv templates.")
    subprocess.run(["python", "init.py"], check=True)

import numpy as np 
from backend import run_dating_simulation, all_men_ids, all_women_ids, all_user_ids

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Parse parameters from the form
        try:
            incoming_order = request.form.get("incoming_order", "FIFO")
            daily_queue_size = int(request.form.get("daily_queue_size", 5))
            weight_reciprocal = float(request.form.get("weight_reciprocal", 1.0))
            weight_queue_penalty = float(request.form.get("weight_queue_penalty", 0.5))
            export_trace = request.form.get("export_trace") == "on"
            export_jack_jill_trace = request.form.get("export_jack_jill_trace") == "on"
            show_match_plots = request.form.get("show_match_plots") == "on"
            show_like_plots = request.form.get("show_like_plots") == "on"
            plot_type = request.form.get("plot_type", "Bar Chart")
        except ValueError:
            return "Invalid parameter(s) provided.", 400

        # Run the simulation
        daily_logs, matches, incoming_likes = run_dating_simulation(
            daily_queue_size=daily_queue_size,
            weight_reciprocal=weight_reciprocal,
            weight_queue_penalty=weight_queue_penalty,
            export_trace=export_trace,
            export_jack_jill_trace=export_jack_jill_trace,
            show_match_plots=show_match_plots,
            show_like_plots=show_like_plots,
            plot_type=plot_type
        )

        # Concatenate all daily logs into a single DataFrame
        full_log = pd.concat(daily_logs, ignore_index=True)

        # Compute summary metrics
        likes_by_men = full_log[
            (full_log["UserID"].str.startswith("M")) & (full_log["Decision"] == "Like")
        ].shape[0]
        likes_by_women = full_log[
            (full_log["UserID"].str.startswith("W")) & (full_log["Decision"] == "Like")
        ].shape[0]
        total_likes = likes_by_men + likes_by_women
        unique_matches = sum(len(matches[uid]) for uid in all_men_ids)
    
        # NEW METRICS: Unseen & Stale Unseen Likes.
        unseen_likes_men = 0
        unseen_likes_women = 0
        for uid in all_user_ids:
            for sender, sent_day in incoming_likes[uid]:
                if sender.startswith("M"):
                    unseen_likes_men += 1
                elif sender.startswith("W"):
                    unseen_likes_women += 1
        total_unseen = unseen_likes_men + unseen_likes_women
        
        stale_likes_men = 0
        stale_likes_women = 0
        for uid in all_user_ids:
            for sender, sent_day in incoming_likes[uid]:
                if sent_day != 3:
                    if sender.startswith("M"):
                        stale_likes_men += 1
                    elif sender.startswith("W"):
                        stale_likes_women += 1
        total_stale = stale_likes_men + stale_likes_women
        
        unseen_percent = (total_unseen / total_likes * 100) if total_likes > 0 else 0
        stale_percent = (total_stale / total_likes * 100) if total_likes > 0 else 0

        # Prepare summary HTML
        summary_html = f"""
        <div style='font-size:14px; line-height:1.5;'>
          <b>=== Hinge-Style Simulation Results ===</b><br>
          <br>
          <b>Total Likes Sent:</b> {total_likes}<br>
          - Likes by men: {likes_by_men}<br>
          - Likes by women: {likes_by_women}<br><br>
          <b>Total Unseen Likes Sent:</b> {total_unseen} ({unseen_percent:.2f}%)<br>
          - Likes by men: {unseen_likes_men}<br>
          - Likes by women: {unseen_likes_women}<br><br>
          <b>Total Stale Unseen Likes Sent:</b> {total_stale} ({stale_percent:.2f}%)<br>
          - Likes by men: {stale_likes_men}<br>
          - Likes by women: {stale_likes_women}<br><br>
          <b>Matches Created: <span style="color:purple; font-size:20px;">{unique_matches}</span></b>
        </div>
        """

        # Generate plots
        plot_img = None
        if show_match_plots or show_like_plots:
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14,15))

            men_matches = sorted([(uid, len(matches[uid])) for uid in all_men_ids], key=lambda x: x[1])
            women_matches = sorted([(uid, len(matches[uid])) for uid in all_women_ids], key=lambda x: x[1])
            
            men_likes_sent = []
            for uid, _ in men_matches:
                count = full_log[(full_log["UserID"]==uid) & (full_log["Decision"]=="Like")].shape[0]
                men_likes_sent.append(count)
            women_likes_sent = []
            for uid, _ in women_matches:
                count = full_log[(full_log["UserID"]==uid) & (full_log["Decision"]=="Like")].shape[0]
                women_likes_sent.append(count)
            
            # Prepare likes received counts (sorted by match count)
            men_likes_received = []
            for uid, _ in men_matches:
                count = full_log[(full_log["CandidateID"]==uid) & (full_log["Decision"]=="Like")].shape[0]
                men_likes_received.append(count)
            women_likes_received = []
            for uid, _ in women_matches:
                count = full_log[(full_log["CandidateID"]==uid) & (full_log["Decision"]=="Like")].shape[0]
                women_likes_received.append(count)
            
            def compute_hist_counts(data):
                data = np.array(data)
                bin0 = np.sum(data == 0)
                bin1 = np.sum((data >= 1) & (data <= 2))
                bin2 = np.sum((data >= 3) & (data <= 4))
                bin3 = np.sum(data >= 5)
                return [bin0, bin1, bin2, bin3]
            bin_labels = ["0", "1-2", "3-4", "5+"]

            if plot_type == "Bar Chart":
              # Match plots - Bar Chart
              if show_match_plots:
                  axes[0,0].bar(range(len(men_matches)), [x[1] for x in men_matches],
                                color="skyblue", edgecolor="black")
                  axes[0,0].set_title("Men's Match Counts (Sorted)")
                  axes[0,0].set_xlabel("Men (sorted by match count)")
                  axes[0,0].set_ylabel("Number of Matches")
                  
                  axes[0,1].bar(range(len(women_matches)), [x[1] for x in women_matches],
                                color="lightpink", edgecolor="black")
                  axes[0,1].set_title("Women's Match Counts (Sorted)")
                  axes[0,1].set_xlabel("Women (sorted by match count)")
                  axes[0,1].set_ylabel("Number of Matches")
              else:
                  axes[0,0].axis('off')
                  axes[0,1].axis('off')
              
              # Likes Sent plots - Bar Chart
              if show_like_plots:
                  axes[1,0].bar(range(len(men_matches)), men_likes_sent,
                                color="skyblue", edgecolor="black")
                  axes[1,0].set_title("Men's Likes Sent (Sorted by Match Count)")
                  axes[1,0].set_xlabel("Men (sorted by match count)")
                  axes[1,0].set_ylabel("Number of Likes Sent")
                  
                  axes[1,1].bar(range(len(women_matches)), women_likes_sent,
                                color="lightpink", edgecolor="black")
                  axes[1,1].set_title("Women's Likes Sent (Sorted by Match Count)")
                  axes[1,1].set_xlabel("Women (sorted by match count)")
                  axes[1,1].set_ylabel("Number of Likes Sent")
              else:
                  axes[1,0].axis('off')
                  axes[1,1].axis('off')
              
              # Likes Received plots - Bar Chart
              if show_like_plots:
                  axes[2,0].bar(range(len(men_matches)), men_likes_received,
                                color="skyblue", edgecolor="black")
                  axes[2,0].set_title("Men's Likes Received (Sorted by Match Count)")
                  axes[2,0].set_xlabel("Men (sorted by match count)")
                  axes[2,0].set_ylabel("Number of Likes Received")
                  
                  axes[2,1].bar(range(len(women_matches)), women_likes_received,
                                color="lightpink", edgecolor="black")
                  axes[2,1].set_title("Women's Likes Received (Sorted by Match Count)")
                  axes[2,1].set_xlabel("Women (sorted by match count)")
                  axes[2,1].set_ylabel("Number of Likes Received")
              else:
                  axes[2,0].axis('off')
                  axes[2,1].axis('off')

            elif plot_type == "Histogram":
              men_match_hist = compute_hist_counts([x[1] for x in men_matches])
              women_match_hist = compute_hist_counts([x[1] for x in women_matches])
              men_likes_hist = compute_hist_counts(men_likes_sent)
              women_likes_hist = compute_hist_counts(women_likes_sent)
              men_likes_received_hist = compute_hist_counts(men_likes_received)
              women_likes_received_hist = compute_hist_counts(women_likes_received)
              
              # Men's match histogram
              if show_match_plots:
                  axes[0,0].bar(range(len(men_match_hist)), men_match_hist,
                                color="skyblue", edgecolor="black", width=0.8)
                  axes[0,0].set_title("Histogram of Men's Match Counts")
                  axes[0,0].set_xlabel("Match Count Bins")
                  axes[0,0].set_ylabel("Number of Men")
                  axes[0,0].set_xticks(range(len(bin_labels)))
                  axes[0,0].set_xticklabels(bin_labels)
                  
                  axes[0,1].bar(range(len(women_match_hist)), women_match_hist,
                                color="lightpink", edgecolor="black", width=0.8)
                  axes[0,1].set_title("Histogram of Women's Match Counts")
                  axes[0,1].set_xlabel("Match Count Bins")
                  axes[0,1].set_ylabel("Number of Women")
                  axes[0,1].set_xticks(range(len(bin_labels)))
                  axes[0,1].set_xticklabels(bin_labels)
              else:
                  axes[0,0].axis('off')
                  axes[0,1].axis('off')
              
              # Men's likes sent histogram
              if show_like_plots:
                  axes[1,0].bar(range(len(men_likes_hist)), men_likes_hist,
                                color="skyblue", edgecolor="black", width=0.8)
                  axes[1,0].set_title("Histogram of Men's Likes Sent")
                  axes[1,0].set_xlabel("Likes Sent Count Bins")
                  axes[1,0].set_ylabel("Number of Men")
                  axes[1,0].set_xticks(range(len(bin_labels)))
                  axes[1,0].set_xticklabels(bin_labels)
                  
                  axes[1,1].bar(range(len(women_likes_hist)), women_likes_hist,
                                color="lightpink", edgecolor="black", width=0.8)
                  axes[1,1].set_title("Histogram of Women's Likes Sent")
                  axes[1,1].set_xlabel("Likes Sent Count Bins")
                  axes[1,1].set_ylabel("Number of Women")
                  axes[1,1].set_xticks(range(len(bin_labels)))
                  axes[1,1].set_xticklabels(bin_labels)
              else:
                  axes[1,0].axis('off')
                  axes[1,1].axis('off')
                  
              # Men's likes received histogram
              if show_like_plots:
                  axes[2,0].bar(range(len(men_likes_received_hist)), men_likes_received_hist,
                                color="skyblue", edgecolor="black", width=0.8)
                  axes[2,0].set_title("Histogram of Men's Likes Received")
                  axes[2,0].set_xlabel("Likes Received Count Bins")
                  axes[2,0].set_ylabel("Number of Men")
                  axes[2,0].set_xticks(range(len(bin_labels)))
                  axes[2,0].set_xticklabels(bin_labels)
                  
                  axes[2,1].bar(range(len(women_likes_received_hist)), women_likes_received_hist,
                                color="lightpink", edgecolor="black", width=0.8)
                  axes[2,1].set_title("Histogram of Women's Likes Received")
                  axes[2,1].set_xlabel("Likes Received Count Bins")
                  axes[2,1].set_ylabel("Number of Women")
                  axes[2,1].set_xticks(range(len(bin_labels)))
                  axes[2,1].set_xticklabels(bin_labels)
              else:
                  axes[2,0].axis('off')
                  axes[2,1].axis('off')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format="svg")
            buf.seek(0)
            plot_img = base64.b64encode(buf.getvalue()).decode("utf8")
            plt.close(fig)
        # TODO: add full simulation trace exports as xlsx, when ready; use download prop
        # Jack & Jill traces too
        return render_template_string("""
        <!DOCTYPE html>
        <html>
          <head>
            <title>Hinge-Style Simulation Results</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 40px; }
              .summary { margin-bottom: 30px; }
            </style>
          </head>
          <body>
            <div class="summary">
              {{ summary_html|safe }}
            </div>
            {% if plot_img %}
            <div>
              <h3>Match Distribution Plots</h3>
              <img src="data:image/svg+xml;base64,{{ plot_img }}" alt="Plots">
            </div>
            {% endif %}
            <div style="margin-top: 20px;">
              <a href="{{ url_for('index') }}">Run another simulation</a>
            </div>
          </body>
        </html>
        """, summary_html=summary_html, plot_img=plot_img)

    return render_template_string("""
    <!DOCTYPE html>
    <html>
      <head>
        <title>Hinge-Style Simulation</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; }
          form { max-width: 400px; }
          label { display: block; margin-top: 15px; }
          input[type="number"], input[type="text"], select { width: 100%; padding: 8px; }
          input[type="submit"] { margin-top: 20px; padding: 10px 20px; }
        </style>
      </head>
      <body>
        <h1>A "Hinge-like" dating simulation</h1>
        <p>This simulation is similar to the one you saw during class. It is designed to replicate the dynamics of a (highly simplified) dating platform.
        <p>As before, the simulation runs for three "days." Each day, the same 100 men and women (200 people in total), each with their own profiles and preferences, "log into" the platform in a random order and swipe right (like) or left (pass) on five profiles of the opposite sex.</p>
        <p>The difference this time is that the simulation more closely follows the design of the the dating app Hinge. Rather than a single recommendations list of profiles, we imagine that users begin on a separate "Incoming likes" tab. This tab *only* shows profiles that are incoming likes.</p>
        <p>Once the user exhausts their "Incoming likes" tab, if they have any leftover "capacity" (that is, if they have not yet processed 5 profiles that day), they move to a general browsing tab showing fresh profiles from among the remaining candidates, and they continue to like or pass until they reach their daily limit of 5.</p>
        <p>The order of each user's "Incoming likes" tab can be either FIFO or LIFO (the real Hinge uses LIFO). Simultaneously, the order of each user's recommended profiles is determined by the platform's algorithm. Just like in class, that algorithm contains two "weights" that will be chosen by you.
        <ul>
          <li>The first is w<sub>reciprocal</sub> — by increasing this weight, you will <u><em>increase</em></u> the likelihood of being shown for candidate with a higher likelihood of liking back.</li>
          <li>The second is w<sub>queue</sub> — by increasing this weight, you will <u><em>decrease</em></u> the likelihood of being shown for a candidate with a higher number of pending likes.</p>
        </ul>
        <p>(If you're interested in the full equation of the algorithm, please see the slide deck for Session 2 on Canvas.)</p>
        <p>As before, we report likes, "stale" likes (not seen for more than one day), "unseen" likes (not seen at all), and matches. Results can be reported as either a bar chart (one bar is one man or one woman) or as a histogram of counts.</p>
        <h2>Hinge-Style Simulation Parameters</h2>
        <form method="post">
          <label for="incoming_order">Incoming Queue Order:</label>
          <select id="incoming_order" name="incoming_order">
            <option value="FIFO" selected>FIFO</option>
            <option value="LIFO">LIFO</option>
          </select>
          <label for="weight_reciprocal">Reciprocal Weight (w<sub>reciprocal</sub>):</label>
          <input type="number" id="weight_reciprocal" name="weight_reciprocal" value="0.0" step="0.1" min="0" max="5.0">
          
          <label for="weight_queue_penalty">Queue Penalty Weight (w<sub>queue</sub>):</label>
          <input type="number" id="weight_queue_penalty" name="weight_queue_penalty" value="0.0" step="0.1" min="0" max="2.0">
          
          <label>
            <input type="checkbox" name="export_trace" disabled>
            Export Excel Trace?
          </label>
          
          <label>
            <input type="checkbox" name="export_jack_jill_trace" disabled>
            Export Jack & Jill Trace?
          </label>
          
          <label>
            <input type="checkbox" name="show_match_plots" checked>
            Show Match Plots?
          </label>
          
          <label>
            <input type="checkbox" name="show_like_plots" checked>
            Show Like Plots?
          </label>

          <label for="plot_type">Plot Type:</label>
          <select id="plot_type" name="plot_type">
            <option value="Bar Chart">Bar Chart</option>
            <option value="Histogram">Histogram</option>
          </select>

          <input type="submit" value="Run Simulation">
        </form>
      </body>
    </html>
    """)

if __name__ == "__main__":
    app.run(debug=True)
