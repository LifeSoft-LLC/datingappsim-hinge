from flask import Flask, request, render_template_string, url_for
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import os
from backend import run_dating_simulation, all_men_ids, all_women_ids

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Parse parameters from the form
        try:
            daily_queue_size = int(request.form.get("daily_queue_size", 5))
            weight_reciprocal = float(request.form.get("weight_reciprocal", 1.0))
            weight_queue_penalty = float(request.form.get("weight_queue_penalty", 0.5))
            export_trace = request.form.get("export_trace") == "on"
            export_jack_jill_trace = request.form.get("export_jack_jill_trace") == "on"
            show_match_plots = request.form.get("show_match_plots") == "on"
            show_like_plots = request.form.get("show_like_plots") == "on"
            plot_type = request.form.get("plot_type", "Bar")
        except ValueError:
            return "Invalid parameter(s) provided.", 400

        # Run the simulation
        daily_logs, matches = run_dating_simulation(
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

        # Prepare summary HTML
        summary_html = f"""
        <h2>Tinder-Style Simulation Results</h2>
        <p><b>Days:</b> 3 (fixed) &nbsp;&nbsp;
           <b>Daily Queue Size:</b> {daily_queue_size}</p>
        <p><b>Total Likes Sent:</b> {total_likes} 
           (Men: {likes_by_men}, Women: {likes_by_women})</p>
        <p><b>Unique Matches Created:</b> {unique_matches}</p>
        """

        # Generate match and like distribution plots
        plot_img = None
        if show_match_plots or show_like_plots:
            fig, axes = plt.subplots(ncols=2, figsize=(14, 5))

            # Sort match counts for men and women
            men_matches = sorted([(uid, len(matches[uid])) for uid in all_men_ids], key=lambda x: x[1])
            women_matches = sorted([(uid, len(matches[uid])) for uid in all_women_ids], key=lambda x: x[1])

            if plot_type == "Histogram":
                axes[0].hist([x[1] for x in men_matches], bins=[0, 1, 2, 3, 4, 5], edgecolor="black")
                axes[1].hist([x[1] for x in women_matches], bins=[0, 1, 2, 3, 4, 5], edgecolor="black")
            else:  # Default to Bar Chart
                axes[0].bar(range(len(men_matches)), [x[1] for x in men_matches], color="skyblue", edgecolor="black")
                axes[1].bar(range(len(women_matches)), [x[1] for x in women_matches], color="lightpink", edgecolor="black")

            axes[0].set_title("Men's Match Counts (Sorted)")
            axes[0].set_xlabel("Men (sorted by match count)")
            axes[0].set_ylabel("Number of Matches")

            axes[1].set_title("Women's Match Counts (Sorted)")
            axes[1].set_xlabel("Women (sorted by match count)")
            axes[1].set_ylabel("Number of Matches")

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_img = base64.b64encode(buf.getvalue()).decode("utf8")
            plt.close(fig)

        return render_template_string("""
        <!DOCTYPE html>
        <html>
          <head>
            <title>Tinder-Style Simulation Results</title>
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
              <img src="data:image/png;base64,{{ plot_img }}" alt="Plots">
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
        <title>Tinder-Style Simulation</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; }
          form { max-width: 400px; }
          label { display: block; margin-top: 15px; }
          input[type="number"], input[type="text"], select { width: 100%; padding: 8px; }
          input[type="submit"] { margin-top: 20px; padding: 10px 20px; }
        </style>
      </head>
      <body>
        <h2>Tinder-Style Simulation Parameters</h2>
        <form method="post">
          <label for="daily_queue_size">Daily Queue Size:</label>
          <input type="number" id="daily_queue_size" name="daily_queue_size" value="5" min="3" max="10">
          
          <label for="weight_reciprocal">Reciprocal Weight:</label>
          <input type="number" id="weight_reciprocal" name="weight_reciprocal" value="1.0" step="0.1" min="0" max="5.0">
          
          <label for="weight_queue_penalty">Queue Penalty Weight:</label>
          <input type="number" id="weight_queue_penalty" name="weight_queue_penalty" value="0.5" step="0.1" min="0" max="2.0">
          
          <label>
            <input type="checkbox" name="export_trace">
            Export Excel Trace?
          </label>
          
          <label>
            <input type="checkbox" name="export_jack_jill_trace">
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
            <option value="Bar">Bar Chart</option>
            <option value="Histogram">Histogram</option>
          </select>

          <input type="submit" value="Run Simulation">
        </form>
      </body>
    </html>
    """)

if __name__ == "__main__":
    app.run(debug=True)
