# frontend.py
"""
Flask Frontend for Tinder-Style Simulation (Interactive Version)

This Flask app provides a web form for configuring and running the simulation.
It uses Plotly to generate interactive bar charts for match distributions.
Additionally, it provides download buttons for the full simulation trace and
the Jack & Jill trace as Excel files.
"""

from flask import Flask, request, render_template_string, url_for, send_file
import io
import os

# Import the simulation backend and relevant globals.
from backend import run_tinder_simulation, all_men_ids, all_women_ids, all_user_ids

# For interactive plots.
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Parse parameters from the form.
        try:
            num_days = int(request.form.get("num_days", 3))
            daily_queue_size = int(request.form.get("daily_queue_size", 5))
            weight_queue_penalty = float(request.form.get("weight_queue_penalty", 0.5))
            weight_reciprocal = float(request.form.get("weight_reciprocal", 1.0))
            random_seed = int(request.form.get("random_seed", 42))
        except ValueError:
            return "Invalid parameter(s) provided.", 400

        # Run the simulation with export enabled (for download buttons later)
        full_log, matches = run_tinder_simulation(
            num_days=num_days,
            daily_queue_size=daily_queue_size,
            weight_queue_penalty=weight_queue_penalty,
            weight_reciprocal=weight_reciprocal,
            random_seed=random_seed,
            export_trace=True,
            export_jack_jill_trace=True,
            show_plots=False,  # We'll generate our own interactive plots below.
            summary_out=None,
            plot_out=None,
            trace_out=None,
            trace_jj_out=None
        )

        # Compute summary metrics.
        likes_by_men = full_log[
            (full_log["UserID"].str.startswith("M")) & (full_log["Decision"]=="Like")
        ].shape[0]
        likes_by_women = full_log[
            (full_log["UserID"].str.startswith("W")) & (full_log["Decision"]=="Like")
        ].shape[0]
        total_likes = likes_by_men + likes_by_women
        unique_matches = sum(len(matches[uid]) for uid in all_men_ids)

        summary_html = f"""
        <h2>Tinder-Style Simulation Results (Seed = {random_seed})</h2>
        <p><b>Days:</b> {num_days} &nbsp;&nbsp;
           <b>Daily Queue Size:</b> {daily_queue_size}</p>
        <p><b>Total Likes Sent:</b> {total_likes} 
           (Men: {likes_by_men}, Women: {likes_by_women})</p>
        <p><b>Unique Matches Created:</b> {unique_matches}</p>
        """

        # Build interactive plots using Plotly.
        men_matches = sorted([(uid, len(matches[uid])) for uid in all_men_ids],
                             key=lambda x: x[1])
        women_matches = sorted([(uid, len(matches[uid])) for uid in all_women_ids],
                               key=lambda x: x[1])

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Men's Match Counts (Sorted)",
                                            "Women's Match Counts (Sorted)"])
        fig.add_trace(
            go.Bar(x=[x[0] for x in men_matches],
                   y=[x[1] for x in men_matches],
                   marker_color="skyblue",
                   name="Men"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=[x[0] for x in women_matches],
                   y=[x[1] for x in women_matches],
                   marker_color="lightpink",
                   name="Women"),
            row=1, col=2
        )
        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        # Convert the Plotly figure to an HTML div.
        plot_div = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

        # Render the results page with summary, interactive plots, and export buttons.
        return render_template_string("""
        <!DOCTYPE html>
        <html>
          <head>
            <title>Tinder-Style Simulation Results</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 40px; }
              .summary { margin-bottom: 30px; }
              .download-buttons a {
                  display: inline-block;
                  padding: 10px 20px;
                  margin-right: 10px;
                  background-color: #4CAF50;
                  color: white;
                  text-decoration: none;
                  border-radius: 4px;
              }
              .download-buttons a:hover {
                  background-color: #45a049;
              }
            </style>
          </head>
          <body>
            <div class="summary">
              {{ summary_html|safe }}
            </div>
            <div>
              <h3>Interactive Match Distribution Plots</h3>
              {{ plot_div|safe }}
            </div>
            <div class="download-buttons" style="margin-top:20px;">
              <a href="{{ url_for('download_full_trace') }}">Download Full Trace Excel</a>
              <a href="{{ url_for('download_jack_jill_trace') }}">Download Jack &amp; Jill Trace Excel</a>
            </div>
            <div style="margin-top: 20px;">
              <a href="{{ url_for('index') }}">Run another simulation</a>
            </div>
          </body>
        </html>
        """, summary_html=summary_html, plot_div=plot_div)
    else:
        # GET request: Render the simulation parameters form.
        return render_template_string("""
        <!DOCTYPE html>
        <html>
          <head>
            <title>Tinder-Style Simulation</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 40px; }
              form { max-width: 400px; }
              label { display: block; margin-top: 15px; }
              input[type="number"], input[type="text"] { width: 100%; padding: 8px; }
              input[type="submit"] { margin-top: 20px; padding: 10px 20px; }
            </style>
          </head>
          <body>
            <h2>Tinder-Style Simulation Parameters</h2>
            <form method="post">
              <label for="num_days">Days:</label>
              <input type="number" id="num_days" name="num_days" value="3" min="1" max="7">
              
              <label for="daily_queue_size">Daily Queue Size:</label>
              <input type="number" id="daily_queue_size" name="daily_queue_size" value="5" min="3" max="10">
              
              <label for="weight_queue_penalty">Queue Penalty Weight:</label>
              <input type="number" id="weight_queue_penalty" name="weight_queue_penalty" value="0.5" step="0.1" min="0" max="2.0">
              
              <label for="weight_reciprocal">Reciprocal Weight:</label>
              <input type="number" id="weight_reciprocal" name="weight_reciprocal" value="1.0" step="0.1" min="0" max="5.0">
              
              <label for="random_seed">Random Seed:</label>
              <input type="number" id="random_seed" name="random_seed" value="42" min="1" max="5000">
              
              <input type="submit" value="Run Simulation">
            </form>
          </body>
        </html>
        """)

# Endpoints to download the exported Excel files.
@app.route("/download/full_trace")
def download_full_trace():
    # For a production system, you should generate unique file names per session.
    filename = "tinder_simulation_trace.xlsx"
    if not os.path.exists(filename):
        return "File not found. Run a simulation first.", 404
    return send_file(filename, as_attachment=True)

@app.route("/download/jack_jill_trace")
def download_jack_jill_trace():
    filename = "tinder_simulation_jack_jill_trace.xlsx"
    if not os.path.exists(filename):
        return "File not found. Run a simulation first.", 404
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)