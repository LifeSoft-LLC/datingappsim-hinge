[ ! -f "probability_matrix_men_likes_women.csv" ] && python3 init.py 

gunicorn -w 2 -b 0.0.0.0:8000 frontend_flask:app