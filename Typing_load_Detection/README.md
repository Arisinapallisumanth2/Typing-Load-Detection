# Cognitive Load Detection from Typing Patterns

This project detects **cognitive load** based on how a user types. It uses simulated typing behavior, extracts features like dwell time, flight time, speed, pauses, and errors, and predicts the mental load using a machine learning model.

# Features
- Simulates typing behavior and extracts features in real-time.
- Trained ML model predicts cognitive load: `low` or `high`.
- Visual EDA dashboard using **Streamlit**.
- Live typing + analysis + auto-save to dataset.

# Technologies Used
- Python, Streamlit
- scikit-learn (Random Forest)
- pandas, seaborn, matplotlib
- joblib (for model persistence)

# Features Extracted
- `avg_dwell`: Average key hold time
- `avg_flight`: Time between key presses
- `speed`: Typing speed (characters/sec)
- `pauses`: Count of idle pauses
- `errors`: Simulated errors or backspace use

