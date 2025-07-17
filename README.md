# Cognitive Load Detection from Typing Patterns

This project detects **cognitive load** based on how a user types. It uses simulated typing behavior, extracts features like dwell time, flight time, speed, pauses, and errors, and predicts the mental load using a machine learning model.

# Features
- Simulates typing behavior and extracts features in real-time.
- Trained ML model predicts cognitive load: `low` or `high`.
- Visual EDA dashboard using **Streamlit**.
- Live typing + analysis + auto-save to dataset.
- Real-time typing input logger
- Live feature extraction (typing speed, pauses, word length)
- Machine Learning model prediction (High/Low Cognitive Load)
- EDA insights from recorded dataset
- Clean UI with Streamlit

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
  
# ML Model Info
- Features used: typing speed, word length, pause duration
- Model: RandomForest Classisfier (or any ML model)
- Accuracy: ~95-100% on validation data

# Future Scope
- Expand to multitasking tests (switching tasks)
- Use webcam to analyze facial fatigue + typing
- Add audio-based analysis for stress signals

##  Author
Sumanth Arisinapalli
🔗 GitHub
---

