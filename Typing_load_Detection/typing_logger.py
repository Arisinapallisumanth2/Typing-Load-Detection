from pynput import keyboard
import time
import csv
import pandas as pd

#configuration variables
cognitive_load = input("Cognitive Load [low/high]: ").strip().lower() #it changes high depends on typing speed
output_file = f"keystroke_data_{cognitive_load}.csv"
FEATURES_SUMMARY_FILE = "features_summary.csv"


'''
COGNITIVE_LOAD: Manual label to tag the condition during data collection.
"low" → normal/relaxed state
"high" → distracted/stressed typing
OUTPUT_FILE: Output file name changes based on the label
'''

#Intialize Data score
keystroke_data = []   #A list to store keystroke info (each key press/release).
last_release_time = None  #Will be used to calculate flight time (delay between key release and next key press).

#key press Handler
def on_press(key):
    try:
        key_char = key.char   #Tries to get the character from the key.
    except AttributeError:
        key_char = str(key)   #if key.char causes an AttributeError (like for Key.shift or Key.esc) it dfalls the str(key)
        
    press_time = time.time()  #Records the exact time the key was pressed.
    
    flight_time = None
    
    if last_release_time: 
        flight_time = press_time - last_release_time  #Time between last key released and current key pressed.
        
    #Stores key press event details (key, time, flight time, load label) into the list.
    keystroke_data.append({
        "key":key_char,
        "event":"press",
        "timestamp":press_time,
        "flight_time":flight_time,
        "cognitive_load": cognitive_load
    })
    
    
#key release Handler
def on_release(key):
    global last_release_time
    try:
        key_char = key.char
    except AttributeError:
        key_char = str(key)
    
    release_time = time.time()         # same as press it captures and time and character
    last_release_time = release_time    #updated here to be used for the next flight time.
    
    keystroke_data.append({
        "key": key_char,
        "event": "release",
        "timestamp": release_time,
        "flight_time": None,
        "cognitive_load": cognitive_load
    })
    
    # Stop logging on ESC key
    if key == keyboard.Key.esc:   #This stops the recording when the ESC key is pressed.
        return False
    
#save to csv file
def save_data():
    fieldnames = ['key', 'event', 'timestamp', 'flight_time', 'cognitive_load']
    with open(output_file,mode='w',newline="") as f:   # Opens a file with the name stored in OUTPUT_FILE| Assigns the file object to the variable f.
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()   #Writes the column headers (fieldnames) to the CSV file.
        writer.writerows(keystroke_data)  #Writes all the dictionaries in keystroke_data as individual rows in the CSV.
    print(f"Data saved to {output_file}")

# === FEATURE EXTRACTION ===
def extract_features(csv_file):
    df = pd.read_csv(csv_file)

    press_df = df[df['event'] == 'press'].reset_index(drop=True)
    release_df = df[df['event'] == 'release'].reset_index(drop=True)

    # 1. Average Dwell Time
    dwell_times = []
    for i in range(min(len(press_df), len(release_df))):
        if press_df.loc[i, 'key'] == release_df.loc[i, 'key']:
            dwell = release_df.loc[i, 'timestamp'] - press_df.loc[i, 'timestamp']
            dwell_times.append(dwell)
    avg_dwell = sum(dwell_times) / len(dwell_times) if dwell_times else 0

    # 2. Average Flight Time
    flight_times = press_df['flight_time'].dropna().tolist()
    avg_flight = sum(flight_times) / len(flight_times) if flight_times else 0

    # 3. Typing Speed (chars per sec)
    total_time = df['timestamp'].max() - df['timestamp'].min()
    total_chars = len(press_df[~press_df['key'].str.startswith("Key.")])
    speed = total_chars / total_time if total_time > 0 else 0

    # 4. Pause Count (>0.5 sec)
    pauses = [f for f in flight_times if f > 0.5]
    pause_count = len(pauses)

    # 5. Error Rate (Backspaces)
    error_count = len(press_df[press_df['key'] == 'Key.backspace'])

    # 6. Label
    label = df['cognitive_load'].iloc[0]

    features = {
        'avg_dwell': round(avg_dwell, 3),
        'avg_flight': round(avg_flight, 3),
        'speed': round(speed, 2),
        'pauses': pause_count,
        'errors': error_count,
        'label': label
    }

    return features

def main():
    print(f"\nStart typing... (Cognitive Load: {cognitive_load})")
    print("Press ESC to stop.\n")
        
    with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
        listener.join()
    
    #When a key is pressed, it will call the on_press() function (which you already defined earlier).
    #When a key is released, it will call the on_release() function.
    #When you press the ESC key, the on_release() function returns False, which tells the listener to stop.
    #.join() keeps the program running and listening for key events until it's manually stopped.
    
    print("Typing session complted")
    save_data()
    
    # Extract features
    features = extract_features(output_file)
    print("\n🔍 Extracted Features:")
    for k, v in features.items():
        print(f"{k}: {v}")

    # Optional: Save features to summary CSV
    try:
        df_summary = pd.read_csv(FEATURES_SUMMARY_FILE)
    except FileNotFoundError:
        df_summary = pd.DataFrame()

    df_summary = pd.concat([df_summary, pd.DataFrame([features])], ignore_index=True)
    df_summary.to_csv(FEATURES_SUMMARY_FILE, index=False)
    print(f"\n✅ Features saved to: {FEATURES_SUMMARY_FILE}")
        
if __name__ == "__main__":
    main()
