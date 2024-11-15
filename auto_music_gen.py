import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from music21 import *
import glob
from tqdm import tqdm
import numpy as np
import random
import threading  # For threading support
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from sklearn.model_selection import train_test_split
import warnings  # Import Python's warnings module

# Suppress TranslateWarning
try:
    from music21 import TranslateWarning
    warnings.filterwarnings("ignore", category=TranslateWarning)
except ImportError:
    warnings.filterwarnings("ignore", module="music21")


class MusicGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Generator")
        self.root.geometry("600x400")
        self.root.config(bg="#282828")

        # Style
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 12), padding=10)
        style.configure('TLabel', font=('Helvetica', 14), foreground="white", background="#282828")
        
        # Title Label
        title_label = ttk.Label(root, text="Music Generator", font=("Helvetica", 18, "bold"))
        title_label.pack(pady=20)

        # Description Label
        desc_label = ttk.Label(root, text="Generate Music with Neural Networks\nLoad MIDI Files, Train the Model, Generate, and Save Music.", font=("Helvetica", 12))
        desc_label.pack(pady=10)

        # Create a frame for the buttons
        button_frame = ttk.Frame(root)
        button_frame.pack(pady=20)

        # Buttons with brief descriptions
        self.load_btn = ttk.Button(button_frame, text="Load MIDI Files", command=self.load_files_thread)
        self.load_btn.grid(row=0, column=0, padx=10, pady=10)
        self.create_tooltip(self.load_btn, "Load MIDI files to train the model.")

        self.train_btn = ttk.Button(button_frame, text="Train Model", command=self.train_model_thread)
        self.train_btn.grid(row=0, column=1, padx=10, pady=10)
        self.create_tooltip(self.train_btn, "Train the model on loaded MIDI files.")

        self.load_model_btn = ttk.Button(button_frame, text="Load Model", command=self.load_model_file)
        self.load_model_btn.grid(row=1, column=0, padx=10, pady=10)
        self.create_tooltip(self.load_model_btn, "Load a pre-trained model to generate music.")

        self.generate_btn = ttk.Button(button_frame, text="Generate Music", command=self.generate_music_thread)
        self.generate_btn.grid(row=1, column=1, padx=10, pady=10)
        self.create_tooltip(self.generate_btn, "Generate new music using the trained model.")

        self.save_btn = ttk.Button(button_frame, text="Save Music", command=self.save_music)
        self.save_btn.grid(row=2, column=0, padx=10, pady=10)
        self.create_tooltip(self.save_btn, "Save the generated music as a MIDI file.")

        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate", variable=self.progress_var)
        self.progress_bar.pack(pady=10)

        # Variables
        self.model = None
        self.notes_list = []
        self.new_notes = []
        self.ind2note = {}
        self.note2ind = {}
        self.music_pattern = []
        self.output_notes = []

    # Function to add tooltips for buttons
    def create_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        tooltip_label = tk.Label(tooltip, text=text, font=("Helvetica", 10), background="white", borderwidth=1)
        tooltip_label.pack()

        def on_enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            tooltip.geometry(f"+{x}+{y}")
            tooltip.deiconify()

        def on_leave(event):
            tooltip.withdraw()

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    # Thread function for loading files
    def load_files_thread(self):
        threading.Thread(target=self.load_files).start()

    # Thread function for training model
    def train_model_thread(self):
        threading.Thread(target=self.train_model).start()

    # Thread function for generating music
    def generate_music_thread(self):
        threading.Thread(target=self.generate_music).start()

    # Load Model from a file
    def load_model_file(self):
        model_path = filedialog.askopenfilename(title="Select Trained Model", filetypes=[("Keras Model Files", "*.keras"), ("All Files", "*.*")])
        if model_path:
            try:
                self.model = load_model(model_path)
                messagebox.showinfo("Success", "Model Loaded Successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    # Remaining methods for loading MIDI files, training, generating music, and saving MIDI...
    # (Add the rest of your methods here, such as load_files, train_model, generate_music, and save_music)


    # Load MIDI files function with tqdm and GUI progress bar
    def load_files(self):
        folder_path = filedialog.askdirectory(title="Select Folder with MIDI Files")
        if not folder_path:
            messagebox.showerror("Error", "No folder selected")
            return

        all_files = glob.glob(f'{folder_path}/*.mid', recursive=True)
        total_files = len(all_files)

        if total_files == 0:
            messagebox.showerror("Error", "No MIDI files found in the selected folder")
            return

        # Initialize progress bars
        self.progress_var.set(0)
        self.progress_bar['maximum'] = total_files
        self.root.update()

        self.notes_list = []
        for i, file in enumerate(tqdm(all_files, position=0, leave=True)):  # Terminal-based progress
            try:
                self.notes_list.append(self.read_files(file))
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

            # Update the GUI progress bar
            self.progress_var.set(i + 1)
            self.root.update()

        self.process_notes()

        # Reset progress bar after loading is done
        self.progress_var.set(0)
        self.root.update()

        messagebox.showinfo("Success", "MIDI Files Loaded and Processed")

    # Read MIDI files and process notes
    def read_files(self, file):
        notes = []
        notes_to_parse = None
        midi = converter.parse(file)
        instrmt = instrument.partitionByInstrument(midi)

        for part in instrmt.parts:
            if 'Piano' in str(part):  # Only process piano parts
                notes_to_parse = part.recurse()

                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
            else:
                print(f"Skipping non-piano track in {file}.")
        return notes

    # Process the notes to prepare for training
    def process_notes(self):
        notess = sum(self.notes_list, [])
        unique_notes = list(set(notess))

        freq = dict(map(lambda x: (x, notess.count(x)), unique_notes))
        freq_notes = dict(filter(lambda x: x[1] >= 50, freq.items()))

        self.new_notes = [[i for i in j if i in freq_notes] for j in self.notes_list]

        self.ind2note = dict(enumerate(freq_notes))
        self.note2ind = dict(map(reversed, self.ind2note.items()))

    # Train the neural network model
    def train_model(self):
        timesteps = 50
        x = []
        y = []

        for i in self.new_notes:
            for j in range(0, len(i) - timesteps):
                inp = i[j:j + timesteps]
                out = i[j + timesteps]
                x.append(list(map(lambda x: self.note2ind[x], inp)))
                y.append(self.note2ind[out])

        x_new = np.array(x)
        y_new = np.array(y)

        x_new = np.reshape(x_new, (len(x_new), timesteps, 1))
        y_new = np.reshape(y_new, (-1, 1))

        x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(LSTM(256, return_sequences=True, input_shape=(x_new.shape[1], x_new.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(256))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(len(self.note2ind), activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=128, epochs=80, validation_data=(x_test, y_test))

        model.save("s2s.keras")
        self.model = model


        # Reset progress bar after training is done
        self.progress_var.set(0)
        self.root.update()

        messagebox.showinfo("Success", "Model Trained and Saved")

        # Generate new music based on the trained or loaded model
    def generate_music(self):
        if not self.model:
            messagebox.showerror("Error", "No model is loaded or trained")
            return

        if not self.new_notes:
            messagebox.showerror("Error", "No training data loaded. Load MIDI files or train the model first.")
            return

        # Randomly select a starting pattern from the training data (as indices, not notes)
        index = np.random.randint(0, len(self.new_notes) - 1)
        initial_pattern = self.new_notes[index][:50]
        self.music_pattern = list(map(lambda x: self.note2ind[x], initial_pattern))
        out_pred = []

        # Generate notes
        for note_index in range(200):  # Generate 200 notes
            music_pattern_input = np.reshape(self.music_pattern, (1, len(self.music_pattern), 1))

            # Predict the next note (as an index)
            predicted_index = np.argmax(self.model.predict(music_pattern_input, verbose=0))

            # Append the predicted note (as an index) to the output pattern
            out_pred.append(self.ind2note[predicted_index])

            # Update music_pattern: add the predicted note and remove the first one
            self.music_pattern.append(predicted_index)
            self.music_pattern = self.music_pattern[1:]

        # Convert the output prediction indices back to notes or chords
        self.output_notes = []
        for offset, pattern in enumerate(out_pred):
            if ('.' in pattern) or pattern.isdigit():  # Chord
                notes_in_chord = pattern.split('.')
                notes = [note.Note(int(current_note)) for current_note in notes_in_chord]
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                self.output_notes.append(new_chord)
            else:  # Single note
                new_note = note.Note(pattern)
                new_note.offset = offset
                self.output_notes.append(new_note)

        # Reset progress bar after generation is done
        self.progress_var.set(0)
        self.root.update()

        messagebox.showinfo("Success", "Music Generated")


    # Save the generated music as a MIDI file
    def save_music(self):
        if not self.output_notes:
            messagebox.showerror("Error", "No music generated")
            return

        midi_stream = stream.Stream(self.output_notes)
        save_path = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI files", "*.mid")])

        if save_path:
            midi_stream.write("midi", fp=save_path)
            messagebox.showinfo("Success", f"Music saved as {save_path}")
        else:
            messagebox.showerror("Error", "No file path provided")


# Main loop to run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = MusicGeneratorApp(root)
    root.mainloop()
