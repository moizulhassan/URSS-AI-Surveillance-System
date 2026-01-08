import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class Dashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Urban Risk Surveillance Dashboard")
        self.root.geometry("500x400")

        self.person_label = tk.Label(self.root, text="Persons: 0", font=("Arial", 16))
        self.person_label.pack(pady=10)

        self.fire_label = tk.Label(self.root, text="Fire Detected: NO", font=("Arial", 16), fg="green")
        self.fire_label.pack(pady=10)

        self.risk_label = tk.Label(self.root, text="Risk Level: LOW", font=("Arial", 16))
        self.risk_label.pack(pady=10)

        # Risk chart
        self.fig, self.ax = plt.subplots(figsize=(4,2))
        self.ax.set_title("Risk Score Over Time")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Score")
        self.line, = self.ax.plot([], [], color='r')
        self.x_data = []
        self.y_data = []

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

    def update(self, person_count, fire_detected, risk, frame_number):
        self.person_label.config(text=f"Persons: {person_count}")
        if fire_detected:
            self.fire_label.config(text="Fire Detected: YES", fg="red")
        else:
            self.fire_label.config(text="Fire Detected: NO", fg="green")
        self.risk_label.config(text=f"Risk Level: {risk}")

        self.x_data.append(frame_number)
        self.y_data.append(person_count)
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def run(self):
        self.root.mainloop()

