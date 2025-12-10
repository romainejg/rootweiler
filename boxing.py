import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.font import Font
from tkinter import PhotoImage
from PIL import Image, ImageTk
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BoxingApp:
    def __init__(self, root, back_callback):
        self.root = root
        self.back_callback = back_callback
        self.create_initial_interface()

    def clear_root_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def create_initial_interface(self):
        self.clear_root_widgets()

        self.header_font = Font(family="Rubik", size=28, weight="bold")
        self.subtext_font = Font(family="Rubik", size=16, slant="italic")
        self.button_font = Font(family="Rubik", size=16, weight="bold")

        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill="both", expand=True)

        header_label = ttk.Label(frame, text="Boxing Script", font=self.header_font)
        header_label.pack(pady=10)

        subtext_label = ttk.Label(frame, text="Select an option below to start", font=self.subtext_font)
        subtext_label.pack(pady=10)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(script_dir, "logo.png")

        try:
            logo_image = Image.open(logo_path)
            logo_image = logo_image.resize((300, 300), Image.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = ttk.Label(frame, image=logo_photo)
            logo_label.image = logo_photo
            logo_label.pack(pady=20)
        except Exception as e:
            print(f"Error loading logo image: {e}")

        style = ttk.Style()
        style.configure('TButton', font=self.button_font, padding=10)

        run_button = ttk.Button(frame, text="Run boxing.py", command=self.run_boxing, style='TButton')
        run_button.pack(pady=10, fill="x")

        back_button = ttk.Button(frame, text="Back", command=self.back_to_main, style='TButton')
        back_button.pack(pady=10, fill="x")

    def run_boxing(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if file_path:
            self.create_column_selection_interface(file_path)

    def create_column_selection_interface(self, file_path):
        self.clear_root_widgets()

        self.header_font = Font(family="Rubik", size=24, weight="bold")
        self.subtext_font = Font(family="Rubik", size=16)

        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill="both", expand=True)

        header_label = ttk.Label(frame, text="Select Columns for Box Plot", font=self.header_font)
        header_label.pack(pady=10)

        self.data = pd.read_excel(file_path)
        columns = self.data.columns.tolist()

        form_frame = ttk.Frame(frame)
        form_frame.pack(pady=10)

        x_label = ttk.Label(form_frame, text="X-axis column:", font=self.subtext_font)
        x_label.grid(row=0, column=0, pady=5, sticky="e")
        self.x_combobox = ttk.Combobox(form_frame, values=columns)
        self.x_combobox.grid(row=0, column=1, pady=5)

        y_label = ttk.Label(form_frame, text="Y-axis column:", font=self.subtext_font)
        y_label.grid(row=1, column=0, pady=5, sticky="e")
        self.y_combobox = ttk.Combobox(form_frame, values=columns)
        self.y_combobox.grid(row=1, column=1, pady=5)

        title_label = ttk.Label(form_frame, text="Plot Title:", font=self.subtext_font)
        title_label.grid(row=2, column=0, pady=5, sticky="e")
        self.title_entry = ttk.Entry(form_frame)
        self.title_entry.grid(row=2, column=1, pady=5)

        xlabel_label = ttk.Label(form_frame, text="X-axis Label:", font=self.subtext_font)
        xlabel_label.grid(row=3, column=0, pady=5, sticky="e")
        self.xlabel_entry = ttk.Entry(form_frame)
        self.xlabel_entry.grid(row=3, column=1, pady=5)

        ylabel_label = ttk.Label(form_frame, text="Y-axis Label:", font=self.subtext_font)
        ylabel_label.grid(row=4, column=0, pady=5, sticky="e")
        self.ylabel_entry = ttk.Entry(form_frame)
        self.ylabel_entry.grid(row=4, column=1, pady=5)

        self.x_combobox.bind("<<ComboboxSelected>>", self.auto_fill_labels)
        self.y_combobox.bind("<<ComboboxSelected>>", self.auto_fill_labels)

        generate_button = ttk.Button(frame, text="Generate Box Plot", command=self.generate_box_plot)
        generate_button.pack(pady=20)

        back_button = ttk.Button(frame, text="Back", command=self.create_initial_interface)
        back_button.pack(pady=10)

        self.file_path = file_path

    def auto_fill_labels(self, event):
        x_column = self.x_combobox.get()
        y_column = self.y_combobox.get()
        if x_column and y_column:
            self.title_entry.delete(0, tk.END)
            self.title_entry.insert(0, f"Box Plot of {y_column} by {x_column}")
            self.xlabel_entry.delete(0, tk.END)
            self.xlabel_entry.insert(0, x_column)
            self.ylabel_entry.delete(0, tk.END)
            self.ylabel_entry.insert(0, y_column)

    def generate_box_plot(self):
        x_column = self.x_combobox.get()
        y_column = self.y_combobox.get()
        plot_title = self.title_entry.get()
        x_label = self.xlabel_entry.get()
        y_label = self.ylabel_entry.get()

        if not all([x_column, y_column, plot_title, x_label, y_label]):
            messagebox.showerror("Error", "All fields must be filled out")
            return

        sns.set(style="whitegrid")

        plt.figure(figsize=(12, 8))
        order = self.data.groupby(x_column)[y_column].mean().sort_values(ascending=False).index
        box_palette = ["#64D273", "#9B91F9", "#ED695D", "#FFD750"]
        box_colors = {key: box_palette[i % len(box_palette)] for i, key in enumerate(order)}

        box_plot = sns.boxplot(
            x=x_column, y=y_column, data=self.data, order=order,
            showmeans=True, meanprops={"marker": "*", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 15},
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 2},
            whiskerprops={"color": "black"},
            capprops={"color": "black"},
            medianprops={"color": "black"}
        )

        for i, box in enumerate(box_plot.artists):
            box.set_edgecolor(box_colors[order[i]])
            box.set_linewidth(2)

        strip_plot = sns.stripplot(
            x=x_column, y=y_column, data=self.data, order=order, jitter=True,
            hue=x_column, palette=box_colors, edgecolor="none", linewidth=0, dodge=False, legend=False
        )

        # Overlay the mean markers on top of the stripplot
        means = self.data.groupby(x_column)[y_column].mean().reindex(order)
        for i, mean in enumerate(means):
            plt.plot(i, mean, marker="*", color="black", markersize=15, zorder=10)

        plt.title(plot_title, fontsize=16, fontweight='bold', fontname='Urbane Rounded')
        plt.xlabel(x_label, fontsize=14, fontname='Rubik')
        plt.ylabel(y_label, fontsize=14, fontname='Rubik')
        plt.xticks(rotation=0)  # Rotate x-axis labels to be horizontal
        plt.subplots_adjust(bottom=0.2)  # Adjust the bottom to ensure labels are not cut off
        plt.gca().patch.set_facecolor('#E3E3E2')

        # Set grid lines to white
        plt.grid(color='white', which='both', linestyle='-', linewidth=1)
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(plt.gca().get_yticks()[1] / 2))
        plt.gca().xaxis.grid(True, linestyle='-', linewidth=1, color='white')

        output_path = os.path.join(os.path.dirname(self.file_path), "box_plot.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        messagebox.showinfo("Success", f"Box plot saved as {output_path}")

    def back_to_main(self):
        self.clear_root_widgets()
        self.back_callback()

def main_interface(root, back_callback):
    BoxingApp(root, back_callback)

if __name__ == "__main__":
    root = tk.Tk()
    app = BoxingApp(root, lambda: print("Back button pressed"))
    root.mainloop()
