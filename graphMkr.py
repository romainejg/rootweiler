import os
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, Frame, Label, Button, filedialog, ttk, StringVar, messagebox
from tkinter.font import Font
import logging

# Set up logging to save in the same directory as the script
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'graphMkr.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GraphMaker:
    def __init__(self, root, back_callback):
        self.root = root
        self.back_callback = back_callback
        self.data = None
        self.create_interface()

    def create_interface(self):
        try:
            self.clear_root_widgets()

            self.header_font = Font(family="Helvetica", size=28, weight="bold")
            self.subtext_font = Font(family="Helvetica", size=16, slant="italic")
            self.button_font = Font(family="Helvetica", size=16, weight="bold")

            frame = ttk.Frame(self.root, padding="20")
            frame.pack(fill="both", expand=True)

            header_label = ttk.Label(frame, text="Graph Maker", font=self.header_font)
            header_label.pack(pady=10)

            subtext_label = ttk.Label(frame, text="Upload an Excel file to create graphs", font=self.subtext_font)
            subtext_label.pack(pady=10)

            style = ttk.Style()
            style.configure('TButton', font=self.button_font, padding=10)

            upload_button = ttk.Button(frame, text="Upload Excel File", command=self.upload_file, style='TButton')
            upload_button.pack(pady=10, fill="x")

            self.filename_label = ttk.Label(frame, text="", font=self.subtext_font)
            self.filename_label.pack(pady=10)

            self.column_selection_frame = Frame(frame)
            self.column_selection_frame.pack(pady=10, fill="x")

            graph_button = ttk.Button(frame, text="Create Graphs", command=self.create_graphs, style='TButton')
            graph_button.pack(pady=10, fill="x")

            back_button = ttk.Button(frame, text="Back", command=self.back_callback, style='TButton')
            back_button.pack(pady=10, fill="x")

            logging.debug("Interface created successfully.")
        except Exception as e:
            logging.error(f"Error in create_interface: {e}")
            messagebox.showerror("Interface Error", f"An error occurred while creating the interface: {e}")

    def upload_file(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
            if file_path:
                logging.debug(f"File selected: {file_path}")
                self.data = pd.read_excel(file_path)
                self.filename_label.config(text=os.path.basename(file_path))
                self.show_column_selection()
        except Exception as e:
            logging.error(f"Error uploading file: {e}")
            messagebox.showerror("File Upload Error", f"An error occurred while uploading the file: {e}")

    def show_column_selection(self):
        try:
            for widget in self.column_selection_frame.winfo_children():
                widget.destroy()

            self.x_axis_var = StringVar()
            self.columns_var = {col: StringVar() for col in self.data.columns}

            Label(self.column_selection_frame, text="Select X Axis:").grid(row=0, column=0, sticky='w')
            self.x_axis_menu = ttk.OptionMenu(self.column_selection_frame, self.x_axis_var, self.data.columns[0], *self.data.columns)
            self.x_axis_menu.grid(row=0, column=1, sticky='w')

            Label(self.column_selection_frame, text="Select Columns for Graph:").grid(row=1, column=0, sticky='w')
            for i, col in enumerate(self.data.columns):
                chk = ttk.Checkbutton(self.column_selection_frame, text=col, variable=self.columns_var[col])
                chk.grid(row=1 + (i // 2), column=1 + (i % 2), sticky='w')

            logging.debug("Column selection interface created successfully.")
        except Exception as e:
            logging.error(f"Error showing column selection: {e}")
            messagebox.showerror("Column Selection Error", f"An error occurred while showing column selection: {e}")

    def create_graphs(self):
        try:
            selected_columns = [col for col, var in self.columns_var.items() if var.get()]
            x_axis = self.x_axis_var.get()

            if not selected_columns or not x_axis:
                messagebox.showwarning("Selection Error", "Please select columns and an X axis for the graph.")
                return

            fig, axs = plt.subplots(len(selected_columns), 2, figsize=(15, 5 * len(selected_columns)))
            if len(selected_columns) == 1:
                axs = [axs]

            for i, col in enumerate(selected_columns):
                # Bar Graph
                axs[i][0].bar(self.data[x_axis], self.data[col])
                axs[i][0].set_title(f'Bar Graph of {col} vs {x_axis}')
                axs[i][0].set_xlabel(x_axis)
                axs[i][0].set_ylabel(col)

                # Scatter Plot
                axs[i][1].scatter(self.data[x_axis], self.data[col])
                axs[i][1].set_title(f'Scatter Plot of {col} vs {x_axis}')
                axs[i][1].set_xlabel(x_axis)
                axs[i][1].set_ylabel(col)

            plt.tight_layout()
            plt.show()

            self.save_graphs(fig)
        except Exception as e:
            logging.error(f"Error creating graphs: {e}")
            messagebox.showerror("Graph Creation Error", f"An error occurred while creating the graphs: {e}")

    def save_graphs(self, fig):
        try:
            save_option = messagebox.askquestion("Save Option", "Do you want to save the graphs as a combined PDF?")
            if save_option == "yes":
                pdf_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
                if pdf_path:
                    self.save_as_pdf(fig, pdf_path)
            else:
                img_path = filedialog.askdirectory()
                if img_path:
                    self.save_as_images(fig, img_path)
        except Exception as e:
            logging.error(f"Error saving graphs: {e}")
            messagebox.showerror("Save Error", f"An error occurred while saving the graphs: {e}")

    def save_as_pdf(self, fig, path):
        try:
            fig.savefig(path, format="pdf")
            logging.debug(f"Graphs saved as PDF: {path}")
        except Exception as e:
            logging.error(f"Error saving PDF: {e}")
            messagebox.showerror("Save Error", f"An error occurred while saving the PDF: {e}")

    def save_as_images(self, fig, folder_path):
        try:
            for i in range(len(fig.axes) // 2):
                fig.axes[i * 2].figure.savefig(os.path.join(folder_path, f"bar_graph_{i}.png"))
                fig.axes[i * 2 + 1].figure.savefig(os.path.join(folder_path, f"scatter_plot_{i}.png"))
            logging.debug(f"Graphs saved as images in: {folder_path}")
        except Exception as e:
            logging.error(f"Error saving images: {e}")
            messagebox.showerror("Save Error", f"An error occurred while saving the images: {e}")

    def clear_root_widgets(self):
        try:
            for widget in self.root.winfo_children():
                widget.destroy()
            logging.debug("Cleared root widgets.")
        except Exception as e:
            logging.error(f"Error clearing root widgets: {e}")
            messagebox.showerror("Widget Clear Error", f"An error occurred while clearing the widgets: {e}")

def main_interface(root, back_callback):
    try:
        app = GraphMaker(root, back_callback)
        logging.debug("GraphMaker instance created.")
    except Exception as e:
        logging.error(f"Error in main_interface: {e}")
        messagebox.showerror("Initialization Error", f"An error occurred during initialization: {e}")

if __name__ == "__main__":
    try:
        root = Tk()
        main_interface(root, lambda: root.destroy())
        root.mainloop()
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
        messagebox.showerror("Main Loop Error", f"An error occurred in the main loop: {e}")
