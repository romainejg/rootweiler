import os
import cv2
import numpy as np
from tkinter import Tk, ttk, filedialog, messagebox
from tkinter.font import Font
from PIL import Image, ImageTk
from openpyxl import Workbook
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LeafAnalysisApp:
    def __init__(self, root, back_callback):
        self.root = root
        self.back_callback = back_callback
        self.images = []
        self.filenames = []
        self.current_image_index = 0
        self.selected_objects = []
        self.masks = []
        self.current_mask_index = 0
        self.folder_path = None
        self.measurements = []
        self.fig_canvas = None  # Initialize fig_canvas to None
        self.show_options()

    def show_options(self):
        self.clear_root_widgets()
        self.header_font = Font(family="Helvetica", size=28, weight="bold")
        self.subtext_font = Font(family="Helvetica", size=16, slant="italic")
        self.button_font = Font(family="Helvetica", size=16, weight="bold")

        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill="both", expand=True)

        header_label = ttk.Label(frame, text="Leaf Analysis", font=self.header_font)
        header_label.pack(pady=10)

        subtext_label = ttk.Label(frame, text="Select an option below", font=self.subtext_font)
        subtext_label.pack(pady=10)

        # Load and display logo image (e.g., a dog photo)
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
        logo_path = os.path.join(script_dir, "logo.png")  # Construct the full path to the logo image

        try:
            print(f"Loading logo image from {logo_path}")
            logo_image = Image.open(logo_path)  # Load the logo image from the script directory
            logo_image = logo_image.resize((300, 300), Image.LANCZOS)
            logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = ttk.Label(frame, image=logo_photo)
            logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
            logo_label.pack(pady=20)
        except Exception as e:
            print(f"Error loading logo image: {e}")

        style = ttk.Style()
        style.configure('TButton', font=self.button_font, padding=10)

        conduct_button = ttk.Button(frame, text="Conduct Analysis", command=self.select_folder_for_analysis, style='TButton')
        conduct_button.pack(pady=10, fill="x")

        view_button = ttk.Button(frame, text="View Analysis", command=self.select_folder_for_viewing, style='TButton')
        view_button.pack(pady=10, fill="x")

        back_button = ttk.Button(frame, text="Back", command=self.back_callback, style='TButton')
        back_button.pack(pady=10, fill="x")

    def select_folder_for_analysis(self):
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
        if folder_path:
            self.folder_path = folder_path
            self.process_images(folder_path)

    def select_folder_for_viewing(self):
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
        if folder_path:
            self.folder_path = folder_path
            self.view_analysis_images(folder_path)

    def create_mask(self, image, mask_type=0):
        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for green color and create a binary mask
        lower_green = np.array([0, 40, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Remove noise using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Compute the exact Euclidean distance from every binary pixel to the nearest zero pixel
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Threshold to obtain the sure foreground
        ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Sure background area
        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        
        # Unknown region (where we are not sure whether it is foreground or background)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that the sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed algorithm
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]
        
        # Create the mask based on watershed result
        mask = np.zeros_like(image[:, :, 0])
        mask[markers > 1] = 255
        
        return mask

    def measure_objects(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        measurements = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 50000:  # Only count objects larger than so many pixels
                measurements.append((x, y, w, h))
        return contours, measurements

    def calculate_pixels_per_cm(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        squares = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if abs(w - h) < 10 and w * h > 1000:
                squares.append((x, y, w, h))
        if len(squares) < 20:
            return None
        areas = [w * h for _, _, w, h in squares]
        median_area = np.median(areas)
        squares = sorted(squares, key=lambda s: abs((s[2] * s[3]) - median_area))
        squares = squares[:20]
        average_area = np.mean([w * h for _, _, w, h in squares])
        pixels_per_cm2 = average_area
        return pixels_per_cm2

    def process_images(self, folder_path):
        workbook_path = os.path.join(folder_path, "measurements.xlsx")
        self.workbook = Workbook()
        self.sheet = self.workbook.active
        self.sheet.append(["Filename", "Object", "Width (cm)", "Height (cm)"])

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                pixels_per_cm2 = self.calculate_pixels_per_cm(image)
                if pixels_per_cm2 is None:
                    print(f"Could not determine pixels per cm for {filename}")
                    continue
                mask = self.create_mask(image)
                contours, measurements = self.measure_objects(mask)
                x_locations = [x + w // 2 for x, y, w, h in measurements]
                median_x = np.median(x_locations)
                valid_measurements = []
                for x, y, w, h in measurements:
                    center_x = x + w // 2
                    close_count = sum(1 for xx in x_locations if abs(center_x - xx) < 10 * np.sqrt(pixels_per_cm2))
                    if close_count >= 3:
                        valid_measurements.append((x, y, w, h))
                for i, (x, y, w, h) in enumerate(valid_measurements):
                    width_cm = round(w / np.sqrt(pixels_per_cm2), 1)
                    height_cm = round(h / np.sqrt(pixels_per_cm2), 1)
                    self.sheet.append([filename, i + 1, width_cm, height_cm])

        self.workbook.save(workbook_path)
        messagebox.showinfo("Success", f"Analysis complete. Measurements saved to {workbook_path}")

    def view_analysis_images(self, folder_path):
        self.clear_root_widgets()
        self.images = []
        self.filenames = []
        self.current_image_index = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    self.images.append(image)
                    self.filenames.append(filename)
        if not self.images:
            messagebox.showinfo("Info", "No images found in the selected folder.")
            self.show_options()
            return
        self.show_image_for_selection()

    def show_image_for_selection(self):
        if self.current_image_index >= len(self.images):
            messagebox.showinfo("Info", "All images reviewed.")
            self.show_options()
            return

        self.clear_root_widgets()

        # Display the image title at the top
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.pack(fill="x")
        image_title = ttk.Label(title_frame, text=self.filenames[self.current_image_index], font=("Helvetica", 16, "bold"))
        image_title.pack(pady=5)

        image = self.images[self.current_image_index]
        mask = self.create_mask(image, self.current_mask_index)
        contours, self.measurements = self.measure_objects(mask)
        self.selected_objects = list(range(len(self.measurements)))

        self.image_with_boxes = image.copy()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # Initialize fig_canvas here
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.update_plot()

        self.fig_canvas.mpl_connect('button_press_event', self.on_click)

        self.plot_button_controls()

    def on_click(self, event):
        if event.inaxes == self.axes[0]:
            for idx, (x, y, w, h) in enumerate(self.measurements):
                if x <= event.xdata <= x + w and y <= event.ydata <= y + h:
                    if idx in self.selected_objects:
                        self.selected_objects.remove(idx)
                    else:
                        self.selected_objects.append(idx)
                    self.update_plot()
                    break

    def update_plot(self):
        self.axes[0].clear()
        self.axes[1].clear()
        self.axes[0].imshow(cv2.cvtColor(self.image_with_boxes, cv2.COLOR_BGR2RGB))
        self.axes[1].imshow(self.create_mask(self.image_with_boxes, self.current_mask_index), cmap='gray')

        for i, (x, y, w, h) in enumerate(self.measurements):
            color = 'green' if i in self.selected_objects else 'red'
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            self.axes[0].add_patch(rect)
            self.axes[0].text(x, y, str(i + 1), fontsize=12, color=color, weight='bold', backgroundcolor='none')

        self.axes[0].set_title('Original Image with Objects')
        self.axes[0].axis('off')
        self.axes[1].set_title('Mask Image')
        self.axes[1].axis('off')
        self.fig_canvas.draw()

    def plot_button_controls(self):
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        save_button = ttk.Button(button_frame, text="Objects Selected", command=self.save_selected_objects, style='TButton')
        save_button.pack(pady=10, side="left")

        back_button = ttk.Button(button_frame, text="Back", command=self.back_callback, style='TButton')
        back_button.pack(pady=10, side="left")

        green_leaf_button = ttk.Button(button_frame, text="Green Leaf", command=lambda: self.apply_mask(0), style='TButton')
        green_leaf_button.pack(pady=10, side="left")

        red_leaf_button = ttk.Button(button_frame, text="Red Leaf", command=lambda: self.apply_mask(1), style='TButton')
        red_leaf_button.pack(pady=10, side="left")

        red_color_penetration_button = ttk.Button(button_frame, text="Red Color Penetration", command=lambda: self.apply_mask(2), style='TButton')
        red_color_penetration_button.pack(pady=10, side="left")

    def apply_mask(self, mask_index):
        self.current_mask_index = mask_index
        self.show_image_for_selection()

    def save_selected_objects(self):
        if not hasattr(self, 'workbook'):
            self.workbook = Workbook()
            self.sheet = self.workbook.active
            self.sheet.append(["Filename", "Object", "Width (cm)", "Height (cm)"])

        pixels_per_cm2 = self.calculate_pixels_per_cm(self.images[self.current_image_index])
        if pixels_per_cm2 is None:
            print(f"Could not determine pixels per cm for {self.images[self.current_image_index]}")
            return
        
        filename = self.filenames[self.current_image_index]
        for idx in self.selected_objects:
            x, y, w, h = self.measurements[idx]
            width_cm = round(w / np.sqrt(pixels_per_cm2), 1)
            height_cm = round(h / np.sqrt(pixels_per_cm2), 1)
            self.sheet.append([filename, idx + 1, width_cm, height_cm])

        self.workbook.save(os.path.join(self.folder_path, "measurements.xlsx"))
        self.show_next_image()

    def show_next_image(self):
        self.current_image_index += 1
        self.show_image_for_selection()

    def clear_root_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()

def main_interface(root, back_callback):
    LeafAnalysisApp(root, back_callback)

if __name__ == "__main__":
    root = Tk()
    app = LeafAnalysisApp(root, lambda: root.destroy())
    root.mainloop()