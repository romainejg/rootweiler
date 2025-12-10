import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, filedialog, Frame, Label, Entry, Button, ttk
from tkinter.font import Font
from PIL import Image, ImageTk

class ImageSegmentationDebugger:
    def __init__(self, root, back_callback):
        self.root = root
        self.back_callback = back_callback
        self.images = []
        self.current_index = 0

        self.lower_hue = 0
        self.lower_saturation = 40
        self.lower_value = 50
        self.upper_hue = 80
        self.upper_saturation = 255
        self.upper_value = 255
        self.morph_iterations = 2
        self.kernel_size = 3
        self.dilate_iterations = 3
        self.dist_transform_threshold = 0.7

        self.create_initial_interface()

    def create_initial_interface(self):
        self.clear_root_widgets()

        self.header_font = Font(family="Helvetica", size=28, weight="bold")
        self.subtext_font = Font(family="Helvetica", size=16, slant="italic")
        self.button_font = Font(family="Helvetica", size=16, weight="bold")

        frame = ttk.Frame(self.root, padding="20")
        frame.pack(fill="both", expand=True)

        header_label = ttk.Label(frame, text="Debugger", font=self.header_font)
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

        start_button = ttk.Button(frame, text="Leaf Segmentation", command=self.load_images, style='TButton')
        start_button.pack(pady=10, fill="x")

        back_button = ttk.Button(frame, text="Back", command=self.handle_back, style='TButton')
        back_button.pack(pady=10, fill="x")

    def handle_back(self):
        print("Back button pressed, calling back_callback")
        self.back_callback()

    def load_images(self):
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
        if not folder_path:
            print("No folder selected. Exiting.")
            self.back_callback()
            return
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    self.images.append((filename, image))
        
        if not self.images:
            print("No images found in the selected folder. Exiting.")
            self.back_callback()
            return

        self.create_debugging_interface()

    def create_debugging_interface(self):
        self.clear_root_widgets()

        self.controls_frame = Frame(self.root)
        self.controls_frame.pack(side='left', fill='y', padx=10, pady=10)

        Label(self.controls_frame, text="Lower Hue:").grid(row=0, column=0, sticky='w')
        self.entry_lower_hue = Entry(self.controls_frame)
        self.entry_lower_hue.grid(row=0, column=1)
        self.entry_lower_hue.insert(0, str(self.lower_hue))

        Label(self.controls_frame, text="Lower Saturation:").grid(row=1, column=0, sticky='w')
        self.entry_lower_saturation = Entry(self.controls_frame)
        self.entry_lower_saturation.grid(row=1, column=1)
        self.entry_lower_saturation.insert(0, str(self.lower_saturation))

        Label(self.controls_frame, text="Lower Value:").grid(row=2, column=0, sticky='w')
        self.entry_lower_value = Entry(self.controls_frame)
        self.entry_lower_value.grid(row=2, column=1)
        self.entry_lower_value.insert(0, str(self.lower_value))

        Label(self.controls_frame, text="Upper Hue:").grid(row=3, column=0, sticky='w')
        self.entry_upper_hue = Entry(self.controls_frame)
        self.entry_upper_hue.grid(row=3, column=1)
        self.entry_upper_hue.insert(0, str(self.upper_hue))

        Label(self.controls_frame, text="Upper Saturation:").grid(row=4, column=0, sticky='w')
        self.entry_upper_saturation = Entry(self.controls_frame)
        self.entry_upper_saturation.grid(row=4, column=1)
        self.entry_upper_saturation.insert(0, str(self.upper_saturation))

        Label(self.controls_frame, text="Upper Value:").grid(row=5, column=0, sticky='w')
        self.entry_upper_value = Entry(self.controls_frame)
        self.entry_upper_value.grid(row=5, column=1)
        self.entry_upper_value.insert(0, str(self.upper_value))

        Label(self.controls_frame, text="Morph Iterations:").grid(row=6, column=0, sticky='w')
        self.entry_morph_iterations = Entry(self.controls_frame)
        self.entry_morph_iterations.grid(row=6, column=1)
        self.entry_morph_iterations.insert(0, str(self.morph_iterations))

        Label(self.controls_frame, text="Kernel Size:").grid(row=7, column=0, sticky='w')
        self.entry_kernel_size = Entry(self.controls_frame)
        self.entry_kernel_size.grid(row=7, column=1)
        self.entry_kernel_size.insert(0, str(self.kernel_size))

        Label(self.controls_frame, text="Dilate Iterations:").grid(row=8, column=0, sticky='w')
        self.entry_dilate_iterations = Entry(self.controls_frame)
        self.entry_dilate_iterations.grid(row=8, column=1)
        self.entry_dilate_iterations.insert(0, str(self.dilate_iterations))

        Label(self.controls_frame, text="Dist Transform Threshold:").grid(row=9, column=0, sticky='w')
        self.entry_dist_thresh = Entry(self.controls_frame)
        self.entry_dist_thresh.grid(row=9, column=1)
        self.entry_dist_thresh.insert(0, str(self.dist_transform_threshold))

        style = ttk.Style()
        style.configure('TButton', font=self.button_font, padding=5)

        self.update_button = ttk.Button(self.controls_frame, text="Update", command=self.update_params, style='TButton')
        self.update_button.grid(row=10, column=0, columnspan=2, pady=10)

        self.prev_button = ttk.Button(self.controls_frame, text="Previous", command=self.prev_image, style='TButton')
        self.prev_button.grid(row=11, column=0, pady=10)

        self.next_button = ttk.Button(self.controls_frame, text="Next", command=self.next_image, style='TButton')
        self.next_button.grid(row=11, column=1, pady=10)

        self.reset_button = ttk.Button(self.controls_frame, text="Reset", command=self.reset_params, style='TButton')
        self.reset_button.grid(row=12, column=0, pady=10)

        self.save_button = ttk.Button(self.controls_frame, text="Save Config", command=self.save_config, style='TButton')
        self.save_button.grid(row=12, column=1, pady=10)

        self.fig, self.axs = plt.subplots(2, 4, figsize=(20, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side='right', fill='both', expand=True)

        self.show_image()

    def update_params(self):
        self.lower_hue = int(self.entry_lower_hue.get())
        self.lower_saturation = int(self.entry_lower_saturation.get())
        self.lower_value = int(self.entry_lower_value.get())
        self.upper_hue = int(self.entry_upper_hue.get())
        self.upper_saturation = int(self.entry_upper_saturation.get())
        self.upper_value = int(self.entry_upper_value.get())
        self.morph_iterations = int(self.entry_morph_iterations.get())
        self.kernel_size = int(self.entry_kernel_size.get())
        self.dilate_iterations = int(self.entry_dilate_iterations.get())
        self.dist_transform_threshold = float(self.entry_dist_thresh.get())
        self.show_image()

    def reset_params(self):
        self.lower_hue = 0
        self.lower_saturation = 40
        self.lower_value = 50
        self.upper_hue = 80
        self.upper_saturation = 255
        self.upper_value = 255
        self.morph_iterations = 2
        self.kernel_size = 3
        self.dilate_iterations = 3
        self.dist_transform_threshold = 0.7

        self.entry_lower_hue.delete(0, 'end')        
        self.entry_lower_hue.insert(0, str(self.lower_hue))

        self.entry_lower_saturation.delete(0, 'end')
        self.entry_lower_saturation.insert(0, str(self.lower_saturation))

        self.entry_lower_value.delete(0, 'end')
        self.entry_lower_value.insert(0, str(self.lower_value))

        self.entry_upper_hue.delete(0, 'end')
        self.entry_upper_hue.insert(0, str(self.upper_hue))

        self.entry_upper_saturation.delete(0, 'end')
        self.entry_upper_saturation.insert(0, str(self.upper_saturation))

        self.entry_upper_value.delete(0, 'end')
        self.entry_upper_value.insert(0, str(self.upper_value))

        self.entry_morph_iterations.delete(0, 'end')
        self.entry_morph_iterations.insert(0, str(self.morph_iterations))

        self.entry_kernel_size.delete(0, 'end')
        self.entry_kernel_size.insert(0, str(self.kernel_size))

        self.entry_dilate_iterations.delete(0, 'end')
        self.entry_dilate_iterations.insert(0, str(self.dilate_iterations))

        self.entry_dist_thresh.delete(0, 'end')
        self.entry_dist_thresh.insert(0, str(self.dist_transform_threshold))

        self.show_image()

    def save_config(self):
        config = {
            "lower_hue": self.lower_hue,
            "lower_saturation": self.lower_saturation,
            "lower_value": self.lower_value,
            "upper_hue": self.upper_hue,
            "upper_saturation": self.upper_saturation,
            "upper_value": self.upper_value,
            "morph_iterations": self.morph_iterations,
            "kernel_size": self.kernel_size,
            "dilate_iterations": self.dilate_iterations,
            "dist_transform_threshold": self.dist_transform_threshold
        }
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if save_path:
            with open(save_path, 'w') as config_file:
                import json
                json.dump(config, config_file)
            print(f"Configuration saved to {save_path}")

    def show_image(self):
        if self.current_index >= len(self.images):
            return

        filename, image = self.images[self.current_index]
        mask, mask_cleaned, dist_transform, sure_fg, sure_bg, unknown, markers = self.create_mask(image)

        self.axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.axs[0, 0].set_title(f'Original Image\n{filename}')
        self.axs[0, 0].axis('off')

        self.axs[0, 1].imshow(mask, cmap='gray')
        self.axs[0, 1].set_title('Initial Mask')
        self.axs[0, 1].axis('off')

        self.axs[0, 2].imshow(mask_cleaned, cmap='gray')
        self.axs[0, 2].set_title('Mask after Morphological Operations')
        self.axs[0, 2].axis('off')

        self.axs[0, 3].imshow(dist_transform, cmap='gray')
        self.axs[0, 3].set_title('Distance Transform')
        self.axs[0, 3].axis('off')

        self.axs[1, 0].imshow(sure_fg, cmap='gray')
        self.axs[1, 0].set_title('Sure Foreground')
        self.axs[1, 0].axis('off')

        self.axs[1, 1].imshow(sure_bg, cmap='gray')
        self.axs[1, 1].set_title('Sure Background')
        self.axs[1, 1].axis('off')

        self.axs[1, 2].imshow(unknown, cmap='gray')
        self.axs[1, 2].set_title('Unknown Region')
        self.axs[1, 2].axis('off')

        markers_copy = markers.copy()
        markers_copy[markers_copy == -1] = 0
        self.axs[1, 3].imshow(markers_copy, cmap='nipy_spectral')
        self.axs[1, 3].set_title('Markers before Watershed')
        self.axs[1, 3].axis('off')

        self.fig.suptitle('Debugger', fontsize=16)
        self.canvas.draw()

    def create_mask(self, image):
        # Convert image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for green color and create a binary mask
        lower_green = np.array([self.lower_hue, self.lower_saturation, self.lower_value])
        upper_green = np.array([self.upper_hue, self.upper_saturation, self.upper_value])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Remove noise using morphological operations
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
        
        # Compute the exact Euclidean distance from every binary pixel to the nearest zero pixel
        dist_transform = cv2.distanceTransform(mask_cleaned, cv2.DIST_L2, 5)
        
        # Threshold to obtain the sure foreground
        ret, sure_fg = cv2.threshold(dist_transform, self.dist_transform_threshold * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Sure background area
        sure_bg = cv2.dilate(mask_cleaned, kernel, iterations=self.dilate_iterations)
        
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
        image[markers == -1] = [255, 0, 0]  # Boundary marked in red

        return mask, mask_cleaned, dist_transform, sure_fg, sure_bg, unknown, markers

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def next_image(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.show_image()

    def clear_root_widgets(self):
        for widget in self.root.winfo_children():
            widget.destroy()

def main_interface(root, back_callback):
    app = ImageSegmentationDebugger(root, back_callback)

if __name__ == "__main__":
    root = Tk()
    main_interface(root, lambda: root.destroy())
    root.mainloop()
