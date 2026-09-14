"""
app.py - Modern Tkinter GUI for Image-Based Calculator.
Provides 3 interactive image slots (First Number, Operator, Second Number),
in-RAM original/working image management, interactive DIP filters, Reset button,
one-click Process & Calculate, and a visual DIP Pipeline step inspector.
"""

import os
import glob
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

import preprocessing
import edge_detection
import visualization
from image_processor import ImageProcessor

# Theme Color Palette
BG_MAIN = "#12141D"
CARD_BG = "#1E2232"
CARD_BORDER = "#2E344D"
PRIMARY_COLOR = "#4F46E5"
PRIMARY_HOVER = "#4338CA"
ACCENT_GREEN = "#10B981"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED = "#EF4444"
TEXT_PRIMARY = "#F9FAFB"
TEXT_SECONDARY = "#9CA3AF"
BTN_SECONDARY = "#374151"

class ImageSlotCard:
    def __init__(self, parent, title, slot_type, on_change_callback):
        self.parent = parent
        self.title = title
        self.slot_type = slot_type  # 'number' or 'operator'
        self.on_change_callback = on_change_callback
        
        self.original_image = None  # In RAM only (Immutable)
        self.working_image = None   # In RAM (Filters apply here)
        self.last_result = None
        self.tk_image = None
        
        self.build_ui()

    def build_ui(self):
        self.card = tk.Frame(self.parent, bg=CARD_BG, highlightbackground=CARD_BORDER, highlightthickness=1, padx=12, pady=12)
        
        # Title & Badge
        header_frame = tk.Frame(self.card, bg=CARD_BG)
        header_frame.pack(fill="x", pady=(0, 8))
        
        title_lbl = tk.Label(header_frame, text=self.title, font=("Segoe UI", 11, "bold"), fg=TEXT_PRIMARY, bg=CARD_BG)
        title_lbl.pack(side="left")
        
        self.badge_lbl = tk.Label(header_frame, text="Empty", font=("Segoe UI", 8, "bold"), fg=TEXT_SECONDARY, bg="#2D3348", padx=6, pady=2)
        self.badge_lbl.pack(side="right")
        
        # Preview Canvas (Aspect Ratio Preserved)
        self.canvas_w = 200
        self.canvas_h = 140
        self.canvas = tk.Canvas(self.card, width=self.canvas_w, height=self.canvas_h, bg="#0D0F17", highlightthickness=1, highlightbackground=CARD_BORDER)
        self.canvas.pack(pady=4)
        self.canvas.create_text(self.canvas_w // 2, self.canvas_h // 2, text="No Image Loaded\n(Click Browse or Sample)", fill=TEXT_SECONDARY, font=("Segoe UI", 9), justify="center")

        # Load / Sample Buttons
        btn_load_frame = tk.Frame(self.card, bg=CARD_BG)
        btn_load_frame.pack(fill="x", pady=6)
        
        self.btn_browse = tk.Button(btn_load_frame, text="📁 Choose Image", font=("Segoe UI", 9, "bold"), fg="white", bg=PRIMARY_COLOR, activebackground=PRIMARY_HOVER, relief="flat", cursor="hand2", command=self.browse_image)
        self.btn_browse.pack(side="left", expand=True, fill="x", padx=(0, 3))
        
        self.btn_sample = tk.Button(btn_load_frame, text="🎲 Sample", font=("Segoe UI", 9), fg=TEXT_PRIMARY, bg=BTN_SECONDARY, relief="flat", cursor="hand2", command=self.load_random_sample)
        self.btn_sample.pack(side="right", expand=True, fill="x", padx=(3, 0))

        # Filter Actions (Interactive DIP Enhancement)
        filter_label = tk.Label(self.card, text="Enhance / Quick Filters:", font=("Segoe UI", 8, "bold"), fg=TEXT_SECONDARY, bg=CARD_BG, anchor="w")
        filter_label.pack(fill="x", pady=(6, 2))
        
        filter_grid = tk.Frame(self.card, bg=CARD_BG)
        filter_grid.pack(fill="x", pady=2)
        
        filters = [
            ("Median", lambda: self.apply_quick_filter("median")),
            ("Gaussian", lambda: self.apply_quick_filter("gaussian")),
            ("Otsu Thresh", lambda: self.apply_quick_filter("otsu")),
            ("Morphology", lambda: self.apply_quick_filter("morph")),
            ("Canny Edges", lambda: self.apply_quick_filter("canny"))
        ]
        
        # Display filters in 2 rows
        row1 = tk.Frame(filter_grid, bg=CARD_BG)
        row1.pack(fill="x", pady=1)
        for name, cmd in filters[:3]:
            btn = tk.Button(row1, text=name, font=("Segoe UI", 8), fg=TEXT_PRIMARY, bg="#272C3E", relief="flat", cursor="hand2", command=cmd, pady=2)
            btn.pack(side="left", expand=True, fill="x", padx=1)
            
        row2 = tk.Frame(filter_grid, bg=CARD_BG)
        row2.pack(fill="x", pady=1)
        for name, cmd in filters[3:]:
            btn = tk.Button(row2, text=name, font=("Segoe UI", 8), fg=TEXT_PRIMARY, bg="#272C3E", relief="flat", cursor="hand2", command=cmd, pady=2)
            btn.pack(side="left", expand=True, fill="x", padx=1)

        # Reset Button (Reverts to original_image without modifying files)
        self.btn_reset = tk.Button(self.card, text="↺ Reset to Original", font=("Segoe UI", 9, "bold"), fg="#FFA000", bg="#2A2421", activebackground="#3D3028", relief="flat", cursor="hand2", command=self.reset_image)
        self.btn_reset.pack(fill="x", pady=(6, 4))
        
        # Detection Info Label
        self.info_lbl = tk.Label(self.card, text="Detected: None", font=("Segoe UI", 9), fg=TEXT_SECONDARY, bg=CARD_BG, anchor="center")
        self.info_lbl.pack(fill="x", pady=(4, 0))

    def set_image(self, bgr_img, filename="Sample"):
        """Loads image into RAM as original_image and working_image."""
        if bgr_img is None:
            return
        self.original_image = bgr_img.copy()
        self.working_image = bgr_img.copy()
        self.last_result = None
        self.badge_lbl.config(text=f"Loaded", fg=ACCENT_GREEN, bg="#1B382B")
        self.info_lbl.config(text=f"File: {os.path.basename(filename)[:15]}", fg=TEXT_SECONDARY)
        self.update_canvas_display()
        if self.on_change_callback:
            self.on_change_callback()

    def browse_image(self):
        path = filedialog.askopenfilename(
            title=f"Select {self.title}",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.set_image(img, path)

    def load_random_sample(self):
        pattern = "input/num_*_noisy.png" if self.slot_type == "number" else "input/op_*_noisy.png"
        candidates = glob.glob(pattern)
        if candidates:
            chosen = random.choice(candidates)
            img = cv2.imread(chosen)
            self.set_image(img, chosen)

    def apply_quick_filter(self, filter_type):
        if self.working_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        gray = preprocessing.to_grayscale(self.working_image)
        
        if filter_type == "median":
            filtered = preprocessing.apply_median_filter(gray, ksize=3)
        elif filter_type == "gaussian":
            filtered = preprocessing.apply_gaussian_blur(gray, ksize=(3, 3), sigma=1.0)
        elif filter_type == "otsu":
            filtered = preprocessing.apply_threshold(gray, method="otsu", invert=True)
        elif filter_type == "morph":
            # If not yet binary, threshold first
            if len(np.unique(gray)) > 2:
                binary = preprocessing.apply_threshold(gray, method="otsu", invert=True)
            else:
                binary = gray
            filtered = preprocessing.apply_morphology(binary, op="close", ksize=3)
        elif filter_type == "canny":
            filtered = edge_detection.apply_canny(gray)
        else:
            filtered = gray
            
        self.working_image = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR) if len(filtered.shape) == 2 else filtered
        self.update_canvas_display()
        self.badge_lbl.config(text="Filtered", fg=ACCENT_AMBER, bg="#382E19")

    def reset_image(self):
        """Restores working_image from RAM copy without touching file on disk."""
        if self.original_image is not None:
            self.working_image = self.original_image.copy()
            self.update_canvas_display()
            self.badge_lbl.config(text="Original", fg=TEXT_PRIMARY, bg="#2D3348")
            self.info_lbl.config(text="Reset to original", fg=TEXT_SECONDARY)

    def update_canvas_display(self, overlay_bgr=None):
        disp_img = overlay_bgr if overlay_bgr is not None else self.working_image
        if disp_img is None:
            return
            
        # Convert BGR to RGB for PIL
        if len(disp_img.shape) == 2:
            rgb = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
            
        h, w = rgb.shape[:2]
        # Preserve Aspect Ratio
        scale = min(self.canvas_w / float(w), self.canvas_h / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        
        self.canvas.delete("all")
        # Center image in canvas
        x_offset = (self.canvas_w - new_w) // 2
        y_offset = (self.canvas_h - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_image)

    def set_result(self, result_data):
        self.last_result = result_data
        if not result_data:
            return
            
        if self.slot_type == "number":
            val = result_data.get("value")
            conf = result_data.get("confidence", 0)
            txt = f"Detected: {val}  ({int(conf*100)}%)" if val is not None else "Unrecognized"
        else:
            sym = result_data.get("symbol")
            conf = result_data.get("confidence", 0)
            txt = f"Detected: {sym}  ({int(conf*100)}%)" if sym else "Unrecognized"
            
        color = ACCENT_GREEN if result_data.get("success") else ACCENT_RED
        self.info_lbl.config(text=txt, fg=color)
        
        # Display annotated image with bounding boxes
        annotated = visualization.draw_annotations(self.working_image, result_data)
        self.update_canvas_display(overlay_bgr=annotated)


class ImageCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image-Based Calculator — Digital Image Processing")
        self.root.geometry("860x720")
        self.root.minsize(820, 680)
        self.root.configure(bg=BG_MAIN)
        
        self.processor = ImageProcessor()
        self.last_expression_result = None
        
        self.build_ui()
        self.load_default_scenario()

    def build_ui(self):
        # 1. Header Section
        header = tk.Frame(self.root, bg=BG_MAIN, pady=16)
        header.pack(fill="x")
        
        title = tk.Label(header, text="IMAGE-BASED CALCULATOR", font=("Segoe UI", 18, "bold"), fg=TEXT_PRIMARY, bg=BG_MAIN)
        title.pack()
        
        subtitle = tk.Label(header, text="Digital Image Processing Pipeline • Canny Edge Detection • Template & Shape Recognition", font=("Segoe UI", 9), fg=TEXT_SECONDARY, bg=BG_MAIN)
        subtitle.pack(pady=(2, 0))

        # 2. Main Slots Container (3 Cards side by side)
        slots_container = tk.Frame(self.root, bg=BG_MAIN, padx=20)
        slots_container.pack(fill="x", pady=6)
        
        # Configure columns equally
        slots_container.columnconfigure(0, weight=1)
        slots_container.columnconfigure(1, weight=1)
        slots_container.columnconfigure(2, weight=1)

        self.slot1 = ImageSlotCard(slots_container, "1. First Number", "number", self.on_slot_change)
        self.slot1.card.grid(row=0, column=0, sticky="nsew", padx=6)

        self.slot_op = ImageSlotCard(slots_container, "2. Operator", "operator", self.on_slot_change)
        self.slot_op.card.grid(row=0, column=1, sticky="nsew", padx=6)

        self.slot2 = ImageSlotCard(slots_container, "3. Second Number", "number", self.on_slot_change)
        self.slot2.card.grid(row=0, column=2, sticky="nsew", padx=6)

        # 3. Action Section (Process & Calculate Button)
        action_frame = tk.Frame(self.root, bg=BG_MAIN, pady=14)
        action_frame.pack(fill="x")
        
        self.btn_process = tk.Button(
            action_frame,
            text="⚡ Process & Calculate",
            font=("Segoe UI", 13, "bold"),
            fg="white",
            bg=PRIMARY_COLOR,
            activebackground=PRIMARY_HOVER,
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            padx=28,
            pady=10,
            command=self.process_and_calculate
        )
        self.btn_process.pack()

        # 4. Result & Expression Card
        self.result_card = tk.Frame(self.root, bg=CARD_BG, highlightbackground=CARD_BORDER, highlightthickness=1, padx=20, pady=16)
        self.result_card.pack(fill="x", padx=26, pady=(4, 10))

        res_header = tk.Label(self.result_card, text="CALCULATION RESULT", font=("Segoe UI", 10, "bold"), fg=TEXT_SECONDARY, bg=CARD_BG)
        res_header.pack(anchor="w")

        # Main Expression Display
        self.expr_lbl = tk.Label(
            self.result_card,
            text="Detected Expression: [ ? ? ? ] = ?",
            font=("Segoe UI", 20, "bold"),
            fg="#60A5FA",
            bg=CARD_BG,
            pady=6
        )
        self.expr_lbl.pack()

        self.status_lbl = tk.Label(
            self.result_card,
            text="Ready. Click 'Process & Calculate' to analyze images.",
            font=("Segoe UI", 10),
            fg=TEXT_SECONDARY,
            bg=CARD_BG
        )
        self.status_lbl.pack()

        # Visual Pipeline Inspector Button
        self.btn_inspect = tk.Button(
            self.result_card,
            text="🔍 View Pipeline Steps (DIP Visualizer)",
            font=("Segoe UI", 9, "bold"),
            fg=TEXT_PRIMARY,
            bg=BTN_SECONDARY,
            relief="flat",
            cursor="hand2",
            command=self.show_pipeline_dialog
        )
        self.btn_inspect.pack(pady=(10, 0))

    def on_slot_change(self):
        self.expr_lbl.config(text="Detected Expression: [ ? ? ? ] = ?", fg="#60A5FA")
        self.status_lbl.config(text="Images modified. Click 'Process & Calculate' to run recognition.", fg=TEXT_SECONDARY)

    def load_default_scenario(self):
        """Loads the standard 92 ÷ 55 demonstration scenario from generated input folder."""
        f_92 = "input/num_9_92_noisy.png"
        f_div = "input/op_div_noisy.png"
        f_55 = "input/num_5_55_noisy.png"
        
        if os.path.exists(f_92) and os.path.exists(f_div) and os.path.exists(f_55):
            self.slot1.set_image(cv2.imread(f_92), f_92)
            self.slot_op.set_image(cv2.imread(f_div), f_div)
            self.slot2.set_image(cv2.imread(f_55), f_55)

    def process_and_calculate(self):
        img1 = self.slot1.working_image
        img_op = self.slot_op.working_image
        img2 = self.slot2.working_image
        
        if img1 is None or img_op is None or img2 is None:
            messagebox.showwarning("Incomplete Input", "Please provide all 3 images (First Number, Operator, Second Number).")
            return

        # Process through DIP pipeline with automatic fallback
        res = self.processor.process_expression(img1, img_op, img2)
        self.last_expression_result = res
        
        # Update each card with its annotations and confidence
        self.slot1.set_result(res["num1"])
        self.slot_op.set_result(res["op"])
        self.slot2.set_result(res["num2"])
        
        calc = res["calculation"]
        if calc.get("success"):
            equation_str = f"Detected Expression: [ {calc['expression']} ] = {calc['result_str']}"
            self.expr_lbl.config(text=equation_str, fg=ACCENT_GREEN)
            conf_pct = int(res["overall_confidence"] * 100)
            self.status_lbl.config(text=f"Calculation Successful • Pipeline Confidence: {conf_pct}%", fg=TEXT_PRIMARY)
        else:
            self.expr_lbl.config(text=f"Detected Expression: [ {calc.get('expression', 'Error')} ]", fg=ACCENT_RED)
            self.status_lbl.config(text=f"Error: {calc.get('error', 'Calculation Failed')}", fg=ACCENT_RED)

    def show_pipeline_dialog(self):
        """Opens a modal dialog showing the step-by-step DIP stages for all 3 images."""
        if not self.last_expression_result:
            messagebox.showinfo("Info", "Please run 'Process & Calculate' first to inspect the pipeline.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("DIP Pipeline Stages Visualizer")
        dialog.geometry("920x640")
        dialog.configure(bg=BG_MAIN)
        dialog.grab_set()

        # Title
        t_lbl = tk.Label(dialog, text="DIGITAL IMAGE PROCESSING PIPELINE STAGES", font=("Segoe UI", 12, "bold"), fg=TEXT_PRIMARY, bg=BG_MAIN)
        t_lbl.pack(pady=10)

        # Scrollable container for the montages
        canvas_container = tk.Frame(dialog, bg=BG_MAIN)
        canvas_container.pack(fill="both", expand=True, padx=12, pady=6)

        c = tk.Canvas(canvas_container, bg=BG_MAIN, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=c.yview)
        scroll_frame = tk.Frame(c, bg=BG_MAIN)
        
        scroll_frame.bind(
            "<Configure>",
            lambda e: c.configure(scrollregion=c.bbox("all"))
        )
        c.create_window((0, 0), window=scroll_frame, anchor="nw")
        c.configure(yscrollcommand=scrollbar.set)
        
        c.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        items = [
            ("1. First Number Pipeline", self.last_expression_result["num1"]),
            ("2. Operator Pipeline", self.last_expression_result["op"]),
            ("3. Second Number Pipeline", self.last_expression_result["num2"])
        ]

        dialog.photo_refs = []  # Keep references
        for section_title, data in items:
            sec_lbl = tk.Label(scroll_frame, text=f"{section_title} (Used: {data.get('pipeline_used')})", font=("Segoe UI", 10, "bold"), fg="#60A5FA", bg=BG_MAIN, anchor="w")
            sec_lbl.pack(fill="x", pady=(10, 4), padx=6)
            
            montage_bgr = visualization.create_pipeline_montage(data["stages"], cell_size=(90, 130))
            rgb = cv2.cvtColor(montage_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(pil_img)
            dialog.photo_refs.append(tk_img)
            
            img_lbl = tk.Label(scroll_frame, image=tk_img, bg=BG_MAIN)
            img_lbl.pack(padx=6, pady=4)

        close_btn = tk.Button(dialog, text="Close", font=("Segoe UI", 9, "bold"), fg="white", bg=BTN_SECONDARY, relief="flat", cursor="hand2", command=dialog.destroy, padx=20, pady=6)
        close_btn.pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCalculatorApp(root)
    root.mainloop()
