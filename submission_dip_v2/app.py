"""
app.py - Advanced Modern Tkinter GUI for Image-Based Calculator.
Part of DIP-Lab-6 Image-Based Calculator (v2 Architecture).

Features:
- Sleek modern Slate/Emerald interface with responsive card containers.
- 3 Dedicated Image Slots (First Operand, Operator, Second Operand).
- In-RAM immutable original image & editable working image management.
- Quick interactive DIP filters with instant canvas preview.
- One-click end-to-end "Process & Calculate" pipeline execution.
- Digital equation breakdown display with real-time confidence scores.
- Interactive multi-stage DIP Pipeline Inspector montage.
"""

import os
import glob
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from dip_engine import DIPEngine
from processor import SystemImageProcessor
from visualizer import draw_detection_overlays, build_pipeline_strip

# Modern Slate & Emerald Theme Palette
BG_MAIN = "#0F172A"       # Slate 900
PANEL_BG = "#1E293B"      # Slate 800
PANEL_BORDER = "#334155"  # Slate 700
ACCENT_BLUE = "#0EA5E9"   # Sky 500
ACCENT_BLUE_HOVER = "#0284C7"
ACCENT_GREEN = "#10B981"  # Emerald 500
ACCENT_GREEN_HOVER = "#059669"
ACCENT_AMBER = "#F59E0B"  # Amber 500
ACCENT_RED = "#EF4444"    # Red 500
TEXT_MAIN = "#F8FAFC"     # Slate 50
TEXT_MUTED = "#94A3B8"    # Slate 400
CANVAS_BG = "#020617"     # Slate 950
BTN_BG = "#27354A"        # Slate 750

class ImageSlotView:
    """Represents an interactive slot for loading, filtering, and previewing an image."""
    
    def __init__(self, parent, title: str, slot_type: str, on_update_cb):
        self.parent = parent
        self.title = title
        self.slot_type = slot_type  # 'number' or 'operator'
        self.on_update_cb = on_update_cb
        
        self.original_image: np.ndarray = None  # Immutable in RAM
        self.working_image: np.ndarray = None   # Filters apply here
        self.last_result: dict = None
        self.tk_photo = None
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.build_ui()

    def build_ui(self):
        self.frame = tk.Frame(self.parent, bg=PANEL_BG, highlightbackground=PANEL_BORDER, highlightthickness=1, padx=10, pady=10)
        
        # Header
        hdr = tk.Frame(self.frame, bg=PANEL_BG)
        hdr.pack(fill="x", pady=(0, 6))
        
        lbl_title = tk.Label(hdr, text=self.title, font=("Segoe UI", 10, "bold"), fg=TEXT_MAIN, bg=PANEL_BG)
        lbl_title.pack(side="left")
        
        self.badge = tk.Label(hdr, text="فارغ", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg="#334155", padx=6, pady=1)
        self.badge.pack(side="right")
        
        # Canvas
        self.cw, self.ch = 210, 140
        self.canvas = tk.Canvas(self.frame, width=self.cw, height=self.ch, bg=CANVAS_BG, highlightthickness=1, highlightbackground=PANEL_BORDER)
        self.canvas.pack(pady=4)
        self.canvas.create_text(self.cw // 2, self.ch // 2, text="انقر لاختيار صورة\nأو عينة عشوائية", fill=TEXT_MUTED, font=("Segoe UI", 9), justify="center")
        self.canvas.bind("<Button-1>", lambda e: self.on_update_cb(self))

        # Browse & Sample Buttons
        btn_bar = tk.Frame(self.frame, bg=PANEL_BG)
        btn_bar.pack(fill="x", pady=4)
        
        btn_browse = tk.Button(btn_bar, text="📁 اختيار صورة", font=("Segoe UI", 8, "bold"), fg="white", bg=ACCENT_BLUE, activebackground=ACCENT_BLUE_HOVER, relief="flat", cursor="hand2", command=self.browse_file)
        btn_browse.pack(side="left", expand=True, fill="x", padx=(0, 2))
        
        btn_sample = tk.Button(btn_bar, text="🎲 عينة", font=("Segoe UI", 8), fg=TEXT_MAIN, bg=BTN_BG, activebackground="#334155", relief="flat", cursor="hand2", command=self.load_random_sample)
        btn_sample.pack(side="right", expand=True, fill="x", padx=(2, 0))

        # Quick DIP Filters Toolbar
        flt_lbl = tk.Label(self.frame, text="فلاتر معالجة سريعة (DIP Filters):", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=PANEL_BG, anchor="w")
        flt_lbl.pack(fill="x", pady=(4, 2))
        
        flt_box = tk.Frame(self.frame, bg=PANEL_BG)
        flt_box.pack(fill="x", pady=1)
        
        filters = [
            ("Median", lambda: self.apply_filter("median")),
            ("Gaussian", lambda: self.apply_filter("gaussian")),
            ("Otsu", lambda: self.apply_filter("otsu")),
            ("Morph", lambda: self.apply_filter("morph")),
            ("Canny", lambda: self.apply_filter("canny"))
        ]
        
        for name, cmd in filters:
            btn = tk.Button(flt_box, text=name, font=("Segoe UI", 7), fg=TEXT_MAIN, bg="#263346", activebackground=ACCENT_BLUE, relief="flat", cursor="hand2", command=cmd, pady=2)
            btn.pack(side="left", expand=True, fill="x", padx=1)

        # Reset Slot Button
        btn_revert = tk.Button(self.frame, text="↺ استعادة الأصل (Reset)", font=("Segoe UI", 7), fg=TEXT_MUTED, bg=PANEL_BG, relief="flat", cursor="hand2", command=self.reset_working_image)
        btn_revert.pack(fill="x", pady=(3, 0))

    def set_image(self, img: np.ndarray, filename: str = ""):
        """Stores original image in RAM immutably and prepares working copy."""
        if img is None:
            return
        self.original_image = img.copy()  # Safe RAM copy
        self.working_image = img.copy()
        self.last_result = None
        
        disp_name = os.path.basename(filename)[:15] if filename else "Loaded"
        self.badge.config(text=disp_name, fg="#10B981", bg="#064E3B")
        self.render_preview(self.working_image)
        self.on_update_cb(self)

    def browse_file(self):
        initial_dir = os.path.join(self.base_dir, "input")
        if not os.path.exists(initial_dir):
            initial_dir = self.base_dir
            
        fpath = filedialog.askopenfilename(
            initialdir=initial_dir,
            title=f"اختر صورة: {self.title}",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if fpath:
            img = cv2.imread(fpath)
            if img is not None:
                self.set_image(img, fpath)
            else:
                messagebox.showerror("خطأ", "تعذر قراءة ملف الصورة المحدد.")

    def load_random_sample(self):
        input_dir = os.path.join(self.base_dir, "input")
        if not os.path.exists(input_dir):
            messagebox.showwarning("تنبيه", "مجلد العينات input غير موجود.")
            return
            
        pattern = "num_*_noisy.png" if self.slot_type == "number" else "op_*_noisy.png"
        candidates = glob.glob(os.path.join(input_dir, pattern))
        if candidates:
            chosen = random.choice(candidates)
            img = cv2.imread(chosen)
            self.set_image(img, chosen)
        else:
            messagebox.showinfo("معلومة", f"لم يتم العثور على عينات بنمط {pattern}")

    def apply_filter(self, filter_type: str):
        """Applies quick interactive DIP filter to working copy in RAM."""
        if self.working_image is None:
            return
            
        gray = DIPEngine.to_grayscale(self.working_image)
        
        if filter_type == "median":
            filtered = DIPEngine.apply_median(gray, kernel_size=3)
        elif filter_type == "gaussian":
            filtered = DIPEngine.apply_gaussian(gray, ksize=(3, 3), sigma=1.0)
        elif filter_type == "otsu":
            filtered = DIPEngine.binarize_otsu(gray, invert=True)
        elif filter_type == "morph":
            bin_img = DIPEngine.binarize_otsu(gray, invert=True)
            filtered = DIPEngine.morphology_close(bin_img, ksize=3)
        elif filter_type == "canny":
            filtered = DIPEngine.detect_canny_edges(gray)
        else:
            filtered = gray
            
        self.working_image = filtered
        self.render_preview(self.working_image)
        self.badge.config(text=filter_type.capitalize(), fg="#F59E0B", bg="#451A03")
        self.on_update_cb(self)

    def reset_working_image(self):
        if self.original_image is not None:
            self.working_image = self.original_image.copy()
            self.render_preview(self.working_image)
            self.badge.config(text="Original", fg="#10B981", bg="#064E3B")
            self.on_update_cb(self)

    def render_preview(self, img: np.ndarray, annotated_result: dict = None):
        """Displays image centered on canvas preserving aspect ratio."""
        if img is None:
            return
            
        display_mat = img
        if annotated_result:
            display_mat = draw_detection_overlays(img, annotated_result)
            
        # Convert to RGB
        if len(display_mat.shape) == 2:
            rgb = cv2.cvtColor(display_mat, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(display_mat, cv2.COLOR_BGR2RGB)
            
        ih, iw = rgb.shape[:2]
        scale = min(float(self.cw) / iw, float(self.ch) / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        self.tk_photo = ImageTk.PhotoImage(pil_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.cw // 2, self.ch // 2, image=self.tk_photo, anchor="center")


class DIPCalculatorApp:
    """Main Application Controller."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image-Based Smart Calculator (DIP-Lab-6 • v2 Architecture)")
        self.root.geometry("880 driving:820".split()[0] if False else "880x800")
        self.root.minsize(820, 720)
        self.root.configure(bg=BG_MAIN)
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.processor = SystemImageProcessor(base_dir=self.base_dir)
        self.active_inspected_slot = None
        
        self.build_gui()
        self.load_default_samples()

    def build_gui(self):
        # 1. Top Header Banner
        hdr_frame = tk.Frame(self.root, bg=PANEL_BG, padx=16, pady=12, highlightbackground=PANEL_BORDER, highlightthickness=1)
        hdr_frame.pack(fill="x", padx=14, pady=(12, 8))
        
        title_box = tk.Frame(hdr_frame, bg=PANEL_BG)
        title_box.pack(side="left")
        
        lbl_app_title = tk.Label(title_box, text="آلة حاسبة رقمية معتمدة على معالجة الصور الرقمية (DIP-Lab-6)", font=("Segoe UI", 13, "bold"), fg=TEXT_MAIN, bg=PANEL_BG)
        lbl_app_title.pack(anchor="w")
        
        lbl_app_sub = tk.Label(title_box, text="DIP Multi-Pipeline Fallback • Canny Edges • Template & Structural Recognition • Safe Math", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=PANEL_BG)
        lbl_app_sub.pack(anchor="w")

        btn_reset_all = tk.Button(hdr_frame, text="↺ تفريغ الكل", font=("Segoe UI", 8, "bold"), fg=TEXT_MAIN, bg="#334155", activebackground=ACCENT_RED, relief="flat", cursor="hand2", padx=10, pady=4, command=self.reset_all)
        btn_reset_all.pack(side="right")

        # 2. Slots Container (3 Cards)
        slots_container = tk.Frame(self.root, bg=BG_MAIN)
        slots_container.pack(fill="x", padx=10, pady=4)
        
        self.slot_num1 = ImageSlotView(slots_container, "الرقم الأول (Operand 1)", "number", self.on_slot_selected)
        self.slot_num1.frame.pack(side="left", expand=True, fill="both", padx=4)
        
        self.slot_op = ImageSlotView(slots_container, "العملية (Operator)", "operator", self.on_slot_selected)
        self.slot_op.frame.pack(side="left", expand=True, fill="both", padx=4)
        
        self.slot_num2 = ImageSlotView(slots_container, "الرقم الثاني (Operand 2)", "number", self.on_slot_selected)
        self.slot_num2.frame.pack(side="left", expand=True, fill="both", padx=4)

        # 3. Action Toolbar (Calculate Button)
        action_bar = tk.Frame(self.root, bg=BG_MAIN)
        action_bar.pack(fill="x", padx=14, pady=8)
        
        self.btn_calculate = tk.Button(
            action_bar,
            text="⚡ معالجة الصور وحساب التعبير الرياضي (Process & Calculate)",
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg=ACCENT_GREEN,
            activebackground=ACCENT_GREEN_HOVER,
            relief="flat",
            cursor="hand2",
            pady=8,
            command=self.process_and_calculate
        )
        self.btn_calculate.pack(fill="x")

        # 4. Digital Mathematical Display Screen
        self.screen_frame = tk.Frame(self.root, bg=CANVAS_BG, highlightbackground=PANEL_BORDER, highlightthickness=2, padx=14, pady=10)
        self.screen_frame.pack(fill="x", padx=14, pady=4)
        
        screen_top = tk.Frame(self.screen_frame, bg=CANVAS_BG)
        screen_top.pack(fill="x")
        
        lbl_screen_tag = tk.Label(screen_top, text="RESULT DISPLAY (شاشة النتيجة الحسابية):", font=("Consolas", 8, "bold"), fg=ACCENT_BLUE, bg=CANVAS_BG)
        lbl_screen_tag.pack(side="left")
        
        self.lbl_pipeline_tag = tk.Label(screen_top, text="Pipeline: None", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=CANVAS_BG)
        self.lbl_pipeline_tag.pack(side="right")
        
        # Big Equation Display
        self.lbl_equation = tk.Label(self.screen_frame, text="-- ? -- = ?", font=("Consolas", 22, "bold"), fg="#38BDF8", bg=CANVAS_BG, pady=4)
        self.lbl_equation.pack()

        # Breakdown Badges
        self.badges_frame = tk.Frame(self.screen_frame, bg=CANVAS_BG)
        self.badges_frame.pack(fill="x", pady=(4, 0))
        
        self.badge_num1 = tk.Label(self.badges_frame, text="Num1: --", font=("Segoe UI", 8), fg=TEXT_MUTED, bg="#1E293B", padx=6, pady=2)
        self.badge_num1.pack(side="left", padx=2)
        
        self.badge_op = tk.Label(self.badges_frame, text="Op: --", font=("Segoe UI", 8), fg=TEXT_MUTED, bg="#1E293B", padx=6, pady=2)
        self.badge_op.pack(side="left", padx=2)
        
        self.badge_num2 = tk.Label(self.badges_frame, text="Num2: --", font=("Segoe UI", 8), fg=TEXT_MUTED, bg="#1E293B", padx=6, pady=2)
        self.badge_num2.pack(side="left", padx=2)
        
        self.badge_conf = tk.Label(self.badges_frame, text="الثقة الإجمالية: --%", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg="#1E293B", padx=6, pady=2)
        self.badge_conf.pack(side="right", padx=2)

        # 5. Visual Pipeline Inspector Section
        insp_container = tk.Frame(self.root, bg=PANEL_BG, highlightbackground=PANEL_BORDER, highlightthickness=1, padx=10, pady=8)
        insp_container.pack(fill="both", expand=True, padx=14, pady=(6, 12))
        
        insp_hdr = tk.Frame(insp_container, bg=PANEL_BG)
        insp_hdr.pack(fill="x", pady=(0, 4))
        
        self.lbl_inspector_title = tk.Label(insp_hdr, text="🔬 مستعرض مراحل خط المعالجة (DIP Pipeline Inspector):", font=("Segoe UI", 9, "bold"), fg=TEXT_MAIN, bg=PANEL_BG)
        self.lbl_inspector_title.pack(side="left")
        
        lbl_hint = tk.Label(insp_hdr, text="(انقر على أي صورة أعلاه لعرض مراحلها)", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=PANEL_BG)
        lbl_hint.pack(side="left", padx=6)
        
        # Inspector Canvas
        self.insp_canvas = tk.Canvas(insp_container, bg=CANVAS_BG, highlightthickness=0)
        self.insp_canvas.pack(fill="both", expand=True)
        self.insp_photo = None

    def on_slot_selected(self, slot: ImageSlotView):
        self.active_inspected_slot = slot
        if slot.last_result and "stages" in slot.last_result:
            self.display_pipeline_strip(slot.last_result["stages"], f"مراحل معالجة: {slot.title}")
        elif slot.working_image is not None:
            # Run on-demand pipeline preview
            prep = self.processor.pipelines[0]
            stages = self.processor.process_number(slot.working_image)["stages"] if slot.slot_type == "number" else self.processor.process_operator(slot.working_image)["stages"]
            self.display_pipeline_strip(stages, f"مراحل معالجة سريعة: {slot.title}")

    def display_pipeline_strip(self, stages: dict, title: str = ""):
        if not stages:
            return
            
        strip_mat = build_pipeline_strip(stages, cell_dim=(110, 135))
        h, w = strip_mat.shape[:2]
        
        # Canvas dimensions
        cw = self.insp_canvas.winfo_width()
        if cw < 50:
            cw = 840
        ch = max(140, self.insp_canvas.winfo_height())
        
        # Scale strip to fit nicely inside canvas
        scale = min(float(cw) / w, float(ch) / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        
        resized = cv2.resize(strip_mat, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.insp_photo = ImageTk.PhotoImage(pil_img)
        
        self.insp_canvas.delete("all")
        self.insp_canvas.create_image(cw // 2, ch // 2, image=self.insp_photo, anchor="center")
        if title:
            self.lbl_inspector_title.config(text=f"🔬 {title}")

    def process_and_calculate(self):
        """Executes full multi-stage calculation pipeline across all 3 slots."""
        if self.slot_num1.working_image is None or self.slot_op.working_image is None or self.slot_num2.working_image is None:
            messagebox.showwarning("تنبيه", "يرجى تحميل الصور الثلاث أولاً (الرقم الأول، العملية، الرقم الثاني).")
            return
            
        # Process full expression
        result = self.processor.process_expression(
            self.slot_num1.working_image,
            self.slot_op.working_image,
            self.slot_num2.working_image
        )
        
        res_n1 = result["num1"]
        res_op = result["op"]
        res_n2 = result["num2"]
        calc = result["calculation"]
        
        # Cache results into slots
        self.slot_num1.last_result = res_n1
        self.slot_op.last_result = res_op
        self.slot_num2.last_result = res_n2
        
        # Update Slot Annotations
        self.slot_num1.render_preview(self.slot_num1.working_image, res_n1)
        self.slot_op.render_preview(self.slot_op.working_image, res_op)
        self.slot_num2.render_preview(self.slot_num2.working_image, res_n2)
        
        # Update Digital Screen Display
        eq_text = calc.get("full_equation", "N/A")
        self.lbl_equation.config(text=eq_text)
        
        pipe_str = f"Pipelines: {res_n1['pipeline_id'].capitalize()} | {res_op['pipeline_id'].capitalize()} | {res_n2['pipeline_id'].capitalize()}"
        self.lbl_pipeline_tag.config(text=pipe_str)
        
        # Update Breakdown Badges
        n1_str = f"{res_n1['string']} ({int(res_n1['confidence']*100)}%)" if res_n1['success'] else "Num1: ?"
        op_str = f"{res_op['symbol']} ({int(res_op['confidence']*100)}%)" if res_op['success'] else "Op: ?"
        n2_str = f"{res_n2['string']} ({int(res_n2['confidence']*100)}%)" if res_n2['success'] else "Num2: ?"
        
        self.badge_num1.config(text=f"Num1: {n1_str}", fg="#38BDF8")
        self.badge_op.config(text=f"Op: {op_str}", fg="#38BDF8")
        self.badge_num2.config(text=f"Num2: {n2_str}", fg="#38BDF8")
        
        total_conf = int(result["overall_confidence"] * 100)
        conf_color = "#10B981" if total_conf >= 85 else "#F59E0B"
        self.badge_conf.config(text=f"الثقة: {total_conf}%", fg=conf_color)
        
        # Show strip of operand 1 by default
        self.display_pipeline_strip(res_n1["stages"], f"مراحل معالجة الرقم الأول ({res_n1['string']})")

    def reset_all(self):
        for slot in [self.slot_num1, self.slot_op, self.slot_num2]:
            slot.original_image = None
            slot.working_image = None
            slot.last_result = None
            slot.canvas.delete("all")
            slot.canvas.create_text(slot.cw // 2, slot.ch // 2, text="انقر لاختيار صورة\nأو عينة عشوائية", fill=TEXT_MUTED, font=("Segoe UI", 9), justify="center")
            slot.badge.config(text="فارغ", fg=TEXT_MUTED, bg="#334155")
            
        self.lbl_equation.config(text="-- ? -- = ?")
        self.lbl_pipeline_tag.config(text="Pipeline: None")
        self.badge_num1.config(text="Num1: --", fg=TEXT_MUTED)
        self.badge_op.config(text="Op: --", fg=TEXT_MUTED)
        self.badge_num2.config(text="Num2: --", fg=TEXT_MUTED)
        self.badge_conf.config(text="الثقة: --%", fg=TEXT_MUTED)
        self.insp_canvas.delete("all")
        self.lbl_inspector_title.config(text="🔬 مستعرض مراحل خط المعالجة (DIP Pipeline Inspector):")

    def load_default_samples(self):
        """Loads default demonstration expression: 95 ÷ 15"""
        input_dir = os.path.join(self.base_dir, "input")
        p_n1 = os.path.join(input_dir, "num_9_95_noisy.png")
        p_op = os.path.join(input_dir, "op_div_noisy.png")
        p_n2 = os.path.join(input_dir, "num_2_15_noisy.png")
        
        if os.path.exists(p_n1) and os.path.exists(p_op) and os.path.exists(p_n2):
            self.slot_num1.set_image(cv2.imread(p_n1), p_n1)
            self.slot_op.set_image(cv2.imread(p_op), p_op)
            self.slot_num2.set_image(cv2.imread(p_n2), p_n2)

if __name__ == "__main__":
    app_root = tk.Tk()
    app = DIPCalculatorApp(app_root)
    app_root.mainloop()
