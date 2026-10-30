"""OpenRAW Denoise - Desktop Application (Canon EOS 6D AI RAW Denoiser)."""
import glob
import os
import subprocess
import sys
import threading
import time
from tkinter import filedialog, messagebox

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk
import rawpy
import torch
import windnd

from denoise.cfa import pack_bayer
from denoise.dng_writer import write_linear_dng
from denoise.tiling import process_tiled
from models.nafnet import NAFNet

# Set dark theme matching Lightroom / DxO
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class RawDenoiseApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OpenRAW Denoise - Canon EOS 6D Edition")
        self.geometry("1220x780")
        self.minsize(1020, 660)

        # Device selection & model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = os.path.abspath("models/checkpoints/nafnet_best.pt")
        self.model = None
        self.is_processing = False

        self.files_queue = []  # list of dicts: {"path": str, "name": str, "iso": int, "status": str, "time": str, "out_dng": str}
        self.selected_idx = None
        self.output_dir = os.path.abspath("data/denoised_exports")

        self._build_ui()
        self._load_model_async()

        # Hook native Windows Drag and Drop
        try:
            windnd.hook_dropfiles(self, func=self._on_windows_drop)
        except Exception as e:
            print(f"Notice: Drag-and-drop hook: {e}")

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=4, minsize=480)
        self.grid_columnconfigure(1, weight=5, minsize=520)
        self.grid_rowconfigure(0, weight=1)

        # ================= LEFT PANEL =================
        left_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#18181b")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left_frame.grid_rowconfigure(4, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # Header Title & Hardware Badge
        title_box = ctk.CTkFrame(left_frame, fg_color="transparent")
        title_box.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 6))

        title_lbl = ctk.CTkLabel(
            title_box,
            text="OpenRAW Denoise",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#f4f4f5",
        )
        title_lbl.pack(side="left")

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.badge = ctk.CTkLabel(
            title_box,
            text=f" GPU: {gpu_name} ",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#0284c7" if torch.cuda.is_available() else "#71717a",
            text_color="#ffffff",
            corner_radius=6,
        )
        self.badge.pack(side="right")

        # Dropzone Banner
        drop_banner = ctk.CTkFrame(left_frame, fg_color="#27272a", corner_radius=8, border_width=1, border_color="#3f3f46")
        drop_banner.grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 8))

        drop_lbl = ctk.CTkLabel(
            drop_banner,
            text="📁 Drag & Drop any RAW Folder here to Auto-Process",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#38bdf8",
        )
        drop_lbl.pack(pady=10)

        # Action Buttons row
        btn_bar = ctk.CTkFrame(left_frame, fg_color="transparent")
        btn_bar.grid(row=2, column=0, sticky="ew", padx=16, pady=4)

        add_folder_btn = ctk.CTkButton(
            btn_bar,
            text="+ Add Folder",
            command=self._on_add_folder,
            width=110,
            fg_color="#0284c7",
            hover_color="#0369a1",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        add_folder_btn.pack(side="left", padx=(0, 6))

        add_files_btn = ctk.CTkButton(
            btn_bar,
            text="+ Add Files",
            command=self._on_add_files,
            width=100,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        add_files_btn.pack(side="left", padx=(0, 6))

        clear_btn = ctk.CTkButton(
            btn_bar,
            text="Clear",
            command=self._on_clear_queue,
            width=70,
            fg_color="#27272a",
            hover_color="#3f3f46",
        )
        clear_btn.pack(side="left")

        # Auto-start switch
        self.autostart_var = ctk.BooleanVar(value=True)
        auto_switch = ctk.CTkSwitch(
            btn_bar,
            text="Auto-Start",
            variable=self.autostart_var,
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8",
            progress_color="#10b981",
        )
        auto_switch.pack(side="right")

        # Output folder row
        out_bar = ctk.CTkFrame(left_frame, fg_color="transparent")
        out_bar.grid(row=3, column=0, sticky="ew", padx=16, pady=4)

        out_lbl = ctk.CTkLabel(out_bar, text="Output:", font=ctk.CTkFont(size=12), text_color="#a1a1aa")
        out_lbl.pack(side="left", padx=(0, 6))

        self.out_entry = ctk.CTkEntry(out_bar, font=ctk.CTkFont(size=11))
        self.out_entry.insert(0, self.output_dir)
        self.out_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        browse_btn = ctk.CTkButton(
            out_bar,
            text="Browse",
            width=65,
            command=self._on_browse_output,
            fg_color="#3f3f46",
            hover_color="#52525b",
        )
        browse_btn.pack(side="right")

        # Queue List Table (Scrollable)
        self.queue_frame = ctk.CTkScrollableFrame(left_frame, fg_color="#09090b", corner_radius=8)
        self.queue_frame.grid(row=4, column=0, sticky="nsew", padx=16, pady=8)
        self.queue_frame.grid_columnconfigure(0, weight=5)
        self.queue_frame.grid_columnconfigure(1, weight=2)
        self.queue_frame.grid_columnconfigure(2, weight=3)

        # Progress bar & Process Button
        bottom_box = ctk.CTkFrame(left_frame, fg_color="transparent")
        bottom_box.grid(row=5, column=0, sticky="ew", padx=16, pady=(6, 16))

        self.status_lbl = ctk.CTkLabel(
            bottom_box,
            text="Ready. Drop a folder or add files.",
            font=ctk.CTkFont(size=12),
            text_color="#a1a1aa",
            anchor="w",
        )
        self.status_lbl.pack(fill="x", pady=(0, 4))

        self.progress_bar = ctk.CTkProgressBar(bottom_box)
        self.progress_bar.set(0.0)
        self.progress_bar.pack(fill="x", pady=(0, 10))

        self.process_btn = ctk.CTkButton(
            bottom_box,
            text="START BATCH DENOISE",
            command=self._on_start_processing,
            height=46,
            fg_color="#10b981",
            hover_color="#059669",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self.process_btn.pack(fill="x")

        # ================= RIGHT PANEL (PREVIEW) =================
        right_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#18181b")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Preview Header
        prev_header = ctk.CTkFrame(right_frame, fg_color="transparent")
        prev_header.grid(row=0, column=0, sticky="ew", padx=16, pady=12)

        prev_title = ctk.CTkLabel(
            prev_header,
            text="Interactive 100% Loupe Preview",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#f4f4f5",
        )
        prev_title.pack(side="left")

        open_folder_btn = ctk.CTkButton(
            prev_header,
            text="📂 Open Output Folder",
            width=150,
            command=self._open_output_folder,
            fg_color="#27272a",
            hover_color="#3f3f46",
            font=ctk.CTkFont(size=12),
        )
        open_folder_btn.pack(side="right")

        # Canvas / Image Display
        self.preview_canvas = ctk.CTkLabel(
            right_frame,
            text="Drop a folder or select an image to inspect the 100% Before/After preview",
            fg_color="#09090b",
            corner_radius=8,
            text_color="#71717a",
            font=ctk.CTkFont(size=13),
        )
        self.preview_canvas.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

    def _load_model_async(self):
        def worker():
            try:
                self.status_lbl.configure(text="Loading NAFNet model to RTX 5060 GPU...")
                m = NAFNet(img_channel=5, out_channel=4, width=32).to(self.device)
                if os.path.exists(self.weights_path):
                    state = torch.load(self.weights_path, map_location=self.device)
                    m.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
                m.eval()
                self.model = m
                dev_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
                self.status_lbl.configure(text=f"Ready. Model active on {dev_name}.")
            except Exception as e:
                self.status_lbl.configure(text=f"Model load error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_windows_drop(self, files):
        """Called when user drags and drops files/folders from Windows Explorer."""
        dropped_paths = [f.decode("gbk", errors="ignore") if isinstance(f, bytes) else str(f) for f in files]
        self._ingest_paths(dropped_paths)

    def _on_add_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing RAW Photos")
        if folder:
            self._ingest_paths([folder])

    def _on_add_files(self):
        files = filedialog.askopenfilenames(
            title="Select Canon RAW Files",
            filetypes=[("Canon RAW", "*.CR2"), ("All Files", "*.*")],
        )
        if files:
            self._ingest_paths(list(files))

    def _ingest_paths(self, paths: list):
        """Recursively scan dropped folders and files for .CR2 raw photos."""
        found_files = []
        target_folder = None

        for p in paths:
            if os.path.isdir(p):
                if target_folder is None:
                    target_folder = p
                found_files.extend(glob.glob(os.path.join(p, "**", "*.CR2"), recursive=True))
                found_files.extend(glob.glob(os.path.join(p, "**", "*.cr2"), recursive=True))
            elif os.path.isfile(p) and p.lower().endswith(".cr2"):
                if target_folder is None:
                    target_folder = os.path.dirname(p)
                found_files.append(p)

        found_files = sorted(list(set(found_files)))
        if not found_files:
            messagebox.showinfo("No RAW Files", "No Canon .CR2 files found in the dropped selection.")
            return

        # Auto-configure output directory to <folder>/Denoised_DNG
        if target_folder:
            auto_out = os.path.join(target_folder, "Denoised_DNG")
            self.output_dir = auto_out
            self.out_entry.delete(0, "end")
            self.out_entry.insert(0, auto_out)

        added = 0
        for f in found_files:
            if any(item["path"] == f for item in self.files_queue):
                continue
            iso = self._extract_iso(f)
            self.files_queue.append({
                "path": f,
                "name": os.path.basename(f),
                "iso": iso,
                "status": "Ready",
                "time": "--",
                "out_dng": "",
            })
            added += 1

        self._refresh_queue_ui()
        self.status_lbl.configure(text=f"Loaded {len(self.files_queue)} RAW files into queue.")

        if self.selected_idx is None and len(self.files_queue) > 0:
            self._select_image(0)

        # If Auto-Start is checked, launch denoising immediately!
        if self.autostart_var.get() and added > 0 and not self.is_processing:
            self.after(300, self._on_start_processing)

    def _extract_iso(self, path: str) -> int:
        try:
            ex = Image.open(path).getexif()
            return int(ex.get_ifd(0x8769).get(0x8827, 800))
        except Exception:
            return 800

    def _on_clear_queue(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "Cannot clear queue while processing.")
            return
        self.files_queue.clear()
        self.selected_idx = None
        self._refresh_queue_ui()
        self.progress_bar.set(0.0)
        self.status_lbl.configure(text="Queue cleared. Drop a folder to start.")
        self.preview_canvas.configure(image=None, text="Drop a folder or select an image to inspect the 100% Before/After preview")

    def _on_browse_output(self):
        chosen = filedialog.askdirectory(title="Select Output Directory", initialdir=self.output_dir)
        if chosen:
            self.output_dir = chosen
            self.out_entry.delete(0, "end")
            self.out_entry.insert(0, chosen)

    def _open_output_folder(self):
        out_dir = self.out_entry.get().strip() or self.output_dir
        if os.path.exists(out_dir):
            subprocess.Popen(f'explorer "{os.path.abspath(out_dir)}"')
        else:
            messagebox.showinfo("Folder Not Found", "Output folder has not been created yet.")

    def _refresh_queue_ui(self):
        for widget in self.queue_frame.winfo_children():
            widget.destroy()

        h0 = ctk.CTkLabel(self.queue_frame, text="File Name", font=ctk.CTkFont(size=12, weight="bold"), text_color="#a1a1aa", anchor="w")
        h0.grid(row=0, column=0, sticky="ew", padx=6, pady=4)
        h1 = ctk.CTkLabel(self.queue_frame, text="ISO", font=ctk.CTkFont(size=12, weight="bold"), text_color="#a1a1aa", anchor="center")
        h1.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        h2 = ctk.CTkLabel(self.queue_frame, text="Status", font=ctk.CTkFont(size=12, weight="bold"), text_color="#a1a1aa", anchor="e")
        h2.grid(row=0, column=2, sticky="ew", padx=6, pady=4)

        for i, item in enumerate(self.files_queue):
            is_sel = (i == self.selected_idx)
            row_bg = "#27272a" if is_sel else ("#18181b" if i % 2 == 0 else "#09090b")

            row_frame = ctk.CTkFrame(self.queue_frame, fg_color=row_bg, corner_radius=6)
            row_frame.grid(row=i + 1, column=0, columnspan=3, sticky="ew", pady=2)
            row_frame.grid_columnconfigure(0, weight=5)
            row_frame.grid_columnconfigure(1, weight=2)
            row_frame.grid_columnconfigure(2, weight=3)

            name_lbl = ctk.CTkLabel(row_frame, text=item["name"], font=ctk.CTkFont(size=12), anchor="w")
            name_lbl.grid(row=0, column=0, sticky="ew", padx=8, pady=6)

            iso_lbl = ctk.CTkLabel(row_frame, text=f"ISO {item['iso']}", font=ctk.CTkFont(size=11), text_color="#94a3b8")
            iso_lbl.grid(row=0, column=1, sticky="ew", padx=4, pady=6)

            st_color = "#10b981" if "Done" in item["status"] else ("#f59e0b" if "Denoising" in item["status"] else "#a1a1aa")
            status_text = f"{item['status']} ({item['time']})" if item["time"] != "--" else item["status"]
            status_lbl = ctk.CTkLabel(row_frame, text=status_text, font=ctk.CTkFont(size=11, weight="bold"), text_color=st_color, anchor="e")
            status_lbl.grid(row=0, column=2, sticky="ew", padx=8, pady=6)

            for w in [row_frame, name_lbl, iso_lbl, status_lbl]:
                w.bind("<Button-1>", lambda e, idx=i: self._select_image(idx))

    def _select_image(self, idx: int):
        if idx >= len(self.files_queue):
            return
        self.selected_idx = idx
        self._refresh_queue_ui()
        self._generate_preview_for_selected()

    def _generate_preview_for_selected(self):
        if self.selected_idx is None or self.selected_idx >= len(self.files_queue):
            return

        item = self.files_queue[self.selected_idx]
        self.preview_canvas.configure(text=f"Rendering 100% Loupe Preview for {item['name']}...")

        def worker():
            try:
                with rawpy.imread(item["path"]) as r:
                    raw = r.raw_image
                    top = r.sizes.top_margin
                    left = r.sizes.left_margin
                    h = (r.sizes.height // 2) * 2
                    w = (r.sizes.width // 2) * 2
                    active = raw[top:top + h, left:left + w].astype(np.float32)
                    black = float(r.black_level_per_channel[0])
                    white = float(r.white_level)

                norm = np.clip((active - black) / (white - black), 0.0, 1.0)
                packed = pack_bayer(norm)
                _, ph, pw = packed.shape

                cy, cx = ph // 2, pw // 2
                cs = 256
                patch = packed[:, cy - cs // 2:cy + cs // 2, cx - cs // 2:cx + cs // 2]

                r_b = patch[0]
                g_b = (patch[1] + patch[2]) * 0.5
                b_b = patch[3]
                before_rgb = np.stack([r_b, g_b, b_b], axis=-1)
                before_rgb = np.power(np.clip(before_rgb * 1.5, 0.0, 1.0), 1 / 2.2)

                if self.model is not None:
                    iso_cond = np.full((1, cs, cs), np.log2(max(item["iso"], 100) / 100.0) / 10.0, dtype=np.float32)
                    inp_t = torch.from_numpy(np.concatenate([patch, iso_cond], axis=0)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                            pred_t = self.model(inp_t).squeeze(0).cpu().float().numpy()
                    pred_t = np.clip(pred_t, 0.0, 1.0)
                    r_a = pred_t[0]
                    g_a = (pred_t[1] + pred_t[2]) * 0.5
                    b_a = pred_t[3]
                    after_rgb = np.stack([r_a, g_a, b_a], axis=-1)
                    after_rgb = np.power(np.clip(after_rgb * 1.5, 0.0, 1.0), 1 / 2.2)
                else:
                    after_rgb = before_rgb

                split_col = cs // 2
                split_img = np.zeros_like(before_rgb)
                split_img[:, :split_col] = before_rgb[:, :split_col]
                split_img[:, split_col:] = after_rgb[:, split_col:]
                split_img[:, split_col - 1:split_col + 1] = 1.0

                pil_img = Image.fromarray((split_img * 255.0).astype(np.uint8))
                display_img = pil_img.resize((500, 500), Image.Resampling.NEAREST)
                ctk_img = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=(500, 500))

                self.after(0, lambda: self.preview_canvas.configure(image=ctk_img, text=""))
            except Exception as e:
                self.after(0, lambda: self.preview_canvas.configure(text=f"Preview error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_start_processing(self):
        if self.is_processing:
            return
        if not self.files_queue:
            messagebox.showinfo("Queue Empty", "Please drop a folder or add some Canon .CR2 RAW files first.")
            return
        if self.model is None:
            messagebox.showwarning("Model Loading", "Please wait for the model to finish loading.")
            return

        self.is_processing = True
        self.process_btn.configure(state="disabled", text="PROCESSING BATCH (GPU ACTIVE)...", fg_color="#3f3f46")
        self.output_dir = self.out_entry.get().strip() or os.path.abspath("data/denoised_exports")
        os.makedirs(self.output_dir, exist_ok=True)

        def batch_worker():
            total = len(self.files_queue)
            for i, item in enumerate(self.files_queue):
                if item["status"] == "Done":
                    continue

                item["status"] = "Denoising..."
                self.after(0, self._refresh_queue_ui)
                self.after(0, lambda idx=i: self.status_lbl.configure(text=f"Denoising [{idx + 1}/{total}]: {item['name']} on RTX 5060..."))

                out_dng_name = f"denoised_{os.path.splitext(item['name'])[0]}.dng"
                out_dng_path = os.path.join(self.output_dir, out_dng_name)

                t0 = time.time()
                try:
                    self._denoise_file(item["path"], out_dng_path, item["iso"])
                    elapsed = time.time() - t0
                    item["status"] = "Done"
                    item["time"] = f"{elapsed:.2f}s"
                    item["out_dng"] = out_dng_path
                except Exception as e:
                    item["status"] = "Error"
                    item["time"] = "Fail"
                    print(f"Error on {item['path']}: {e}")

                progress = (i + 1) / total
                self.after(0, lambda p=progress: self.progress_bar.set(p))
                self.after(0, self._refresh_queue_ui)

            self.after(0, self._on_processing_complete)

        threading.Thread(target=batch_worker, daemon=True).start()

    def _denoise_file(self, cr2_path: str, out_dng_path: str, iso: int):
        with rawpy.imread(cr2_path) as r:
            raw = r.raw_image
            top = r.sizes.top_margin
            left = r.sizes.left_margin
            h = (r.sizes.height // 2) * 2
            w = (r.sizes.width // 2) * 2
            active = raw[top:top + h, left:left + w].astype(np.float32)
            black = float(r.black_level_per_channel[0])
            white = float(r.white_level)
            daylight_wb = r.daylight_whitebalance
            wb_neutral = (1.0 / daylight_wb[0], 1.0 / daylight_wb[1], 1.0 / daylight_wb[2])

        active_range = white - black
        norm = np.clip((active - black) / active_range, 0.0, 1.0)
        packed = pack_bayer(norm)
        c, ph, pw = packed.shape

        iso_cond = np.full((1, ph, pw), np.log2(max(iso, 100) / 100.0) / 10.0, dtype=np.float32)
        inp_hwc = np.transpose(np.concatenate([packed, iso_cond], axis=0), (1, 2, 0))

        def process_tile_fn(tile_hwc: np.ndarray) -> np.ndarray:
            t_in = torch.from_numpy(np.transpose(tile_hwc, (2, 0, 1))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    pred = self.model(t_in)
            pred_np = pred.squeeze(0).cpu().float().numpy()
            return np.transpose(pred_np, (1, 2, 0))

        denoised_hwc = process_tiled(inp_hwc, process_tile_fn, tile=512, overlap=64)
        denoised_packed = np.clip(np.transpose(denoised_hwc, (2, 0, 1)), 0.0, 1.0)

        import scipy.ndimage
        r_ch = scipy.ndimage.zoom(denoised_packed[0], 2.0, order=1)
        g_ch = scipy.ndimage.zoom((denoised_packed[1] + denoised_packed[2]) * 0.5, 2.0, order=1)
        b_ch = scipy.ndimage.zoom(denoised_packed[3], 2.0, order=1)
        rgb_linear = np.stack([r_ch, g_ch, b_ch], axis=-1)

        write_linear_dng(out_dng_path, rgb_linear, camera_model="Canon EOS 6D", as_shot_neutral=wb_neutral)

    def _on_processing_complete(self):
        self.is_processing = False
        self.process_btn.configure(state="normal", text="START BATCH DENOISE", fg_color="#10b981")
        self.status_lbl.configure(text=f"Batch complete! Exported to: {self.output_dir}")
        # Open output folder in Explorer automatically
        self._open_output_folder()


def main():
    app = RawDenoiseApp()
    app.mainloop()


if __name__ == "__main__":
    main()
