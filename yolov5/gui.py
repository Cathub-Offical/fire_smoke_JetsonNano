import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import cv2
from detect import run  # 直接导入run函数

WEIGHTS_PATH = "E:/Flame_Smoke/yolov5_fire_smoke_detection/model/best.pt" #本地权重
CANVAS_W, CANVAS_H = 800, 600

class FireSmokeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("火焰与烟雾检测系统")
        self.root.geometry("1000x600")
        self.file_path = tk.StringVar()
        self.confidence_threshold = tk.DoubleVar(value=0.4)
        self.image = None
        self.running = False   
        self.video_cap = None       
        self.video_after_id = None  
        self.setup_ui()

    def setup_ui(self):
        control_frame = tk.Frame(self.root, width=300, bg="lightgray")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(control_frame, text="文件输入", bg="lightgray").pack(pady=10)
        tk.Button(control_frame, text="打开图像", command=self.open_image).pack(pady=5)
        tk.Button(control_frame, text="打开视频", command=self.open_video).pack(pady=5)
        tk.Entry(control_frame, textvariable=self.file_path, state="readonly", width=30).pack(pady=5)
        tk.Label(control_frame, text="置信度阈值", bg="lightgray").pack(pady=10)
        tk.Scale(control_frame, from_=0.1, to=0.9, resolution=0.1, orient=tk.HORIZONTAL, variable=self.confidence_threshold).pack(pady=5)
        tk.Button(control_frame, text="开始检测", command=self.start_detection).pack(pady=10)
        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            self._stop_video_playback()
            self.file_path.set(file_path)
            self._show_image_on_canvas(file_path)

    def open_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if file_path:
            self._stop_video_playback()
            self.file_path.set(file_path)
            self._play_video_on_canvas(file_path)

    def _show_image_on_canvas(self, file_path):
        try:
            image = Image.open(file_path).resize((CANVAS_W, CANVAS_H))
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图像: {e}")
            return
        self.image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    def _play_video_on_canvas(self, file_path):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开视频!")
            return
        self.video_cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        self.video_delay_ms = max(int(1000 / fps), 1)
        self._video_loop()

    def _video_loop(self):
        if self.video_cap is None:
            return
        ret, frame = self.video_cap.read()
        if not ret:                                          # 循环预览
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_cap.read()
            if not ret:
                self._stop_video_playback()
                return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame).resize((CANVAS_W, CANVAS_H))
        self.image = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        self.video_after_id = self.root.after(self.video_delay_ms, self._video_loop)

    def _stop_video_playback(self):
        if self.video_after_id is not None:
            try:
                self.root.after_cancel(self.video_after_id)
            except Exception:
                pass
            self.video_after_id = None
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None

    def _run_detect(self, file_path):
        output_dir = "inference/output"
        os.makedirs(output_dir, exist_ok=True)
        run(
            weights=WEIGHTS_PATH,
            source=file_path,
            imgsz=(640, 640),
            conf_thres=self.confidence_threshold.get(),
            iou_thres=0.5,
            device="cpu",
            save_txt=False,
            save_conf=False,
            nosave=False,
            project=output_dir,
            name="result",
            exist_ok=True,
        )
        return os.path.join(output_dir, "result", os.path.basename(file_path))

    def detect_image(self, file_path):
        try:
            result_img = self._run_detect(file_path)
        except Exception as e:
            self.running = False
            messagebox.showerror("错误", f"检测失败: {e}")
            return
        self.running = False
        if os.path.exists(result_img):
            self.root.after(0, self._show_image_on_canvas, result_img)
        else:
            messagebox.showerror("错误", "未找到检测结果图片！")

    def detect_video(self, file_path):
        try:
            result_video = self._run_detect(file_path)
        except Exception as e:
            self.running = False
            messagebox.showerror("错误", f"检测失败: {e}")
            return
        self.running = False
        if os.path.exists(result_video):
            self.root.after(0, self._restart_video_playback, result_video)
        else:
            messagebox.showerror("错误", "未找到检测结果视频！")

    def _restart_video_playback(self, path):
        self._stop_video_playback()
        self._play_video_on_canvas(path)

    def start_detection(self):
        if self.running:
            messagebox.showinfo("提示", "检测正在进行中,请稍候...")
            return
        if not self.file_path.get():
            messagebox.showwarning("警告", "请先选择文件！")
            return
        file = self.file_path.get()
        ext = os.path.splitext(file)[1].lower()
        self._stop_video_playback()
        self.running = True
        if ext in [".jpg", ".jpeg", ".png"]:
            threading.Thread(target=self.detect_image, args=(file,), daemon=True).start()
        elif ext in [".mp4", ".avi", ".mov"]:
            threading.Thread(target=self.detect_video, args=(file,), daemon=True).start()
        else:
            self.running = False
            messagebox.showerror("错误", "不支持的文件类型！")

if __name__ == "__main__":
    root = tk.Tk()
    app = FireSmokeDetectionApp(root)
    root.mainloop()