# camera.py
from chaquopy import asset_manager
from android.hardware import Camera
import threading

class CameraPreview:
    def __init__(self, camera_id):
        self.camera = Camera.open(camera_id)
        self.camera.setPreviewCallback(self.preview_callback)
        self.thread = threading.Thread(target=self.start_preview)
        self.thread.start()

    def preview_callback(self, data, camera):
        # 处理摄像头预览帧
        asset_manager().put("frame", data)

    def start_preview(self):
        # 启动摄像头预览
        self.camera.startPreview()

    def stop_preview(self):
        # 停止摄像头预览
        self.camera.stopPreview()
