import cv2
from inference import ThermalInferenceEngine

engine = ThermalInferenceEngine(
    model_path="checkpoints/best_microghost_thermal_v1.pth",
    override_num_anchors=2   # v1 checkpoint was trained with 2 anchors
)

img_rgb     = cv2.cvtColor(cv2.imread("./rgb/190003.jpg"),     cv2.COLOR_BGR2RGB)
img_thermal = cv2.imread("./thermal/190003.jpg", cv2.IMREAD_GRAYSCALE)

detections = engine.detect(img_rgb, img_thermal)

if not detections:
    print("No intrusion detected.")
else:
    for i, det in enumerate(detections):
        print(f"[{i+1}] {det['class']} | conf={det['combined_conf']:.2f} "
              f"| temp={det.get('temp_c', 0)}°C | bbox={det['bbox']}")