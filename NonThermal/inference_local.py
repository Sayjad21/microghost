import cv2  # Added to handle window displaying and pausing
from ultralytics import YOLO

model = YOLO("best.pt")  # swap for thermal best.pt for the other one

# Added stream=True to process the webcam frame-by-frame in a loop
results = model.predict(
    source="../tests/190007.jpg",  # 0 = webcam, or put an image/video path
    imgsz=256,
    conf=0.3,      # was 0.4, lets weaker boxes through
    iou=0.7,    # was 0.7, stops NMS from merging two people
    max_det=50,     # default is 300, fine to keep
    device=0,       # RTX 3070
    show=True,
    #tream=True,
)


# Loop through each frame and pause until a key is pressed
for result in results:
    # Plot the bounding boxes onto the frame
    frame = result.plot()
    print(result.boxes)  # Print the detected boxes for debugging
    # Show the frame in a window
    cv2.imshow("YOLOv8 - Press any key to next frame", frame)

    # 0 means wait indefinitely until a key is pressed
    key = cv2.waitKey(0) & 0xFF

    # Optional safety: press 'q' or 'Esc' (27) to close the webcam entirely
    if key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()



# from ultralytics import YOLO
# model = YOLO("best.pt")

# for c in [0.25, 0.3, 0.35, 0.4]:
#     for i in [0.6, 0.7, 0.75]:
#         r = model.predict("../tests/190007.jpg", imgsz=256, conf=c, iou=i, device=0, verbose=False)
#         print(f"conf={c} iou={i} -> {len(r[0].boxes)} people")