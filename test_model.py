from ultralytics import YOLO
import cv2

if __name__ == "__main__":
    # Load your trained model
    model = YOLO(r"C:\Users\shaqi\runs\detect\train7\weights\best.pt")  # update path accordingly

    # For image testing:
    # results = model.predict(source="test_image.jpg", conf=0.25, show=True)

    # For video testing:
    video_path = r"C:\Users\shaqi\Downloads\auto\auto\207537_small.mp4"  # change to your video path
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on current frame
        results = model.predict(source=frame, conf=0.75, device=0, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Model Test", annotated_frame)

        # Press ESC key to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()