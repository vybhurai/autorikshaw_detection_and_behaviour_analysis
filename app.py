if __name__ == "__main__":
    from ultralytics import YOLO
    import cv2
    import numpy as np
    from sort import Sort
    import math

    # Load YOLO model
    model = YOLO(r"C:\Users\shaqi\runs\detect\train7\weights\best.pt")

    # Load video
    cap = cv2.VideoCapture(r"C:\Users\shaqi\Downloads\auto\auto\207537_small.mp4")

    # Initialize SORT tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Store past locations for speed and direction calculation
    speed_history = {}

    def calculate_angle(p1, p2):
        """Calculate angle in degrees between two points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection: boxes, confidences
        results = model.predict(source=frame, conf=0.25, device=0, verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
        confidences = results[0].boxes.conf.cpu().numpy()  # confidence scores

        # Prepare detections for SORT tracker
        dets = np.hstack((detections, confidences.reshape(-1,1)))

        # Update SORT tracker
        tracks = tracker.update(dets)

        for track in tracks:
            x1, y1, x2, y2, track_id = track
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)

            if track_id not in speed_history:
                speed_history[track_id] = []

            speed_history[track_id].append((cx, cy))

            if len(speed_history[track_id]) > 10:
                # Calculate speed over last 10 frames
                (x0, y0) = speed_history[track_id][-10]
                dx = cx - x0
                dy = cy - y0
                dist = np.sqrt(dx*dx + dy*dy)

                # Calculate angle change to detect sudden turns
                angle_current = calculate_angle(speed_history[track_id][-2], speed_history[track_id][-1])
                angle_prev = calculate_angle(speed_history[track_id][-10], speed_history[track_id][-9])
                angle_diff = abs(angle_current - angle_prev)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff  # Normalize

                # Behavior classification
                if dist < 3:
                    behavior = "Stopped"
                elif angle_diff > 30 and dist > 20:
                    behavior = "Sudden Turn"
                elif dist > 20:
                    behavior = "Weaving"
                else:
                    behavior = "Moving"

                # Keep only last 20 points to save memory
                speed_history[track_id] = speed_history[track_id][-20:]
            else:
                behavior = "Moving"

            # Draw bounding box and behavior label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"ID:{int(track_id)} {behavior}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Show output
        cv2.imshow("Auto Rickshaw Behavior Analysis", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
            break

    cap.release()
    cv2.destroyAllWindows()
