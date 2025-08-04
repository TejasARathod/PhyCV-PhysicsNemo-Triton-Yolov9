import numpy as np
import cv2
import sys
import tritonclient.grpc as grpcclient
from utils.processing import preprocess, postprocess
from utils.render import render_box, get_text_size, render_filled_box, render_text, RAND_COLORS

TRITON_URL = "localhost:8001"
ENSEMBLE_MODEL = "ensemble"
MODEL_WIDTH = 640
MODEL_HEIGHT = 640
INPUT_VIDEO_PATH = "BB1.mp4"
OUTPUT_VIDEO_PATH = "output_video.mp4"
OUTPUT_FPS = 24.0

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
COLOR = (0, 255, 0)
THICKNESS = 2

if __name__ == "__main__":
    try:
        client = grpcclient.InferenceServerClient(url=TRITON_URL)
    except Exception as e:
        print("Failed to connect to Triton server:", e)
        sys.exit(1)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Failed to open video: {INPUT_VIDEO_PATH}")
        sys.exit(1)

    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]

        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, OUTPUT_FPS, (orig_w, orig_h))

        # Preprocess
        input_image_buffer = preprocess(frame, [MODEL_WIDTH, MODEL_HEIGHT])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)

        inputs = [grpcclient.InferInput("images", input_image_buffer.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_image_buffer)

        outputs = [
            grpcclient.InferRequestedOutput("num_dets"),
            grpcclient.InferRequestedOutput("det_boxes"),
            grpcclient.InferRequestedOutput("det_scores"),
            grpcclient.InferRequestedOutput("det_classes"),
            grpcclient.InferRequestedOutput("57")  # u, v, p
        ]

        results = client.infer(model_name=ENSEMBLE_MODEL, inputs=inputs, outputs=outputs)

        num_dets = results.as_numpy("num_dets")
        det_boxes = results.as_numpy("det_boxes")
        det_scores = results.as_numpy("det_scores")
        det_classes = results.as_numpy("det_classes")
        flow = results.as_numpy("57")[0]  # shape: (4, 3) — u, v, p

        detected_objects = postprocess(num_dets, det_boxes, det_scores, det_classes,
                                       orig_w, orig_h, [MODEL_WIDTH, MODEL_HEIGHT])
        print(f"Detected objects: {len(detected_objects)}")

        for i, box in enumerate(detected_objects[:4]):
            label = "Ball"
            conf = box.confidence
            x1, y1, x2, y2 = box.box()

            frame = render_box(frame, box.box(), color=tuple(RAND_COLORS[box.classID % 64].tolist()))
            label_text = f"{label}: {conf:.2f}"
            size = get_text_size(frame, label_text, normalised_scaling=0.6)
            frame = render_filled_box(frame, (x1 - 3, y1 - 3, x1 + size[0], y1 + size[1]), color=(220, 220, 220))
            frame = render_text(frame, label_text, (x1, y1), color=(30, 30, 30), normalised_scaling=0.5)

            # Extract and show (u, v, p)
            u, v, p = flow[i]
            right_x = orig_w - 200
            cv2.putText(frame, f"U: {u:.2f}", (right_x, 30 + i * 60), FONT, FONT_SCALE, COLOR, THICKNESS)
            cv2.putText(frame, f"V: {v:.2f}", (right_x, 50 + i * 60), FONT, FONT_SCALE, COLOR, THICKNESS)
            cv2.putText(frame, f"P: {p:.2f}", (right_x, 70 + i * 60), FONT, FONT_SCALE, COLOR, THICKNESS)

            print(f"Object {i+1}: ({x1}, {y1}) | u: {u:.2f}, v: {v:.2f}, p: {p:.2f}")

        out.write(frame)
        cv2.imshow("Detections", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

