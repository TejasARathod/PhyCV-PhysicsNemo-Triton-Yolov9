import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.input_size = (640, 640)  # input image size

    def execute(self, requests):
        responses = []
        for request in requests:
            num_dets = pb_utils.get_input_tensor_by_name(request, "num_dets").as_numpy()[0]
            det_boxes = pb_utils.get_input_tensor_by_name(request, "det_boxes").as_numpy()
            det_scores = pb_utils.get_input_tensor_by_name(request, "det_scores").as_numpy()
            det_classes = pb_utils.get_input_tensor_by_name(request, "det_classes").as_numpy()

            print("Received num_dets:", num_dets)
            print("det_scores:", det_scores[:int(num_dets)])
            print("det_boxes:", det_boxes[:int(num_dets)])

            coords = []
            for i in range(min(int(num_dets), 100)):
                box = det_boxes[i]
                score = det_scores[i]

                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2 
                cy = (y1 + y2) / 2 
                coords.append([cx, cy])
                print(f"Detection {i}: score={score:.3f}, box={box}")
                print(f" --> Accepted coord: ({cx}, {cy})")

                if len(coords) == 4:
                    break

            if len(coords) == 0:
                print("No detections found.")

            while len(coords) < 4:
                coords.append([0.0, 0.0])  # pad with zeros

            coords_np = np.array(coords, dtype=np.float32)
            print("Final output coords:", coords_np)

            output_tensor = pb_utils.Tensor("onnx::Gemm_0", coords_np)
            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses

