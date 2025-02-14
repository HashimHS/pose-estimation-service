import math
import grpc
import pipeline_pb2
import pipeline_pb2_grpc
from PIL import Image
from io import BytesIO
import numpy as np
import supervision as sv
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from concurrent import futures
import logging
import threading
import os
import torch

class Sam_Model:
    def __init__(self):
         # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # init sam image predictor and video predictor model
        SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
        SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        # GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        # GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading SAM2 model...")
        sam2_image_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_image_model)
        # self.mask_dict = MaskDictionaryModel(promote_type = "mask", mask_name = f"mask_{image_base_name}.npy")

        # init grounding dino model from huggingface
        print("Loading Grounding DINO model...")
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        print("device", self.device)
                
    def init_predict(self, rgb, prompt, box_threshold=0.5):
        # run Grounding DINO on the image
        inputs = self.processor(images=rgb, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=0.3,
            target_sizes=[rgb.size[::-1]]
        )
        
        input_boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        objects = results[0]["labels"]

        # prompt SAM image predictor to get the mask for the object
        self.ann_frame_idx = 0
        self.sam2_predictor.set_image(np.array(rgb.convert("RGB")))

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        ids = list(range(1, len(objects)+1))

        return masks, scores, objects, ids
                    
    def track(self, rgb, masks, ids):
        results = {}
        
        self.sam2_predictor.set_image(np.array(rgb.convert("RGB")))
        for (id, mask) in zip(ids, masks):
            try:
                mask, score, logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    mask_input=mask,
                    multimask_output=False,
                )
                results[id] = (mask, score)
            except:
                continue

        return results

class SegTracking_Service(pipeline_pb2_grpc.ImageModelPipelineServicer):
    def __init__(self, api_keys):

        print("Loading model...")
        self.model = Sam_Model()
        self.api_keys = api_keys
        self.lock = threading.Lock()
        print("Model loaded, waiting for requests...")
        pass

    def Ping(self, request: pipeline_pb2.PingRequest, context)->pipeline_pb2.PingReply:
        return pipeline_pb2.PingReply(seq=request.seq)

    def SegTracking(self, request_iterator, context):
        if request.api_key not in self.api_keys:
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details("Invalid api key")
            return pipeline_pb2.PoseDetectionReply()
        ids = []
        no_of_objects = 0

        for request in request_iterator:
            if len(ids) == 0: #or len(ids) < no_of_objects/2:
                masks, scores, phrases, ids = self.model.init_predict(self, Image.open(BytesIO(request.rgb)), request.prompt, request.box_threshold)
                label_dict = {id: phrase for id, phrase in zip(ids, phrases)}  
                no_of_objects = np.max(no_of_objects, len(ids))
                print("Detected objects: ", phrases)

            with self.lock:
                rgb = Image.open(BytesIO(request.rgb))
                
                results = self.model.track(rgb, masks, ids)

                masks_pb = []
                phrases = []
                for id in results.keys():
                    mask, score = results[id]
                    cpu_mask = mask.cpu().numpy()
                    mask = pipeline_pb2.Mask(w = cpu_mask.shape[1], h=cpu_mask.shape[0], score=score, packedbits=np.packbits(cpu_mask.flatten()).tobytes())
                    masks_pb.append(mask)
                    phrases.append(label_dict[id])

                yield pipeline_pb2.SegTrackingReply(masks=masks_pb, label=phrases)

def serve():
    port = os.environ.get("GRPC_PORT", "50051")
    api_keys = os.environ.get("API_KEYS", "test")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = SegTracking_Service(api_keys=set(api_keys.split(",")))
    print("Starting server on port " + port)
    pipeline_pb2_grpc.add_ImageModelPipelineServicer_to_server(service, server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    serve()