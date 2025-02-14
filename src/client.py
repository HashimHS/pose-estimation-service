from typing import Optional
import grpc
from google.protobuf.internal import containers as _containers
import pipeline_pb2, pipeline_pb2_grpc
import numpy as np
import cv2 as cv

class MLDetector:
    def __init__(self, endpoint, api_key="test", stream=None):
        self.endpoint = endpoint
        self.api_key = api_key
        self.stream = stream

    def check_connection(self):
        with grpc.insecure_channel(self.endpoint) as channel:
            stub = pipeline_pb2_grpc.ImageModelPipelineStub(channel)
            response = stub.Ping(pipeline_pb2.PingRequest(seq=1))
            if response.seq != 1:
                raise Exception("Cannot connect to the detection server")

    def track(self, rgb_stream, prompt="", box_threshold=0.3):
        """
        Track object poses in a live video stream
        
        parameters:
            rgb_image: An OpenCV image in RGB color space.
            prompt: What to search for (may be None, then default value is used)
            box_threshold: Detection threshold for bounding box confidence
            
        returns:
            masks: list of masks of each detected object
            scores: list of detection confidence scores of each detected object
            labels: list of string labels extracted from prompt corresponding to each detected object
        """

        # self.predictions = self.detect_raw_pose(prompt, pipeline_pb2.Image(image_format="jpg", image_data=bytes(rgb_rawdata)), pipeline_pb2.Image(image_format="png", image_data=bytes(depth_rawdata)), intrinsics, box_threshold)
        channel = grpc.insecure_channel(self.endpoint)
        stub = pipeline_pb2_grpc.ImageModelPipelineStub(channel)
        for response in stub.PoseTracking(self.gen_pose_tracking_request(rgb_stream, prompt, box_threshold)):
            
            masks = []
            scores = []
            for i in range(len(self.predictions.masks)):
                    mask = self.predictions.masks[i]
                    masks.append(np.unpackbits(np.frombuffer(mask.packedbits, dtype=np.uint8), count=mask.w*mask.h).reshape(mask.h, mask.w))
                    scores.append(mask.score)
                                        
            yield masks, scores, self.predictions.label

    def get_image(self):
        rgb_image, depth_image, _  = self.stream.get_image()
        return rgb_image