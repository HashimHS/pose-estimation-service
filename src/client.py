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
            else:
                print("Connection established")

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

        channel = grpc.insecure_channel(self.endpoint)
        stub = pipeline_pb2_grpc.ImageModelPipelineStub(channel)

        for predictions in stub.SegTracking(self.gen_seg_tracking_request(rgb_stream, prompt, box_threshold)):
            
            masks = []
            scores = []
            for i in range(len(predictions.masks)):
                    mask = predictions.masks[i]
                    masks.append(np.unpackbits(np.frombuffer(mask.packedbits, dtype=np.uint8), count=mask.w*mask.h).reshape(mask.h, mask.w))
                    scores.append(mask.score)
                                        
            yield masks, scores, predictions.label

    def gen_seg_tracking_request(self, rgb_stream, prompt, box_threshold):
        while True:
            rgb_image = rgb_stream.get_image()
            rgb_image = cv.imencode('.jpg', rgb_image)[1].tobytes()
            yield pipeline_pb2.SegTrackingRequest(api_key=self.api_key, rgb=rgb_image, prompt=prompt, box_threshold=box_threshold)

    def get_image(self):
        rgb_image  = self.stream.get_image()
        return rgb_image