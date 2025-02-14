Prompted Object Tracking Service
===============

This repository contains the code for a prompt based image segmentation and tracking pipeline.

The object detection and segmentation is done using Grounded SAM2.
 * https://github.com/facebookresearch/sam2

<!-- For performance and consitency the model files and other downloaded models is placed in cache.tar.gz. This is done automatically during the docker image building process. -->

This starts a GRPC based service that remotely expects image files to be sent and it will return the predicted results

# Usage
1- Build the Image
```bash
docker build -t object_tracker:latest .
```

2- Create and deploy the container
```bash
docker run --gpus all --net=host -d --name object_tracker object_tracker:latest
```

3- The container can be started/stopped on demand using
```bash
docker start object_tracker
```
or
```bash
docker stop object_tracker
```

# API Implementation
Copy client.py, pipeline_pb2_grpc.py, pipeline_pb2.py and pipeline_pb2.pyi into your client side code.
```python
    from client import MLDetector

    detector = MLDetector("localhost:50051")  # change localhost with ip_address if running the container on separate machine

    # For segmentation
    masks, boxes, scores, labels = detector.detect(rgb_image, text_prompt)

    # For pose-estimation
    masks, scores, labels = detector.detect_pose(
        rgb_image,
        text_prompt,
        Confidence_Threshold)
```