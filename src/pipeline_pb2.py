# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: pipeline.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0epipeline.proto\x12\rimagepipeline\"\x1a\n\x0bPingRequest\x12\x0b\n\x03seq\x18\x01 \x01(\x05\"\x18\n\tPingReply\x12\x0b\n\x03seq\x18\x01 \x01(\x05\"?\n\x04Mask\x12\t\n\x01w\x18\x01 \x01(\x05\x12\t\n\x01h\x18\x02 \x01(\x05\x12\r\n\x05score\x18\x03 \x01(\x02\x12\x12\n\npackedbits\x18\x04 \x01(\x0c\"4\n\x06Region\x12\t\n\x01x\x18\x01 \x01(\x05\x12\t\n\x01y\x18\x02 \x01(\x05\x12\t\n\x01w\x18\x03 \x01(\x05\x12\t\n\x01h\x18\x04 \x01(\x05\"1\n\x05Image\x12\x14\n\x0cimage_format\x18\x01 \x01(\t\x12\x12\n\nimage_data\x18\x02 \x01(\x0c\"-\n\x04Pose\x12\x10\n\x08position\x18\x01 \x03(\x02\x12\x13\n\x0borientation\x18\x02 \x03(\x02\"d\n\x1cPromptObjectDetectionRequest\x12\x0f\n\x07\x61pi_key\x18\x01 \x01(\t\x12\x0e\n\x06prompt\x18\x02 \x01(\t\x12#\n\x05image\x18\x03 \x01(\x0b\x32\x14.imagepipeline.Image\"N\n\x16ObjectDetectionRequest\x12\x0f\n\x07\x61pi_key\x18\x01 \x01(\t\x12#\n\x05image\x18\x02 \x01(\x0b\x32\x14.imagepipeline.Image\"q\n\x14ObjectDetectionReply\x12\"\n\x05masks\x18\x01 \x03(\x0b\x32\x13.imagepipeline.Mask\x12&\n\x07regions\x18\x02 \x03(\x0b\x32\x15.imagepipeline.Region\x12\r\n\x05label\x18\x03 \x03(\t\"\xaa\x01\n\x14PoseDetectionRequest\x12\x0f\n\x07\x61pi_key\x18\x01 \x01(\t\x12\x0e\n\x06prompt\x18\x02 \x01(\t\x12!\n\x03rgb\x18\x03 \x01(\x0b\x32\x14.imagepipeline.Image\x12#\n\x05\x64\x65pth\x18\x04 \x01(\x0b\x32\x14.imagepipeline.Image\x12\x12\n\nintrinsics\x18\x05 \x03(\x02\x12\x15\n\rbox_threshold\x18\x06 \x01(\x02\"\x92\x01\n\x12PoseDetectionReply\x12\"\n\x05masks\x18\x01 \x03(\x0b\x32\x13.imagepipeline.Mask\x12&\n\x07regions\x18\x02 \x03(\x0b\x32\x15.imagepipeline.Region\x12\r\n\x05label\x18\x03 \x03(\t\x12!\n\x04pose\x18\x04 \x03(\x0b\x32\x13.imagepipeline.Pose2\xfd\x02\n\x12ImageModelPipeline\x12>\n\x04Ping\x12\x1a.imagepipeline.PingRequest\x1a\x18.imagepipeline.PingReply\"\x00\x12k\n\x15PromptObjectDetection\x12+.imagepipeline.PromptObjectDetectionRequest\x1a#.imagepipeline.ObjectDetectionReply\"\x00\x12_\n\x0fObjectDetection\x12%.imagepipeline.ObjectDetectionRequest\x1a#.imagepipeline.ObjectDetectionReply\"\x00\x12Y\n\rPoseDetection\x12#.imagepipeline.PoseDetectionRequest\x1a!.imagepipeline.PoseDetectionReply\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'pipeline_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_PINGREQUEST']._serialized_start=33
  _globals['_PINGREQUEST']._serialized_end=59
  _globals['_PINGREPLY']._serialized_start=61
  _globals['_PINGREPLY']._serialized_end=85
  _globals['_MASK']._serialized_start=87
  _globals['_MASK']._serialized_end=150
  _globals['_REGION']._serialized_start=152
  _globals['_REGION']._serialized_end=204
  _globals['_IMAGE']._serialized_start=206
  _globals['_IMAGE']._serialized_end=255
  _globals['_POSE']._serialized_start=257
  _globals['_POSE']._serialized_end=302
  _globals['_PROMPTOBJECTDETECTIONREQUEST']._serialized_start=304
  _globals['_PROMPTOBJECTDETECTIONREQUEST']._serialized_end=404
  _globals['_OBJECTDETECTIONREQUEST']._serialized_start=406
  _globals['_OBJECTDETECTIONREQUEST']._serialized_end=484
  _globals['_OBJECTDETECTIONREPLY']._serialized_start=486
  _globals['_OBJECTDETECTIONREPLY']._serialized_end=599
  _globals['_POSEDETECTIONREQUEST']._serialized_start=602
  _globals['_POSEDETECTIONREQUEST']._serialized_end=772
  _globals['_POSEDETECTIONREPLY']._serialized_start=775
  _globals['_POSEDETECTIONREPLY']._serialized_end=921
  _globals['_IMAGEMODELPIPELINE']._serialized_start=924
  _globals['_IMAGEMODELPIPELINE']._serialized_end=1305
# @@protoc_insertion_point(module_scope)
