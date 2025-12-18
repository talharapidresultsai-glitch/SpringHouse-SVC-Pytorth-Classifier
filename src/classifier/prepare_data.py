#!/usr/bin/env python
import json
from urllib.parse import urlparse
from pathlib import Path
import boto3
TRAINING_INPUT="/opt/ml/input/data/training/"

splits = ["test", "train", "val"]

s3 = boto3.client('s3')

for split in splits:
    # Make destination directories
    image_destination = TRAINING_INPUT + 'images/' + split
    Path(image_destination).mkdir(parents=True, exist_ok=True)
    label_destination = TRAINING_INPUT + 'labels/' + split
    Path(label_destination).mkdir(parents=True, exist_ok=True)

    # Read the manifest
    input_filename = TRAINING_INPUT +  split + "_annotations.json"
    file = open(input_filename)
    images = json.load(file)
    file.close()

    for image in images:
        # for the classifier it only makes sense to train on images that have a label
        # otherwise we can't crop which is an expected part of the process
        if len(image['image_annotations']) == 0:
            continue
        # Download image 
        id = image['id']
        s3_path = image['s3_path']
        uri = urlparse(s3_path)
        bucket = uri.netloc 
        key = uri.path.lstrip('/')

        s3.download_file(bucket, key, image_destination + '/' + str(id) + '.jpg')

        # Create annotations
        def annotation_to_yolo(annotation):
            image_width = annotation['source_width'] or 1280
            image_height = annotation['source_height'] or 1024
            b_center_x = (annotation['left'] + annotation['width'] / 2) / image_width
            b_center_y = (annotation['top'] + annotation['height'] / 2) / image_height
            b_width = annotation['width'] / image_width
            b_height = annotation['height'] / image_height
            class_id = annotation['image_class_id']
            yolo_annotation = "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height)
            return yolo_annotation

        annotation_output = "\n".join(map(annotation_to_yolo, image['image_annotations']))

        with open(label_destination + '/' + str(id) + '.txt', "w") as annotation_file:
            annotation_file.write(annotation_output)
