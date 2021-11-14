import numpy as np
import cv2

def detect_people(frame, net, ln, MIN_CONF, NMS_THRESH, personIdx=0):

    (H, W) = frame.shape[:2]
    results = []

    # construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, centroids, and confidences
    boxes = []
    centroids = []
    confidences = []

    # loop for each layer outputs
    for output in layerOutputs:
        # loop for each detection
        for detection in output:
            # get class ID and confidence of current detected object
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personIdx and confidence > MIN_CONF:
                # scale the bounding box coordinates
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

				# update bounding box coordinates, centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # NMS_THRESH is to suppres weak, overlapping boudning boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# ensure at least one detection exists
    if len(idxs) > 0:
		# loop over the indexes
        for i in idxs.flatten():
			# extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

			# update results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)
            
	# return the list of results
    return results