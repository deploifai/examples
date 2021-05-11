import os
import functools
import operator
import base64
import numpy as np
import cv2

def predict(img):
  weights = os.path.sep.join([os.path.dirname(os.path.realpath(__file__)), "yolov4.weights"])
  config = os.path.sep.join([os.path.dirname(os.path.realpath(__file__)), "yolov4.cfg"])
  
  confidence_threshold = 0.3
  nms_threshold = 0.3

  net = cv2.dnn.readNetFromDarknet(config, weights)
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
  unconnected = functools.reduce(operator.iconcat, net.getUnconnectedOutLayers(), [])
  ln = net.getLayerNames()
  ln = [ln[i-1] for i in unconnected]
  np_img = np.fromstring(base64.b64decode(img[0]), dtype=np.uint8)
  img = cv2.imdecode(np_img, flags=cv2.IMREAD_COLOR)
  (H, W) = img.shape[:2]
  blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (512, 512), swapRB=True, crop=False)
  net.setInput(blob)
  layer_outputs = net.forward(ln)

  boxes = []
  confidences = []
  class_ids = []

  for output in layer_outputs:
    for detection in output:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]

      if confidence > confidence_threshold:
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")

        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        # Adding int(width) and int(height) is really important for some reason.
        # Removing it gives an error in NMSBoxes() call
        # Shall figure out soon and write a justification here.
        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        class_ids.append(class_id)

  indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
  if len(indexes) > 0:
    indexes = indexes.flatten()
    boxes = list(map(lambda idx: boxes[idx], indexes))
    confidences = list(map(lambda idx: confidences[idx], indexes))
    class_ids = list(map(lambda idx: class_ids[idx], indexes))
  return [[boxes, confidences, class_ids], 3]


if __name__ == "__main__":
  img = cv2.imread("animals.jpg")
  retval, buffer = cv2.imencode('.jpg', img)
  base64image = base64.b64encode(buffer).decode('utf-8')
  res = predict([base64image])
  print(res)
  detector_result = list(zip(res[0][0], res[0][1], res[0][2]))
  for box in detector_result:
    bbox = box[0]
    label = str(box[2])
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    cv2.rectangle(img=img,
                  pt1=(x, y),
                  pt2=(x + w, y + h),
                  color=(36, 255, 12),
                  thickness=2)
    cv2.putText(img=img,
                text=label,
                org=(x, y - 30),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7,
                color=(36, 255, 12),
                thickness=2)
  cv2.imshow("img", img)
  cv2.waitKey(0)