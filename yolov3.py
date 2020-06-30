import cv2
import numpy as np

#initialize confidence threshold and non-maxima threshold
CONF_THRESH = 0.3
NMS_THRESH = 0.3

#load coco class labels, model config and model weights
label_path = 'coco.names'
LABELS = []

with open(label_path, 'rt') as f:
    LABELS = f.read().strip().split("\n")

model_Config = 'yolov3.cfg'
model_Weights = 'yolov3.weights'

#initialize colors that are used for different detection
COLORS = np.random.randint(0,255, size=(len(LABELS), 3), dtype="uint8")

#Load detector and obtain the output layers
net = cv2.dnn.readNetFromDarknet(model_Config, model_Weights)
ln = net.getLayerNames()
output_Layers = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

#start video capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    H,W = frame.shape[:2]

    #a blob is required as input for yolo
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    final_Output = net.forward(output_Layers)

    #initialize our bounding boxes, classIds, and confidence
    bboxes = []
    class_IDs = []
    confidence = []

    #loop over the three final outputs stages in yolov3 architecture
    for output in final_Output:
        #loop the detections
        for detection in output:
            #first 5 elements are the x,y,w,h and conf, the remaining ones are detections
            scores = detection[5:]
            #collect the position of the highest rank probability
            class_ID = np.argmax(scores)
            #obtain the confidence level
            conf = scores[class_ID]

            #filter weak predicitons
            if conf > CONF_THRESH:
                #obtain bbox
                width, height = int(detection[2]*W), int(detection[3]*H)
                centreX, centreY = int(detection[0]*W - width/2), int(detection[1]*H - height/2)
                bboxes.append([centreX, centreY, width, height])
                class_IDs.append(class_ID)
                confidence.append(float(conf))
    
    #using nms to suppress weak bboxes
    indices = cv2.dnn.NMSBoxes(bboxes, confidence, CONF_THRESH, NMS_THRESH)

    #ensure that there is at least one detection
    if len(indices) > 0:
        #loop over the indices
        for i in indices.flatten():
            #sort bounding box values to respective x,y,w,h
            (x,y) = (bboxes[i][0], bboxes[i][1])
            (w,h) = (bboxes[i][2], bboxes[i][3])

            #for variety in color scheme
            COLOR = [int(grab) for grab in COLORS[class_IDs[i]]]
            cv2.rectangle(frame, (x,y), (x+w, y+h), COLOR, 3)
            text = "{}: {:.1f}%".format(LABELS[class_IDs[i]], confidence[i]*100)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2)

    cv2.imshow("Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows










