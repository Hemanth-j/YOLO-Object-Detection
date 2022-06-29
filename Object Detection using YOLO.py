import numpy as np
import time
import cv2
import os
import firebase_admin
from firebase_admin import credentials,db
cred = credentials.Certificate("C:\\Users\\HEMANTH\\OneDrive\\Documents\\face.json")
firebase_admin.initialize_app(cred,{"databaseURL":"https://face-12b8c-default-rtdb.firebaseio.com/"})
store=db.reference("/")
predict=0.5
minimum=0.5
labelsPath="D:\\yolo-coco\\coco.names"
LABELS=open(labelsPath).read().strip().split("\n")
COLORS=np.random.randint(50,255,size=(len(LABELS),3),dtype="uint8")
weightspath="D:\\yolo-coco\\yolov3.weights"
configPath="D:\\yolo-coco\\yolov3.cfg"
net=cv2.dnn.readNetFromDarknet(configPath,weightspath)
#image=cv2.imread("C:\\Users\\HEMANTH\\Downloads\\face recongnition\\1_TBFZedqausle97QkD92d0A.jpeg")
print("streaming started")
video_capture=cv2.VideoCapture(0)
count=1
while True:
  ret, image = video_capture.read()
  (H,W)=image.shape[:2]
  ln=net.getLayerNames()
  ln=net.getUnconnectedOutLayersNames()
  blob=cv2.dnn.blobFromImage(image,1/255.0,(416,416),swapRB=True,crop=False)
  net.setInput(blob)
  layerOutputs=net.forward(ln)
  boxes=[]
  confidences=[]
  classIDs=[]
  for output in layerOutputs:
    for detection in output:
        scores=detection[5:]
        classID=np.argmax(scores)
        confidence=scores[classID]
        if confidence > predict:
            box=detection[0:4]*np.array([W,H,W,H])
            (centerX,centerY,width,height)=box.astype("int")
            x=int(centerX-(width/2))
            y=int(centerY-(height/2))
            boxes.append([x,y,int(width),int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
  idxs=cv2.dnn.NMSBoxes(boxes,confidences,predict,minimum)
  if len(idxs)>0:
    for i in idxs.flatten():
        (x,y)=(boxes[i][0],boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color=[int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image,(x,y),(x+w,y+h),confidences[i])
        test="{}:{:.4f}".format(LABELS[classIDs[i]],confidences[i])
        store.update({"object "+str(count): test})
        cv2.putText(image,test,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        count=count+1
  image1=cv2.resize(image,(900,600))
  cv2.imshow("image",image1)
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break
video_capture.release()
cv2.destroyAllWindows()