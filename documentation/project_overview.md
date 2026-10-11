##### **PARKING LOT AVAILABILITY DETECTION AND MONITORING SYSTEM**

##### **Project Documentation**





###### **PROJECT OVERVIEW**



This system detects and monitors parking slot availability in real time using a pre-recorded CCTV video feed. It uses a YOLO object detection model to identify cars within user-defined parking zones and visually reports how many slots are occupied versus available on each frame.







###### **FILE STRUCTURE**



parking\_detection\_project/

&#x09;- frame\_extractor.py      Extract a reference frame from the input video.

&#x09;- zone\_selector.py        Launch the Tkinter GUI to draw parking zone polygons.

&#x09;- parking\_core.py         Core module: ParkingPtsSelection + ParkingManagement.

&#x09;- detector.py             Main detection script. Reads video, writes output.mp4.

&#x09;- parking1.mp4            Input: sample parking lot CCTV footage.

&#x09;- output.mp4              Output: annotated video (created by detector.py).

&#x09;- bounding\_boxes.json     Zone definitions. Included for parking1.mp4.

&#x09;- img\_0.jpg               Reference frame. Included for parking1.mp4.

&#x09;- User\_Guide.txt          Step-by-step instructions for setup and usage.

&#x09;- README.txt              This file.





Note on input/output files:



For the provided sample video (parking1.mp4), the required bounding\_boxes.json and img\_0.jpg files are already included in the project folder. You can run detector.py immediately without performing Steps 1 and 2.



If you wish to use this system with a DIFFERENT input video:

&#x20; - You MUST delete or replace the existing bounding\_boxes.json and img\_0.jpg

&#x20; - You MUST run Step 1 (frame\_extractor.py) and Step 2 (zone\_selector.py) to generate new files specific to your new video.

&#x20; - The system cannot work without these two files matching your video.







###### **HOW IT WORKS**



1\. Frame Extraction (frame\_extractor.py)

&#x20;  OpenCV reads the video and saves a resized JPEG frame for use as a reference image during zone definition.



2\. Zone Definition (zone\_selector.py + ParkingPtsSelection in parking\_core.py)

&#x20;  A Tkinter canvas displays the reference image. The user clicks 4 points per parking slot. Coordinates are scaled back from canvas resolution to original image resolution before being saved to bounding\_boxes.json.



3\. Detection Pipeline (detector.py + ParkingManagement in parking\_core.py)

&#x20;  - Each video frame is fed to Ultralytics YOLO11s (COCO-pretrained).

&#x20;  - Only class 2 (car) detections are retained.

&#x20;  - For each detected car, the bounding-box centroid is computed.

&#x20;  - cv2.pointPolygonTest checks whether the centroid falls inside each polygon.

&#x20;  - Zones are coloured green (available) or red (occupied) accordingly.

&#x20;  - An occupancy HUD is overlaid and annotated frames are written to output.mp4.







###### **MODEL AND DATASET**



Detection model : YOLO11s (small variant) — good balance of speed and accuracy for overhead parking lot footage at 1080x600 resolution.

Dataset : COCO (80 classes, pre-trained). Only class 2 (car) is used.







###### **INPUT/OUTPUT**



Input

&#x09;- parking1.mp4 : Overhead CCTV footage of a parking lot. 

&#x09;- bounding\_boxes.json : Zone polygon definitions drawn in zone\_selector.py. (Included for parking1.mp4)



Output

&#x09;- output.mp4 : Annotated video with the following overlays:

&#x09;	- Green polygon  = available parking slot

&#x09;	- Red polygon    = occupied parking slot

&#x09;	- Pink dot       = detected vehicle centroid

&#x09;	- White text     = vehicle class label ("car")

&#x09;	- Top-left HUD   = live Occupancy / Available counts



