##### **PARKING LOT AVAILABILITY DETECTION AND MONITORING SYSTEM**

##### **User Guide**







###### **STEP 0 — DOWNLOAD FILES AND INSTALL DEPENDENCIES**



1\. Download all project files from the provided source or repository.

&#x20;  

If you are using the SAME video that we are uploading along with this project, you will already have the following complete set of files in the folder:

&#x20;    	- frame\_extractor.py

&#x20;    	- zone\_selector.py

&#x20;    	- parking\_core.py

&#x20;    	- detector.py

&#x20;    	- parking1.mp4 (input video)

&#x20;    	- bounding\_boxes.json (pre-drawn zones for parking1.mp4) 

&#x20;    	- img\_0.jpg (pre-extracted reference frame for parking1.mp4) 

&#x20;    	- README.txt

&#x20;    	- User\_Guide.txt



IMPORTANT: 

For the provided sample video (parking1.mp4), the required bounding\_boxes.json and img\_0.jpg are already included. You can skip directly to STEP 3 to run the detector.



If you are using a DIFFERENT video (not the one we uploaded), you will only have the Python scripts (.py files). You will need to:

&#x20;    	- Delete or remove the existing bounding\_boxes.json and img\_0.jpg

&#x20;    	- Follow Steps 1 and 2 to generate new files for your video



2\. You should have one of the following operating systems to run the program: 

&#x09;- Windows 10/11 

&#x09;- macOS 12+

&#x09;- Ubuntu 20.04+



3. Make sure you have Python 3.9 or higher installed on your system. (Python 3.10 is recommended.)



4. Install all required Python packages using pip.



Open a terminal (Command Prompt, PowerShell, or Terminal) and run:

&#x09;pip install opencv-python numpy ultralytics Pillow



This will install:

&#x09;- opencv-python    : for video and image processing

&#x20;    	- numpy            : for numerical operations

&#x20;    	- ultralytics      : for YOLO object detection model

&#x20;    	- Pillow           : for image handling in the GUI



5. Verify the installation by running the following command in Python:



python -c "import cv2, numpy, ultralytics, PIL; print('All packages installed successfully!')"

If you see no errors, your setup is ready.



Note: On first run of detector.py, Ultralytics will automatically download the yolo11s.pt weights file (\~20 MB) to your project folder.



6\. All user-configurable paths and parameters are defined as named constants at the top of each script. If needed, make the required changes here.



frame\_extractor.py

&#x20; VIDEO\_PATH    Path to the input parking lot video

&#x20; OUTPUT\_PATH   Path template for saved frames

&#x20; MAX\_FRAMES    Number of frames to extract

&#x20; FRAME\_SKIP    Process every Nth frame



detector.py

&#x20; VIDEO\_INPUT     Path to the input parking lot video

&#x20; VIDEO\_OUTPUT    Path for the annotated output video

&#x20; MODEL\_WEIGHTS   YOLO weights file (default: yolo11s.pt)

&#x20; DETECT\_CLASSES  COCO class IDs to detect (default: \[2] = car)

&#x20; ZONES\_JSON      Path to bounding\_boxes.json

&#x20; FRAME\_WIDTH     Processing resolution width

&#x20; FRAME\_HEIGHT    Processing resolution height









###### **STEP 1 — Extract a reference frame (ONLY for a NEW input video)**



Run frame\_extractor.py to pull a single frame from the input video. This image will be used in the next step to draw parking zones.



&#x09;python frame\_extractor.py



Output : img\_0.jpg  (saved in the project directory)



Note: By default the script reads parking1.mp4. If your video has a different name or path, edit the VIDEO\_PATH variable at the top of frame\_extractor.py.



WHEN TO RUN THIS STEP:

&#x20; - The provided parking1.mp4 (the video we uploaded) already has img\_0.jpg

&#x20;   included. Skip this step.

&#x20; - Run this step ONLY if you are using a DIFFERENT input video.









###### **STEP 2 — Define parking zones (ONLY for a NEW input video)**



Run zone\_selector.py to launch the interactive zone-drawing tool.



&#x20; 	python zone\_selector.py



Inside the GUI:

&#x20; a. Click "Upload Image" and open img\_0.jpg (the reference frame from Step 1).

&#x20; b. Click exactly 4 corner points around the first parking slot. A blue quadrilateral appears when the 4th point is clicked.

&#x20; c. Repeat for every parking slot visible in the image.

&#x20; d. Use "Remove Last BBox" to undo the most recent box if you made a mistake.

&#x20; e. Click "Save" when all slots are marked.



Output : bounding\_boxes.json  (saved in the project directory)



Important: The 4 points must follow a consistent order — either all clockwise or all counter-clockwise — to ensure correct polygon rendering.



WHEN TO RUN THIS STEP:

&#x20; - The provided parking1.mp4 (the video we uploaded) already has bounding\_boxes.json included. Skip this step.

&#x20; - Run this step ONLY if you are using a DIFFERENT input video.









###### **STEP 3 — Run the occupancy detector**



Run detector.py to process the full video.



&#x20; 	python detector.py



What you will see:

&#x20; - A live preview window titled "Parking Occupancy Detection".

&#x20; - Each parking zone is outlined in GREEN (available) or RED (occupied).

&#x20; - A pink/magenta dot marks the centroid of each detected vehicle.

&#x20; - Top-left corner displays live "Occupancy: X" and "Available: Y" counts.

&#x20; - Press ESC in the preview window to stop early.



Output : output.mp4  (annotated video saved in the project directory)



Note: For the provided parking1.mp4 (the video we uploaded), the system will use the included bounding\_boxes.json and img\_0.jpg. No additional setup is required.

