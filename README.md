# fire


## Prerequisites
- Download and extract the [YOLOv5](https://github.com/ultralytics/yolov5) repository.
- Install the required libraries using pip:
```bash
pip install -r requirements.txt
```
- Download the pre-trained YOLOv5s model weights from the official YOLOv5 repository:
  - Go to the [releases page](https://github.com/ultralytics/yolov5/releases) and download the `yolov5s.pt` file.
  - Place the downloaded `yolov5s.pt` file in the `yolov5` directory.

## Installation and Usage
1. Copy the `fire.py` file into the `yolov5` directory.
2. Change the working directory to the `yolov5` folder in the terminal:
```bash
cd /path/to/yolov5
```
3. Run the program using the following command:
```bash
python fire.py
```

The program uses the webcam to detect fire. If fire is detected, a warning message will be printed.

## Code Explanation
- The YOLOv5 model is loaded for object detection using the pre-trained weights in `yolov5s.pt`.
- Images are captured from the webcam and preprocessed.
- The detection results from the model are processed and displayed.
- Red and orange colors are checked to detect fire.
- If fire is detected, a warning message is printed.

Please let me know if you have any further questions or issues.
