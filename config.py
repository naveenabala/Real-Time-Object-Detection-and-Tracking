%%writefile config.py
MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5
CLASSES_OF_INTEREST = [0]  # COCO class 0 = "person". Set to None to detect all 80 COCO classes.

VIDEO_SOURCE = "sample_video.avi"
OUTPUT_VIDEO_PATH = "output_annotated.mp4"
OUTPUT_SUMMARY_JSON = "detection_summary.json"

LINE_START = (0, 288)
LINE_END = (768, 288)

DEMO_IMAGE_PATH = "demo_image.png"
DEMO_OUTPUT_PATH = "opencv_basics_panel.png"
%%writefile download_sample_video.py
import urllib.request
import config

URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"


def main():
    print(f"Downloading sample video to {config.VIDEO_SOURCE} ...")
    urllib.request.urlretrieve(URL, config.VIDEO_SOURCE)
    print("Done.")


if __name__ == "__main__":
    main()

