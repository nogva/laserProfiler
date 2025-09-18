from utils import *
from dotRocognition import *

def main():
    play_video("video\initial_test.mp4", as_gray= True, scale=1, delay_ms=None, apply_threshold=True, thresh=249, show_mask=False)

if __name__ == "__main__":
    main()