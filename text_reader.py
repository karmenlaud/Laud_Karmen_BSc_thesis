import cv2
import numpy as np
import os
from paddleocr import PaddleOCR


class EAST:
    def __init__(
        self,
        path="frozen_east_text_detection.pb",
        conf_thresh=0.95,
        nms_thresh=0.4,
        mean=(122.67891434, 116.66876762, 104.00698793),
        scale=1.0,
        inputSize=(320, 320),
        swapRB=True
    ):
        """ Wrapper around OpenCV EAST text detector.
        Args:
            path (str): path to EAST frozen model
            conf_thresh (float): confidence threshold
            nms_thresh (float): NMS suppression threshold
            mean (tuple): mean normalization values
            scale (float): input scaling factor
            inputSize (tuple): network input size (W, H)
            swapRB (bool): swap R and B channels
        """

        self.net = cv2.dnn_TextDetectionModel_EAST(path)
        self.net.setConfidenceThreshold(conf_thresh)
        self.net.setNMSThreshold(nms_thresh)
        self.net.setInputParams(scale, inputSize, mean, swapRB)

        self.inputSize = inputSize
        self.scale = scale
        self.mean = mean
        self.swapRB = swapRB

    def detect(self, frame: np.ndarray):
        """ Detect text boxes in a full image.
        Args:
            frame (np.ndarray): input BGR image
        Returns:
            list[np.ndarray]: detected quadrilateral boxes in ORIGINAL image coordinates
            list[float]: confidence scores
        """
        boxes, confidences = self.net.detect(frame)

        if len(boxes) == 0:
            return [], []

        return [np.asarray(b, dtype=np.int32) for b in boxes], confidences


    def detect_tiled(self, frame: np.ndarray, overlap: float = 0.25):
        """ Detect text using overlapping tiles.
        Args:
            frame (np.ndarray): input image
            overlap (float): tile overlap ratio
        Returns:
            list[np.ndarray], list[float]
        """
        h, w = frame.shape[:2]
        tile_w, tile_h = self.inputSize

        step_x = int(tile_w * (1 - overlap))
        step_y = int(tile_h * (1 - overlap))

        all_boxes = []
        all_confs = []

        for y in range(0, h, step_y):
            for x in range(0, w, step_x):

                tile = frame[y:y + tile_h, x:x + tile_w]
                if tile.size == 0:
                    continue

                boxes, confs = self.net.detect(tile)

                for b in boxes:
                    b = np.asarray(b, dtype=np.float32)
                    b[:, 0] += x
                    b[:, 1] += y

                    all_boxes.append(b.astype(np.int32))

                all_confs.extend(confs)

        return all_boxes, all_confs

class TextReader:
    def __init__(self, east_model = None):
        self.EAST = east_model if east_model else EAST()
        self.ocr = PaddleOCR(lang="en", use_textline_orientation=True)
    

    def ocr_frame(self, frame: np.ndarray, east: bool = True, tiled: bool = True, debug: bool = False) -> list:
        """ Takes a frame from the camera containing text
            and extracts all text from it.
        Args:
            frame (list): a frame with confirmed text
        Returns:
            list: list of all the lines of text detected
        """
        screen_res = (1280, 720)
        
        h, w, _ = frame.shape
        
        if east:
            if tiled:
                boxes, _ = self.EAST.detect_tiled(frame)
            else:
                boxes, _ = self.EAST.detect(frame)
            
            if len(boxes) == 0:
                return []
            boxes = self.merge_boxes(boxes, h, w)
        else:
            boxes = [np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)]
        
        extracted_text = []
        
        
        for box in boxes:
            # box is 4x2 array: 4 vertices
            x_min = np.min(box[:, 0])
            y_min = np.min(box[:, 1])
            x_max = np.max(box[:, 0])
            y_max = np.max(box[:, 1])
            
            # Crop region
            crop = frame[y_min:y_max+1, x_min:x_max+1]
            if crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            
            # Run OCR
            ocr_result = self.ocr.predict(crop)
            if ocr_result[0] is not None:
                text = self.extract_text_simple(ocr_result)

                extracted_text.extend(text)
            
            if debug: 
                cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.namedWindow("Debug")
                cv2.moveWindow("Debug", 100, 100)
                cv2.imshow("Debug", crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        if debug:
            h, w = frame.shape[:2]
            scale = min(screen_res[0] / w, screen_res[1] / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            cv2.namedWindow("Debug")
            cv2.moveWindow("Debug", 100, 100)
            cv2.imshow("Debug", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return extracted_text

    @staticmethod
    def merge_boxes(boxes, h, w, offset = 5, merge_thresh = 5):
        """Merge any boxes that overlap partially or fully"""
        if not boxes:
            return []

        # Convert each box to [x_min, y_min, x_max, y_max] -> sorting in essence
        rects = []
        for b in boxes:
            x_min, y_min = np.min(b, axis=0)
            x_max, y_max = np.max(b, axis=0)
            rects.append([x_min-offset, y_min-offset, x_max+offset, y_max+offset])

        rects = np.array(rects, dtype=np.float32)

        merged = []
        used = np.zeros(len(rects), dtype=bool)

        for i in range(len(rects)):
            if used[i]:
                continue
            x1, y1, x2, y2 = rects[i]
            changed = True
            while changed:
                changed = False
                for j in range(len(rects)):
                    if i == j or used[j]:
                        continue
                    xx1, yy1, xx2, yy2 = rects[j]
                    # Check if boxes overlap
                    if  not (x2 + merge_thresh < xx1 or xx2 + merge_thresh < x1 or 
                             y2 + merge_thresh < yy1 or yy2 + merge_thresh < y1):
                        # Merge boxes
                        x1 = max(min(x1, xx1), 0)
                        y1 = max(min(y1, yy1), 0)
                        x2 = min(max(x2, xx2), w-1)
                        y2 = min(max(y2, yy2), h-1)
                        used[j] = True
                        changed = True
            merged.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32))
            used[i] = True

        return merged
    
    @staticmethod
    def extract_text_simple(ocr_result):
        """
        Extract all recognized text strings from PaddleOCR v3.x / PaddleX OCR results.
        
        Args:
            ocr_result (list of dicts): output from PaddleOCR.ocr(image)
            
        Returns:
            list: list of text strings detected in the image
        """
        texts = []
        for res in ocr_result:
            if 'rec_texts' in res and res['rec_texts'] is not None:
                texts.extend(res['rec_texts'])
        return texts
  
def main():
    script_path = os.path.dirname(__file__)
    img_path = os.path.join(script_path, 'sign.png')
    
    reader = TextReader()
    img = cv2.imread(img_path)
    empty = np.zeros_like(img)

    print("Extracted text:", reader.ocr_frame(empty, debug=True))

    print("Extracted text:", reader.ocr_frame(img.copy(), east=True, tiled=True, debug=True))
    #print("Extracted text:", reader.ocr_frame(img.copy(), tiled=False, debug=False))


if __name__ == "__main__":
    main()
    pass
