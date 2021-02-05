import sys
import cv2
import yaml
from detectnetv2_grpc_client import Detectnetv2GrpcClient

def detect(cfg):
    detector = Detectnetv2GrpcClient(cfg['triton'])
    cap = cv2.VideoCapture(cfg['input_file'])
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
        sys.exit()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            new_shape = detector.infer_dims()
            frame = cv2.resize(frame, new_shape, interpolation = cv2.INTER_AREA)
            rects = detector(frame)
            for rect in rects:
                left, top, right, bottom = rect
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.imshow('Frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    with open('config/config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    detect(cfg)