import time
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from yolov2 import *

class CocoPredictor:
    def __init__(self):
        # hyper parameters
        weight_file = "./yolov2_darknet.model"
        self.n_classes = 80
        self.n_boxes = 5
        self.detection_thresh = 0.5
        self.iou_thresh = 0.5
        self.labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
        anchors = [[0.738768, 0.874946], [2.42204, 2.65704], [4.30971, 7.04493], [10.246, 4.59428], [12.6868, 11.8741]]

        # load model
        print("loading coco model...")
        yolov2 = YOLOv2(n_classes=self.n_classes, n_boxes=self.n_boxes)
        serializers.load_hdf5(weight_file, yolov2) # load saved model
        model = YOLOv2Predictor(yolov2)
        model.init_anchor(anchors)
        model.predictor.train = False
        model.predictor.finetune = False
        self.model = model

    def __call__(self, orig_img):
        orig_input_height, orig_input_width, _ = orig_img.shape
        #img = cv2.resize(orig_img, (640, 640))
        img = reshape_to_yolo_size(orig_img)
        input_height, input_width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)


        # forward
        x_data = img[np.newaxis, :, :, :]
        x = Variable(x_data)
        x, y, w, h, conf, prob = self.model.predict(x)

        # parse results
        _, _, _, grid_h, grid_w = x.shape
        x = F.reshape(x, (self.n_boxes, grid_h, grid_w)).data
        y = F.reshape(y, (self.n_boxes, grid_h, grid_w)).data
        w = F.reshape(w, (self.n_boxes, grid_h, grid_w)).data
        h = F.reshape(h, (self.n_boxes, grid_h, grid_w)).data
        conf = F.reshape(conf, (self.n_boxes, grid_h, grid_w)).data
        prob = F.transpose(F.reshape(prob, (self.n_boxes, self.n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data
        detected_indices = (conf * prob).max(axis=0) > self.detection_thresh

        results = []
        for i in range(detected_indices.sum()):
            results.append({
                "class_id": prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax(),
                "label": self.labels[prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax()],
                "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
                "conf" : conf[detected_indices][i],
                "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
                "box"  : Box(
                            x[detected_indices][i]*orig_input_width,
                            y[detected_indices][i]*orig_input_height,
                            w[detected_indices][i]*orig_input_width,
                            h[detected_indices][i]*orig_input_height).crop_region(orig_input_height, orig_input_width)
            })

        # nms
        nms_results = nms(results, self.iou_thresh)
        return nms_results

if __name__ == "__main__":
    # argument parse
    parser = argparse.ArgumentParser(description="指定したパスの画像を読み込み、bbox及びクラスの予測を行う")
    parser.add_argument('path', help="画像ファイルへのパスを指定")
    args = parser.parse_args()
    image_file = args.path

    # read image
    print("loading image...")
    orig_img = cv2.imread(image_file)

    predictor = CocoPredictor()
    nms_results = predictor(orig_img)

    # draw result
    for result in nms_results:
        left, top = result["box"].int_left_top()
        cv2.rectangle(
            orig_img,
            result["box"].int_left_top(), result["box"].int_right_bottom(),
            (0, 255, 0),
            5
        )
        text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
        cv2.putText(orig_img, text, (left, top-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print(text)

    print("save results to yolov2_result.jpg")
    cv2.imwrite("yolov2_result.jpg", orig_img)
    cv2.imshow("w", orig_img)
    cv2.waitKey()
