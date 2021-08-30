import cv2


class CropLayer(object):
    """
    Used for HNE Model
    """
    def __init__(self, params, blobs):
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


class HNEModel:
    PROTO_PATH = "./data/Edge_detection_model/deploy.prototxt"
    MODEL_PATH = "./data/Edge_detection_model/hed_pretrained_bsds.caffemodel"

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(self.PROTO_PATH, self.MODEL_PATH)
        # register our new layer with the model
        cv2.dnn_registerLayer("Crop", CropLayer)

    def process_one(self, img):
        (H, W) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=False)
        self.net.setInput(blob)
        result = self.net.forward()
        result = cv2.resize(result[0, 0], (W, H))
        result = (255 * result).astype("uint8")
        return result