import numpy as np
import os
from sklearn.metrics import f1_score, recall_score
import tensorflow as tf
# 안녕하세요!!  지난번에 이어서 좋은 글 잘 읽었습니다.  mIoU 계산시에, true label에서는 등장하지 않고 predict
# label에서는 등장하는 label의 경우는 iou를 0으로 두고 계산을 해주나요?
# 안녕하세요.네 맞습니다.  간단히 교집합이 없으니 0이 되는 원리 입니다.  true label에는 없으나 predict에만 존재한다는 것
# 자체가 성능이 나쁜것이니 0에 수렴해야 하는 논리와 동일합니다.
# 저 위의 논리대라면, 나누
# 아!!!  predict한 이미지에 void부분을 11로 만들고!  miou를 구할 때, void클래스 빈도수는 제외하고 진행하면 되지
# 않을까?  기억해!!!!!  지금생각났음!!!!!!!!!!!!!
# 왜냐면!  어차피 predict와 label 위치에 같은 값이 동일하게 있다면 confusion metrice 할때 중앙성분만있음!
# 그렇기에 cm을 구한 뒤 대각성분을 추출한 뒤 맨 뒤에있는 void라벨을 제거하면 됨!  오키!!  내일 다시한번더
# 생각천천히해봐!!!기억해 꼭해!!!!!!!!!!!


class Measurement:
    def __init__(self, predict, label, shape, total_classes):
        self.predict = predict
        self.label = label
        self.total_classes = total_classes
        self.shape = shape

    def MIOU(self):
        # 0-crop, 1-weed, 2-background
        self.predict = np.reshape(self.predict, self.shape)
        self.label = np.reshape(self.label, self.shape)

        crop_back_predict = np.where(self.predict == 1, 2, self.predict)
        crop_back_predict = np.where(crop_back_predict == 2, 1, crop_back_predict)
        crop_back_label = np.where(self.label == 1, 2, self.label)
        crop_back_label = np.where(crop_back_label == 2, 1, crop_back_label)

        weed_back_predict = np.where(self.predict == 0, 2, self.predict)
        weed_back_predict = np.where(weed_back_predict == 1, 0, weed_back_predict)
        weed_back_predict = np.where(weed_back_predict == 2, 1, weed_back_predict)
        weed_back_label = np.where(self.label == 0, 2, self.label)
        weed_back_label = np.where(weed_back_label == 1, 0, weed_back_label)
        weed_back_label = np.where(weed_back_label == 2, 1, weed_back_label)

        predict_count = np.bincount(self.predict, minlength=self.total_classes)
        label_count = np.bincount(self.label, minlength=self.total_classes)

        crop_back_predict_count = np.bincount(crop_back_predict, minlength=self.total_classes - 1)
        crop_back_label_count = np.bincount(crop_back_label, minlength=self.total_classes - 1)

        weed_back_predict_count = np.bincount(weed_back_predict, minlength=self.total_classes - 1)
        weed_back_label_count = np.bincount(weed_back_label, minlength=self.total_classes - 1)

        crop_back_cm = tf.math.confusion_matrix(crop_back_label,
                                                crop_back_predict,
                                                num_classes=self.total_classes - 1).numpy()
        crop_iou = crop_back_cm[0, 0] / (crop_back_cm[0, 0] + crop_back_cm[0, 1] + crop_back_cm[1, 0])

        weed_back_cm = tf.math.confusion_matrix(weed_back_label,
                                                weed_back_predict,
                                                num_classes=self.total_classes - 1).numpy()
        weed_iou = weed_back_cm[0, 0] / (weed_back_cm[0, 0] + weed_back_cm[0, 1] + weed_back_cm[1, 0])

        total_cm = crop_back_cm + weed_back_cm

        if weed_iou == float('NaN'):
            weed_iou = 0.
        if crop_iou == float('NaN'):
            crop_iou = 0.

        return total_cm, crop_back_cm, weed_back_cm


    def F1_score_and_recall(self):  # recall - sensitivity

        # 1?? ???? predict?? label ?????????? (TP, FP)
        self.predict_positive = np.where(self.predict == 1, 1, 0)
        self.label_positive = np.where(self.label == 1, 1, 0)
        self.predict_positive = np.reshape(self.predict_positive, self.shape)
        self.label_positive = np.reshape(self.label_positive, self.shape)

        TP_func = lambda predict: predict[:] == 1
        TP_func2 = lambda predict, label: predict[:] == label[:]
        TP = np.where(TP_func(self.predict_positive) & TP_func2(self.predict_positive, self.label_positive), 1, 0)
        TP = np.sum(TP, dtype=np.int32)

        FP_func = lambda predict: predict[:] == 1
        FP_func2 = lambda predict, label: predict[:] != label[:]
        FP = np.where(FP_func(self.predict_positive) & FP_func2(self.predict_positive, self.label_positive), 1, 0)
        FP = np.sum(FP, dtype=np.int32)

        # 0?? ???? predict?? label ???????? (TN, FN)
        self.predict_negative = np.where(self.predict == 0, 0, 1)
        self.label_negative = np.where(self.label == 0, 0, 1)
        self.predict_negative = np.reshape(self.predict_negative, self.shape)
        self.label_negative = np.reshape(self.label_negative, self.shape)

        TN_func = lambda predict: predict[:] == 0
        TN_func2 = lambda predict, label: predict[:] == label[:]
        TN = np.where(TN_func(self.predict_negative) & TN_func2(self.predict_negative, self.label_negative), 1, 0)
        TN = np.sum(TN, dtype=np.int32)

        FN_func = lambda predict: predict[:] == 0
        FN_func2 = lambda predict, label: predict[:] != label[:]
        FN = np.where(FN_func(self.predict_negative) & FN_func2(self.predict_negative, self.label_negative), 1, 0)
        FN = np.sum(FN, dtype=np.int32)

        TP_FP = (TP + FP) + 1e-7

        TP_FN = (TP + FN) + 1e-7

        out = np.zeros((1))
        Precision = np.divide(TP, TP_FP)
        Recall = np.divide(TP, TP_FN)

        Pre_Re = (Precision + Recall) + 1e-7

        F1_score = np.divide(2. * (Precision * Recall), Pre_Re)

        return F1_score, Recall, TP, TN, FP, FN
    def show_confusion(self):
        # TP - Red [255, 0, 0] 10, TN - ?ϴû? [0, 255, 255] 20, FP - ??ȫ?? [255, 0, 255] 30, FN - ???? [255, 255, 0] 40

        self.predict = np.squeeze(self.predict, -1)

        color_map = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [236, 184, 199]])
        TP_func = lambda func:func[:] == 1
        output = np.where(TP_func(self.predict) & TP_func(self.label), 10, 0)  # get TP

        TN_func = lambda func:func[:] == 0
        output = np.where(TN_func(self.predict) & TN_func(self.label), 20, output)  # get TN
        
        FP_func = lambda func:func[:] == 1
        FP_func2 = lambda func:func[:] == 0
        FP_func3 = lambda func:func[:] == 2
        output = np.where(FP_func(self.predict) & (FP_func2(self.label)|FP_func3(self.label)), 30, output)  # get FP
        output = np.where(FP_func3(self.label) & FP_func(self.predict), 30, output)
        # ?? ?κп? ???? ?? ?߰????־?????!! ??????!1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        FN_func = lambda func:func[:] == 0
        FN_func2 = lambda func:func[:] == 1
        FN_func3 = lambda func:func[:] == 2
        output = np.where(FN_func(self.predict) & (FN_func2(self.label)|FN_func3(self.label)), 40, output)  # get FN
        output = np.where(FN_func3(self.label) & FN_func(self.predict), 40, output)
        # ?? ?κп? ???? ?? ?߰????־?????!! ??????!1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        output = np.expand_dims(output, -1)
        output_3D = np.concatenate([output, output, output], -1)
        temp_output_3D = output_3D
        
        output_3D = np.where(output == np.array([0, 0, 0], dtype=np.uint8), np.array([0, 0, 0], dtype=np.uint8), output_3D).astype(np.uint8)
        output_3D = np.where(output == np.array([10, 10, 10], dtype=np.uint8), np.array([255, 127, 0], dtype=np.uint8), output_3D).astype(np.uint8)   # TN    - ??Ȳ??
        output_3D = np.where(output == np.array([20, 20, 20], dtype=np.uint8), np.array([128, 128, 128], dtype=np.uint8), output_3D).astype(np.uint8) # TP    - ȸ??
        output_3D = np.where(output == np.array([30, 30, 30], dtype=np.uint8), np.array([139, 0, 255], dtype=np.uint8), output_3D).astype(np.uint8) # FN    - ??????
        output_3D = np.where(output == np.array([40, 40, 40], dtype=np.uint8), np.array([255, 255, 0], dtype=np.uint8), output_3D).astype(np.uint8) # FP    - ??????


        return output_3D, temp_output_3D
