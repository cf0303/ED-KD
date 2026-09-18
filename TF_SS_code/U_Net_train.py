# -*- coding:utf-8 -*-
import tensorflow as tf
from modified_Unet_model import *
from U_Net import *
from PFB_measurement_related import Measurement
from random import shuffle, random


import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 512,

                           "train_txt_path": "",

                           "val_txt_path": "",

                           "test_txt_path": "",

                           "label_path": "",

                           "image_path": "",

                           "pre_checkpoint": False,

                           "pre_checkpoint_path": "",

                           "lr": 0.00001, #default : 0.0001

                           "min_lr": 1e-7,

                           "epochs": 500,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 2,

                        #    "sample_images": "/content/drive/MyDrive/SJRN_v0_2/sample_images",
 
                           "save_checkpoint": "",

                           "save_print": "",

                           "test_images": "",

                           "train": True})


optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.9, beta_2=0.99)
color_map = np.array([[255, 0, 0], [0, 0, 255], [0,0,0]], dtype=np.uint8)

def tr_func(image_list, label_list):

    h = tf.random.uniform([1], 1e-2, 30)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 30)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.random_brightness(img, max_delta=50.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    no_img = img
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)

    return img, no_img, lab

def test_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0, 255)
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

def test_func2(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0, 255)
    temp_img = img
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, temp_img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

def cal_loss(model, images, batch_labels):

    with tf.GradientTape() as tape:

        batch_labels = tf.reshape(batch_labels, [-1,])

        logits = run_model(model, images, True)
        logits = tf.reshape(logits, [-1, FLAGS.total_classes])

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(batch_labels, logits)

        # print(tf.keras.losses.BinaryCrossentropy(from_logits=True)(crop_labels, crop_logits))
        # print(tf.keras.losses.BinaryCrossentropy(from_logits=True)(weed_labels, weed_logits))
       
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# yilog(h(xi;θ))+(1−yi)log(1−h(xi;θ))
def main():
    tf.keras.backend.clear_session()
    # 마지막 plain은 objecttines에 대한 True or False값 즉 (mask값이고), 라벨은 annotation 이미지임 (crop/weed)
    # 학습이미지에 대해 online augmentation을 진행--> 전처리로서 필터링을 하던지 해서 , 피사체에 대한 high frequency 성분을
    # 가지고오자
    #model = PFB_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), OUTPUT_CHANNELS=FLAGS.total_classes-1)\
    model = U_NET(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), classes=FLAGS.total_classes)

    #out = model.get_layer("activation_decoder_2_upsample").output
    #out = tf.keras.layers.Conv2D(FLAGS.total_classes-1, (1,1), name="output_layer")(out)
    #model = tf.keras.Model(inputs=model.input, outputs=out)
    
    #for layer in model.layers:
    #    if isinstance(layer, tf.keras.layers.BatchNormalization):
    #        layer.momentum = 0.9997
    #        layer.epsilon = 1e-5
        #elif isinstance(layer, tf.keras.layers.Conv2D):
        #    layer.kernel_regularizer = tf.keras.regularizers.l2(0.0005)

    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    
    if FLAGS.train:
        count = 0
        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        val_list = np.loadtxt(FLAGS.val_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.image_path + data for data in train_list]
        val_img_dataset = [FLAGS.image_path + data for data in val_list]
        test_img_dataset = [FLAGS.image_path + data for data in test_list]

        train_lab_dataset = [FLAGS.label_path + data for data in train_list]
        val_lab_dataset = [FLAGS.label_path + data for data in val_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        val_ge = tf.data.Dataset.from_tensor_slices((val_img_dataset, val_lab_dataset))
        val_ge = val_ge.map(test_func)
        val_ge = val_ge.batch(1)
        val_ge = val_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(train_ge)
            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, print_images, batch_labels = next(tr_iter)  
                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == FLAGS.ignore_label, 2, batch_labels)    # 2 is void
                batch_labels = np.where(batch_labels == 255, 0, batch_labels)
                batch_labels = np.where(batch_labels == 128, 1, batch_labels)

                # crop_labels = np.where(batch_labels == 0, 1, 0)
                # weed_labels = np.where(batch_labels == 1, 1, 0)

                # crop_labels = np.squeeze(crop_labels, -1)
                # weed_labels = np.squeeze(weed_labels, -1)

                loss = cal_loss(model, batch_images, batch_labels)  # loss까지는 다했고 내일 test iou뽑는 부분코드 고쳐야함!
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))

                if count % 100 == 0:

                    logits = run_model(model, batch_images, False)
                    for i in range(FLAGS.batch_size):
                        logit = logits[i]
                        logit = tf.nn.softmax(logit, -1)
                        predict_image = tf.argmax(logit, -1)
                        
                        label = batch_labels[i]
                        label = np.squeeze(label, -1)
                        
                        pred_mask_color = color_map[predict_image]  # 논문그림처럼 할것!
                        
                        label = np.expand_dims(label, -1)
                        label = np.concatenate((label, label, label), -1)
                        label_mask_color = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.uint8)
                        label_mask_color = np.where(label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                        label_mask_color = np.where(label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                        temp_img = predict_image
                        temp_img = np.expand_dims(temp_img, -1)
                        temp_img2 = temp_img
                        temp_img = np.concatenate((temp_img, temp_img, temp_img), -1)
                        image = np.concatenate((temp_img2, temp_img2, temp_img2), -1)
                        pred_mask_warping = np.where(temp_img == np.array([2,2,2], dtype=np.uint8), print_images[i], image)
                        pred_mask_warping = np.where(temp_img == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), pred_mask_warping)
                        pred_mask_warping = np.where(temp_img == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), pred_mask_warping)
                        pred_mask_warping /= 255.

                        # plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_label.png", label_mask_color)
                        # plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict.png", pred_mask_color)
                        # plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_warping_predict.png", pred_mask_warping)
                    

                count += 1

            tr_iter = iter(train_ge)
            miou = 0.
            f1_score_ = 0.
            crop_iou = 0.
            weed_iou = 0.
            recall_ = 0.
            precision_ = 0.
            for i in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    logits = run_model(model, batch_image, False)
                    logits = tf.nn.softmax(logits, -1)
                    predict_image = tf.argmax(logits, -1)

                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=predict_image,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()

                    miou += miou_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_

            miou_ = miou[0,0]/(miou[0,0] + miou[0,1] + miou[1,0])
            crop_iou_ = crop_iou[0,0]/(crop_iou[0,0] + crop_iou[0,1] + crop_iou[1,0])
            weed_iou_ = weed_iou[0,0]/(weed_iou[0,0] + weed_iou[0,1] + weed_iou[1,0])
            recall_ = miou[0,0] / (miou[0,0] + miou[0,1])
            precision_ = miou[0,0] / (miou[0,0] + miou[1,0])
            f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
            print("train mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), train F1_score = %.4f, train sensitivity(recall) = %.4f, train precision = %.4f" % (miou_,
                                                                                                                                                crop_iou_,
                                                                                                                                                weed_iou_,
                                                                                                                                                f1_score_,
                                                                                                                                                recall_,
                                                                                                                                                precision_))
            output_text.write("Epoch: ")
            output_text.write(str(epoch))
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.write("train mIoU: ")
            output_text.write("%.4f" % (miou_))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou_))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou_))
            output_text.write(", train F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", train sensitivity: ")
            output_text.write("%.4f" % (recall_))
            output_text.write(", train precision: ")
            output_text.write("%.4f" % (precision_))
            output_text.write("\n")

            val_iter = iter(val_ge)
            miou = 0.
            f1_score_ = 0.
            crop_iou = 0.
            weed_iou = 0.
            recall_ = 0.
            precision_ = 0.
            for i in range(len(val_img_dataset)):
                batch_images, batch_labels = next(val_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                for j in range(1):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    logits = run_model(model, batch_image, False)
                    logits = tf.nn.softmax(logits, -1)
                    predict_image = tf.argmax(logits, -1)

                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=predict_image,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()

                    miou += miou_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_

            miou_ = miou[0,0]/(miou[0,0] + miou[0,1] + miou[1,0])
            crop_iou_ = crop_iou[0,0]/(crop_iou[0,0] + crop_iou[0,1] + crop_iou[1,0])
            weed_iou_ = weed_iou[0,0]/(weed_iou[0,0] + weed_iou[0,1] + weed_iou[1,0])
            recall_ = miou[0,0] / (miou[0,0] + miou[0,1])
            precision_ = miou[0,0] / (miou[0,0] + miou[1,0])
            f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
            print("val mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), val F1_score = %.4f, val sensitivity(recall) = %.4f, val precision = %.4f" % (miou_,
                                                                                                                                                crop_iou_,
                                                                                                                                                weed_iou_,
                                                                                                                                                f1_score_,
                                                                                                                                                recall_,
                                                                                                                                                precision_))
            output_text.write("val mIoU: ")
            output_text.write("%.4f" % (miou_))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou_))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou_))
            output_text.write(", val F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", val sensitivity: ")
            output_text.write("%.4f" % (recall_))
            output_text.write(", val precision: ")
            output_text.write("%.4f" % (precision_))
            output_text.write("\n")

            test_iter = iter(test_ge)
            miou = 0.
            f1_score_ = 0.
            crop_iou = 0.
            weed_iou = 0.
            recall_ = 0.
            precision_ = 0.
            for i in range(len(test_img_dataset)):
                batch_images, batch_labels = next(test_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                for j in range(1):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    logits = run_model(model, batch_image, False)
                    logits = tf.nn.softmax(logits, -1)
                    predict_image = tf.argmax(logits, -1)

                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=predict_image,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()

                    miou += miou_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_

            miou_ = miou[0,0]/(miou[0,0] + miou[0,1] + miou[1,0])
            crop_iou_ = crop_iou[0,0]/(crop_iou[0,0] + crop_iou[0,1] + crop_iou[1,0])
            weed_iou_ = weed_iou[0,0]/(weed_iou[0,0] + weed_iou[0,1] + weed_iou[1,0])
            recall_ = miou[0,0] / (miou[0,0] + miou[0,1])
            precision_ = miou[0,0] / (miou[0,0] + miou[1,0])
            f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
            print("test mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), test F1_score = %.4f, test sensitivity(recall) = %.4f, test precision = %.4f" % (miou_,
                                                                                                                                                crop_iou_,
                                                                                                                                                weed_iou_,
                                                                                                                                                f1_score_,
                                                                                                                                                recall_,
                                                                                                                                                precision_))
            print("=================================================================================================================================================")
            output_text.write("test mIoU: ")
            output_text.write("%.4f" % (miou_))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou_))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou_))
            output_text.write(", test F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", test sensitivity: ")
            output_text.write("%.4f" % (recall_))
            output_text.write(", test precision: ")
            output_text.write("%.4f" % (precision_))
            output_text.write("\n")
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.flush()

            model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                print("Make {} folder to store the weight!".format(epoch))
                os.makedirs(model_dir)
            ckpt = tf.train.Checkpoint(model=model, optim=optim)
            ckpt_dir = model_dir + "/Crop_weed_model_{}.ckpt".format(epoch)
            ckpt.save(ckpt_dir)
    else:
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        test_img_dataset = [FLAGS.image_path + data for data in test_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func2)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_iter = iter(test_ge)
        miou = 0.
        f1_score_ = 0.
        crop_iou = 0.
        weed_iou = 0.
        recall_ = 0.
        precision_ = 0.
        for i in range(len(test_img_dataset)):
            batch_images, nomral_img, batch_labels = next(test_iter)
            batch_labels = tf.squeeze(batch_labels, -1)
            for j in range(1):
                batch_image = tf.expand_dims(batch_images[j], 0)
                logits = run_model(model, batch_image, False) # type을 batch label과 같은 type으로 맞춰주어야함

                    
                logits = tf.nn.softmax(logits, -1)
                predict_image = tf.argmax(logits, -1)

                batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                batch_label = np.where(batch_label == 255, 0, batch_label)
                batch_label = np.where(batch_label == 128, 1, batch_label)

                miou_, crop_iou_, weed_iou_ = Measurement(predict=predict_image,
                                    label=batch_label, 
                                    shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                    total_classes=FLAGS.total_classes).MIOU()

                pred_mask_color = color_map[predict_image]  # 논문그림처럼 할것!
                batch_label = np.expand_dims(batch_label, -1)
                batch_label = np.concatenate((batch_label, batch_label, batch_label), -1)
                label_mask_color = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.uint8)
                label_mask_color = np.where(batch_label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                label_mask_color = np.where(batch_label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                predict_image = np.expand_dims(predict_image, -1)
                temp_img = np.concatenate((predict_image, predict_image, predict_image), -1)
                image = np.concatenate((predict_image, predict_image, predict_image), -1)
                pred_mask_warping = np.where(temp_img == np.array([2,2,2], dtype=np.uint8), nomral_img[j], image)
                pred_mask_warping = np.where(temp_img == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), pred_mask_warping)
                pred_mask_warping = np.where(temp_img == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), pred_mask_warping)
                pred_mask_warping /= 255.

                name = test_img_dataset[i].split("/")[-1].split(".")[0]
                plt.imsave(FLAGS.test_images + "/" + name + "_label.png", label_mask_color)
                plt.imsave(FLAGS.test_images + "/" + name + "_predict.png", pred_mask_color)
                plt.imsave(FLAGS.test_images + "/" + name + "_predict_warp.png", pred_mask_warping[0])

                miou += miou_
                crop_iou += crop_iou_
                weed_iou += weed_iou_

        miou_ = miou[0,0]/(miou[0,0] + miou[0,1] + miou[1,0])
        crop_iou_ = crop_iou[0,0]/(crop_iou[0,0] + crop_iou[0,1] + crop_iou[1,0])
        weed_iou_ = weed_iou[0,0]/(weed_iou[0,0] + weed_iou[0,1] + weed_iou[1,0])
        recall_ = miou[0,0] / (miou[0,0] + miou[0,1])
        precision_ = miou[0,0] / (miou[0,0] + miou[1,0])
        f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
        print("test mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), test F1_score = %.4f, test sensitivity(recall) = %.4f, test precision = %.4f" % (miou_,
                                                                                                                                            crop_iou_,
                                                                                                                                            weed_iou_,
                                                                                                                                            f1_score_,
                                                                                                                                            recall_,
                                                                                                                                            precision_))


if __name__ == "__main__":
    main()