# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image
import easydict

# 사용자 정의 모듈 (기존 모듈 불러오기)
from U_Net import U_NET

# 1. 설정
FLAGS = easydict.EasyDict({
    "img_size": 512,
    "total_classes": 3,  # 0: Crop, 1: Weed, 2: Background
    "target_class": 0,  # Grad-CAM을 추출할 타깃 클래스 (0: 작물, 1: 잡초)
    "alpha": 0.75,  # 원본 이미지와 Heatmap 오버레이 가중치
    "test_txt_path": "E:/bonirob_segemtation/IJRR2017_text/test.txt",
    "image_path": "E:/new/label_SR/bonirob/CNCAN_T/",
    "label_path": "E:/bonirob_segemtation/IJRR2017_seg/",
    # 최신 체크포인트가 들어있는 폴더 경로 (폴더명으로 전달)
    "checkpoint_dir": "E:/new/label_Unet_seg/CNCAN_T/label_SR2original_label/checkpoint/1",
    "output_base_dir": "E:/ED_grad_results/CWFID_image2image/CNCAN_L_fold1_SR_results/1/crop"
})


# 2. 데이터 로더 함수 (기존 test_func2 전처리 방식 유지)
def test_func2(image_path, label_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0.0, 255.0)

    # 시각화 오버레이용 원본 이미지 (RGB, 0~255 uint8 형태)
    raw_img = tf.cast(img, tf.uint8)

    # 모델 입력용 전처리 (BGR 변환 및 ImageNet mean 차감)
    norm_img = tf.cast(img, tf.float32)
    norm_img = norm_img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)

    lab = tf.io.read_file(label_path)
    lab = tf.image.decode_png(lab, channels=1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return norm_img, raw_img, lab


# 3. Grad-CAM 연산 함수
def compute_gradcam(grad_model, input_tensor, target_class_idx):
    """
    input_tensor: [1, 512, 512, 3] (전처리 완료된 입력 텐서)
    target_class_idx: 관심 클래스 인덱스 (int)
    """
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)
        # 타깃 클래스 로짓 스코어 합산
        target_score = predictions[0, :, :, target_class_idx]
        loss = tf.reduce_sum(target_score)

    # Feature Map(ud0)에 대한 Gradient 역전파
    grads = tape.gradient(loss, conv_outputs)

    # Global Average Pooling (가중치 계산)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Feature Map과 가중치 선형 결합
    conv_outputs = conv_outputs[0]
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # ReLU 적용 (양수 영향력 추출)
    cam = tf.maximum(cam, 0)

    # 0 ~ 1 정규화
    cam_min, cam_max = tf.reduce_min(cam), tf.reduce_max(cam)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam.numpy()


def main():
    # ----------------------------------------------------
    # (1) 저장 경로 설정 (ED_grad_results_현재시간)
    # ----------------------------------------------------
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(FLAGS.output_base_dir, f"ED_grad_results_{current_time}")
    os.makedirs(save_dir, exist_ok=True)
    print(f">> 결과 저장 경로: {save_dir}")

    # ----------------------------------------------------
    # (2) 모델 생성 및 체크포인트 폴더 단위 복원
    # ----------------------------------------------------
    tf.keras.backend.clear_session()
    model = U_NET(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), classes=FLAGS.total_classes)

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_dir, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f">> 체크포인트 복원 성공: {ckpt_manager.latest_checkpoint}")
    else:
        raise FileNotFoundError(f">> 지정한 경로에 체크포인트가 존재하지 않습니다: {FLAGS.checkpoint_dir}")

    # ----------------------------------------------------
    # (3) Grad-CAM 추출용 Multi-Output 모델 분리
    # ----------------------------------------------------
    # U-Net 디코더의 마지막 합성곱 레이어(ud0: 마지막 Conv2D(64)) 선택
    target_conv_layer = model.layers[-2]  # output 직전의 64채널 Conv2D
    print(f">> Grad-CAM 타깃 레이어: {target_conv_layer.name}")

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[target_conv_layer.output, model.output]
    )

    # ----------------------------------------------------
    # (4) 전체 테스트 데이터셋 로드 및 배치 반복
    # ----------------------------------------------------
    test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)
    test_img_dataset = [FLAGS.image_path + data for data in test_list]
    test_lab_dataset = [FLAGS.label_path + data for data in test_list]

    test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset)) \
        .map(test_func2) \
        .batch(1)

    print(f">> 총 {len(test_list)}개 이미지에 대해 Grad-CAM 추출 시작 (Target Class: {FLAGS.target_class})")

    for i, (norm_img, raw_img, _) in enumerate(test_ge):
        # 1. Grad-CAM 맵 생성
        cam_map = compute_gradcam(grad_model, norm_img, target_class_idx=FLAGS.target_class)

        # 2. 컬러맵(Jet) 변환 및 오버레이
        cam_uint8 = np.uint8(255 * cam_map)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        raw_np = raw_img.numpy()[0]
        overlay = np.uint8(FLAGS.alpha * heatmap + (1.0 - FLAGS.alpha) * raw_np)

        # 3. 파일 저장 (원본명 기반)
        img_name = test_list[i].split("/")[-1].split(".")[0]
        save_path = os.path.join(save_dir, f"{img_name}_class{FLAGS.target_class}_gradcam.png")

        # [원본, 히트맵, 오버레이]를 나란히 비교해서 저장하고 싶다면 아래 주석을 해제하세요.
        # comparison = np.hstack([raw_np, heatmap, overlay])
        # Image.fromarray(comparison).save(save_path)

        Image.fromarray(overlay).save(save_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(test_list):
            print(f"[{i + 1}/{len(test_list)}] 저장 완료: {save_path}")

    print(f">> 모든 이미지의 Grad-CAM 저장이 완료되었습니다: {save_dir}")


if __name__ == "__main__":
    main()