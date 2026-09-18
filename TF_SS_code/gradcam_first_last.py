# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image
import easydict

# 사용자 정의 모듈
from U_Net import U_NET

# ----------------------------------------------------
# 1. 환경 설정
# ----------------------------------------------------
FLAGS = easydict.EasyDict({
    "img_size": 512,
    "total_classes": 3,  # 0: Crop, 1: Weed, 2: Background
    "target_class": 0,  # Grad-CAM을 추출할 타깃 클래스 (0: 작물, 1: 잡초)
    "alpha": 0.75,  # 원본 이미지와 Heatmap 오버레이 가중치
    "test_txt_path": "E:/bonirob_segemtation/IJRR2017_text/test.txt",
    "image_path": "E:/new/label_SR/bonirob/CNCAN_T/",
    "label_path": "E:/bonirob_segemtation/IJRR2017_seg/",
    "checkpoint_dir": "E:/new/label_Unet_seg/CNCAN_T/label_SR2original_label/checkpoint/418",
    "output_base_dir": "E:/ED_grad_results/CWFID_image2image/CNCAN_L_fold1_SR_results/418/crop"
})


# ----------------------------------------------------
# 2. 이미지 데이터 로더 (PNG/JPG 범용 지원)
# ----------------------------------------------------
def test_func_universal(image_path, label_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0.0, 255.0)

    # 원본 이미지 (RGB, uint8)
    raw_img = tf.cast(img, tf.uint8)

    # 모델 입력용 전처리 (BGR 및 Mean 차감)
    norm_img = tf.cast(img, tf.float32)
    norm_img = norm_img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)

    lab = tf.io.read_file(label_path)
    lab = tf.io.decode_image(lab, channels=1, expand_animations=False)
    lab.set_shape([None, None, 1])
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return norm_img, raw_img, lab


# ----------------------------------------------------
# 3. Dual-Layer Grad-CAM 계산 함수
# ----------------------------------------------------
def compute_dual_layer_gradcam(grad_model, input_tensor, target_class_idx):
    """
    first_conv, last_conv 두 레이어의 Grad-CAM을 동시에 계산
    """
    with tf.GradientTape(persistent=True) as tape:
        first_conv_out, last_conv_out, predictions = grad_model(input_tensor, training=False)
        target_score = predictions[0, :, :, target_class_idx]
        loss = tf.reduce_sum(target_score)

    def _get_cam(conv_outputs):
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
        cam = tf.maximum(cam, 0)
        cam_min, cam_max = tf.reduce_min(cam), tf.reduce_max(cam)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.numpy()

    first_cam = _get_cam(first_conv_out)
    last_cam = _get_cam(last_conv_out)
    del tape

    return first_cam, last_cam


def overlay_heatmap(raw_img_np, cam_map, alpha=0.5):
    h, w, _ = raw_img_np.shape
    cam_resized = cv2.resize(cam_map, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap + (1.0 - alpha) * raw_img_np)
    return overlay


# ----------------------------------------------------
# 4. 메인 실행 함수
# ----------------------------------------------------
def main():
    # 1. 저장 디렉터리 생성 (ED_grad_results_YYYYMMDD_HHMMSS)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(FLAGS.output_base_dir, f"ED_grad_results_{current_time}")
    os.makedirs(save_dir, exist_ok=True)
    print(f">> 결과 저장 폴더: {save_dir}")

    # 2. 모델 인스턴스 생성 및 가중치 복원
    tf.keras.backend.clear_session()
    model = U_NET(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), classes=FLAGS.total_classes)

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpoint_dir, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print(f">> 체크포인트 로드 완료: {ckpt_manager.latest_checkpoint}")
    else:
        raise FileNotFoundError(f">> 체크포인트를 찾을 수 없습니다: {FLAGS.checkpoint_dir}")

    # 3. First Layer와 Last Layer 타깃 지정
    # U-Net 구조상:
    # - model.layers[2]: 첫 번째 블록의 두 번째 Conv (dc0)
    # - model.layers[-2]: 디코더 마지막 Conv (ud0)
    first_conv_layer = model.layers[2]
    last_conv_layer = model.layers[-2]

    print(f">> First Layer: {first_conv_layer.name}")
    print(f">> Last Layer:  {last_conv_layer.name}")

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[first_conv_layer.output, last_conv_layer.output, model.output]
    )

    # 4. 파일 리스트 불러오기 및 경로 검증
    test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

    valid_img_paths = []
    valid_lab_paths = []
    valid_filenames = []

    for item in test_list:
        clean_name = os.path.basename(item.replace("\\", "/"))
        img_p = os.path.join(FLAGS.image_path, clean_name).replace("\\", "/")
        lab_p = os.path.join(FLAGS.label_path, clean_name).replace("\\", "/")

        if os.path.exists(img_p) and os.path.exists(lab_p):
            valid_img_paths.append(img_p)
            valid_lab_paths.append(lab_p)
            valid_filenames.append(clean_name)
        else:
            if not os.path.exists(img_p):
                print(f"[경로 오류 건너뜀] 이미지가 존재하지 않음: {img_p}")
            if not os.path.exists(lab_p):
                print(f"[경로 오류 건너뜀] 라벨이 존재하지 않음: {lab_p}")

    print(f">> 총 {len(valid_filenames)}/{len(test_list)} 개 파일 검증 완료. Grad-CAM 추출을 시작합니다.")

    # 5. 데이터셋 구성 및 추론 루프
    test_ge = tf.data.Dataset.from_tensor_slices((valid_img_paths, valid_lab_paths)) \
        .map(test_func_universal) \
        .batch(1)

    for i, (norm_img, raw_img, _) in enumerate(test_ge):
        first_cam, last_cam = compute_dual_layer_gradcam(
            grad_model, norm_img, target_class_idx=FLAGS.target_class
        )

        raw_np = raw_img.numpy()[0]
        first_overlay = overlay_heatmap(raw_np, first_cam, alpha=FLAGS.alpha)
        last_overlay = overlay_heatmap(raw_np, last_cam, alpha=FLAGS.alpha)

        # 텍스트 라벨 추가 (비교 시각화용)
        cv2.putText(raw_np, "Original", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(first_overlay, f"First Layer ({first_conv_layer.name})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        cv2.putText(last_overlay, f"Last Layer ({last_conv_layer.name})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        # [Original | First Layer CAM | Last Layer CAM] 3단 가로 결합
        side_by_side = np.hstack([raw_np, first_overlay, last_overlay])

        # 저장
        base_name = os.path.splitext(valid_filenames[i])[0]
        save_path = os.path.join(save_dir, f"{base_name}_class{FLAGS.target_class}_dual_cam.png")
        Image.fromarray(side_by_side).save(save_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(valid_filenames):
            print(f"[{i + 1}/{len(valid_filenames)}] 저장 완료: {save_path}")

    print(f"\n>> 모든 작업이 완료되었습니다. 결과 경로:\n{save_dir}")


if __name__ == "__main__":
    main()