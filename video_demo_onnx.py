#!/usr/bin/env python3
import sys
import argparse
import cv2
import os
import numpy as np
import onnxruntime
import pickle
import pandas as pd
import skimage.transform
DEBUG = True 

# Define the source landmarks for alignment.
SRC = np.array(
    [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]
    ], dtype=np.float32)
SRC[:, 0] += 8.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)
            
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)
    
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def nms(dets, iou_thresh=0.7):
    thresh = iou_thresh
    eps = 1e-7
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + eps) * (y2 - y1 + eps)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + eps)
        h = np.maximum(0.0, yy2 - yy1 + eps)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def build_onnx_model(model_file):
    providers =  ['CPUExecutionProvider']
    session_detect = onnxruntime.InferenceSession(model_file, providers=providers )
    return session_detect

def find_cosine_distance(source_representation, test_representation):
    """
    Find cosine distance between two given vectors or batches of vectors.
    Args:
        source_representation (np.ndarray or list): 1st vector or batch of vectors.
        test_representation (np.ndarray or list): 2nd vector or batch of vectors.
    Returns
        np.float64 or np.ndarray: Calculated cosine distance(s).
        It returns a np.float64 for single embeddings and np.ndarray for batch embeddings.
    """
    # Convert inputs to numpy arrays if necessary
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    if source_representation.ndim == 1 and test_representation.ndim == 1:
        # single embedding
        dot_product = np.dot(source_representation, test_representation)
        source_norm = np.linalg.norm(source_representation)
        test_norm = np.linalg.norm(test_representation)
        distances = 1 - dot_product / (source_norm * test_norm)
    elif source_representation.ndim == 2 and test_representation.ndim == 2:
        # list of embeddings (batch)
        source_normed = l2_normalize(source_representation, axis=1)  # (N, D)
        test_normed = l2_normalize(test_representation, axis=1)  # (M, D)
        cosine_similarities = np.dot(test_normed, source_normed.T)  # (M, N)
        distances = 1 - cosine_similarities
    else:
        raise ValueError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )
    return distances

def find_euclidean_distance(source_representation, test_representation):
    """
    Find Euclidean distance between two vectors or batches of vectors.

    Args:
        source_representation (np.ndarray or list): 1st vector or batch of vectors.
        test_representation (np.ndarray or list): 2nd vector or batch of vectors.

    Returns:
        np.float64 or np.ndarray: Euclidean distance(s).
            Returns a np.float64 for single embeddings and np.ndarray for batch embeddings.
    """
    # Convert inputs to numpy arrays if necessary
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    # Single embedding case (1D arrays)
    if source_representation.ndim == 1 and test_representation.ndim == 1:
        distances = np.linalg.norm(source_representation - test_representation)
    # Batch embeddings case (2D arrays)
    elif source_representation.ndim == 2 and test_representation.ndim == 2:
        diff = (
            source_representation[None, :, :] - test_representation[:, None, :]
        )  # (N, D) - (M, D)  = (M, N, D)
        distances = np.linalg.norm(diff, axis=2)  # (M, N)
    else:
        raise ValueError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )
    return distances


def l2_normalize(x, axis = None, epsilon = 1e-10):
    """
    Normalize input vector with l2
    Args:
        x (np.ndarray or list): given vector
        axis (int): axis along which to normalize
    Returns:
        np.ndarray: l2 normalized vector
    """
    # Convert inputs to numpy arrays if necessary
    x = np.asarray(x)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + epsilon)

def find_distance(
    alpha_embedding,
    beta_embedding,
    distance_metric,
):
    """
    Wrapper to find the distance between vectors based on the specified distance metric.

    Args:
        alpha_embedding (np.ndarray or list): 1st vector or batch of vectors.
        beta_embedding (np.ndarray or list): 2nd vector or batch of vectors.
        distance_metric (str): The type of distance to compute
            ('cosine', 'euclidean', or 'euclidean_l2').

    Returns:
        np.float64 or np.ndarray: The calculated distance(s).
    """
    # Convert inputs to numpy arrays if necessary
    alpha_embedding = np.asarray(alpha_embedding)
    beta_embedding = np.asarray(beta_embedding)

    # Ensure that both embeddings are either 1D or 2D
    if alpha_embedding.ndim != beta_embedding.ndim or alpha_embedding.ndim not in (1, 2):
        raise ValueError(
            f"Both embeddings must be either 1D or 2D, but received "
            f"alpha shape: {alpha_embedding.shape}, beta shape: {beta_embedding.shape}"
        )

    if distance_metric == "cosine":
        distance = find_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_l2":
        axis = None if alpha_embedding.ndim == 1 else 1
        normalized_alpha = l2_normalize(alpha_embedding, axis=axis)
        normalized_beta = l2_normalize(beta_embedding, axis=axis)
        distance = find_euclidean_distance(normalized_alpha, normalized_beta)
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    return np.round(distance, 6)

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def find_image_hash(file_path: str) -> str:
    """
    Find the hash of given image file with its properties
        finding the hash of image content is costly operation
    Args:
        file_path (str): exact image path
    Returns:
        hash (str): digest with sha1 algorithm
    """
    import hashlib
    file_stats = os.stat(file_path)

    # some properties
    file_size = file_stats.st_size
    creation_time = file_stats.st_ctime
    modification_time = file_stats.st_mtime

    properties = f"{file_size}-{creation_time}-{modification_time}"

    hasher = hashlib.sha1()
    hasher.update(properties.encode("utf-8"))
    return hasher.hexdigest()

def detect_faces(
    face_detector,
    img,
    conf_thres_face_detect = 0.35,
    iou_thres_face_detect = 0.7,
    face_area_threshold = 3600,
    expand_percentage = 0
):
    height, width, _ = img.shape
    height_border = int(0.5 * height)
    width_border = int(0.5 * width)

    resp = []
    orig_img = img.copy()
    input_size = (640,640)
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio>model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / img.shape[0]

    resized_img = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
    det_img[:new_height, :new_width, :] = resized_img
    input_size = tuple(det_img.shape[0:2][::-1])
    det_img = cv2.resize(det_img, input_size)
    det_img = det_img.astype(np.float32)
    mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
    det_img = det_img - mean
    det_img = det_img * (1.0 / 128)
    det_img = np.transpose(det_img, (2, 0, 1))
    det_img = np.expand_dims(det_img, axis=0)

    input_name = face_detector.get_inputs()[0].name
    output_heads = face_detector.run(None, {input_name: det_img})

    input_height = det_img.shape[2]
    input_width = det_img.shape[3]
    nbg_outputs = {
        'score_8_439': output_heads[0],
        'score_16_477': output_heads[1],
        'score_32_515': output_heads[2],
        'bbox_8_440': output_heads[3],
        'bbox_16_478': output_heads[4],
        'bbox_32_516': output_heads[5],
        'kps_8_442': output_heads[6],
        'kps_16_480': output_heads[7],
        'kps_32_518': output_heads[8],
    }

    output_heads[0] = sigmoid(nbg_outputs['score_8_439'].transpose(0,2,3,1).reshape(1,12800,1))
    output_heads[1] = sigmoid(nbg_outputs['score_16_477'].transpose(0,2,3,1).reshape(1,3200,1))
    output_heads[2] = sigmoid(nbg_outputs['score_32_515'].transpose(0,2,3,1).reshape(1,800,1))

    output_heads[3] = ((nbg_outputs['bbox_8_440'] * 0.880725085735321).transpose(0,2,3,1).reshape(1,12800,4))
    output_heads[4] = ((nbg_outputs['bbox_16_478'] * 0.9315465092658997).transpose(0,2,3,1).reshape(1,3200,4))
    output_heads[5] = ((nbg_outputs['bbox_32_516'] * 1.1423660516738892).transpose(0,2,3,1).reshape(1,800,4))

    output_heads[6] = (nbg_outputs['kps_8_442'].transpose(0,2,3,1).reshape(1,12800,10))
    output_heads[7] = (nbg_outputs['kps_16_480'].transpose(0,2,3,1).reshape(1,3200,10))
    output_heads[8] = (nbg_outputs['kps_32_518'].transpose(0,2,3,1).reshape(1,800,10))

    fmc = 3 
    scores_list = []
    bboxes_list = []
    kpss_list = []
    batched = True
    center_cache = {}
    feat_stride_fpn = [8, 16, 32]
    num_anchors = 2 
    use_kps = True
    max_num=0
    metric='default'

    for idx, stride in enumerate(feat_stride_fpn):
        # If model support batch dim, take first output
        if batched:
            scores = output_heads[idx][0]
            bbox_preds = output_heads[idx + fmc][0]
            bbox_preds = bbox_preds * stride
            if use_kps:
                kps_preds = output_heads[idx + fmc * 2][0] * stride
        # If model doesn't support batching take output as is
        else:
            scores = output_heads[idx]
            bbox_preds = output_heads[idx + fmc]
            bbox_preds = bbox_preds * stride
            if use_kps:
                kps_preds = output_heads[idx + fmc * 2] * stride

        height = input_height // stride
        width = input_width // stride
        K = height * width
        key = (height, width, stride)

        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
            if num_anchors>1:
                anchor_centers = np.stack([anchor_centers]*num_anchors, axis=1).reshape( (-1,2) )
            if len(center_cache)<100:
                center_cache[key] = anchor_centers

        pos_inds = np.where(scores>=conf_thres_face_detect)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)

        if use_kps:
            kpss = distance2kps(anchor_centers, kps_preds)
            kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    bboxes = np.vstack(bboxes_list) / det_scale
    if use_kps:
        kpss = np.vstack(kpss_list) / det_scale
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det, iou_thres_face_detect)
    det = pre_det[keep, :]
    areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    valid_indices = np.where(areas > face_area_threshold)[0]
    det = det[valid_indices]
    
    if use_kps:
        kpss = kpss[order,:,:]
        kpss = kpss[keep,:,:]
    else:
        kpss = None
    if max_num > 0 and det.shape[0] > max_num:
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                                det[:, 1])
        img_center = img.shape[0] // 2, img.shape[1] // 2
        offsets = np.vstack([
            (det[:, 0] + det[:, 2]) / 2 - img_center[1],
            (det[:, 1] + det[:, 3]) / 2 - img_center[0]
        ])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        if metric=='max':
            values = area
        else:
            values = area - offset_dist_squared * 2.0  # some extra weight on the centering
        bindex = np.argsort(
            values)[::-1]  # some extra weight on the centering
        bindex = bindex[0:max_num]
        det = det[bindex, :]
        if kpss is not None:
            kpss = kpss[bindex, :]
    
    kpss = kpss[valid_indices]
    for i in range(det.shape[0]): 
        Results = {
            'orig_img':orig_img,
            'orig_shape':orig_img.shape[:2],
            'boxes':det[i],
            'keypoints': kpss[i] # pred_kpts = None
        }
        
        x, y, w, h = xyxy2xywh(Results['boxes'][:4]).tolist()
        confidence = Results['boxes'][-1:].tolist()[0]

        left_eye = None
        right_eye = None
        nose = None 
        left_mouth = None 
        right_mouth = None
        if kpss is not None:
            left_eye = Results['keypoints'][0].tolist()
            right_eye = Results['keypoints'][1].tolist()
            nose = Results['keypoints'][2].tolist()
            left_mouth = Results['keypoints'][3].tolist()
            right_mouth = Results['keypoints'][4].tolist()
            eye_distance = abs(right_eye[0] - left_eye[0])
            normalized_eye_distance = eye_distance / w if w > 0 else 0
            if normalized_eye_distance < 0.25:
                continue

        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
        facial_area = {
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'confidence': confidence,
            'left_eye': tuple(left_eye),
            'right_eye': tuple(right_eye),
            'nose': tuple(nose),
            'left_mouth': tuple(left_mouth),
            'right_mouth': tuple(right_mouth)
        }

        resp.append(facial_area)


    return [
        extract_face(
            facial_area=facial_area,
            img=orig_img,
            expand_percentage=expand_percentage,
            width_border=width_border,
            height_border=height_border,
        )
        for facial_area in resp
    ]

def extract_face(
    facial_area,
    img,
    expand_percentage,
    width_border,
    height_border,
):
    x = facial_area['x']
    y = facial_area['y']
    w = facial_area['w']
    h = facial_area['h']
    left_eye = facial_area['left_eye']
    right_eye = facial_area['right_eye']
    confidence = facial_area['confidence']
    nose = facial_area['nose']
    left_mouth = facial_area['left_mouth']
    right_mouth = facial_area['right_mouth']

    if expand_percentage > 0:
        # Compute the increase in width and height.
        increase_w = int(w * expand_percentage / 100)
        increase_h = int(h * expand_percentage / 100)
        # Compute new top-left so that the box expands symmetrically.
        new_x = max(0, x - increase_w // 2)
        new_y = max(0, y - increase_h // 2)
        # Adjust new width and height ensuring the expanded box stays within image boundaries.
        new_w = min(img.shape[1] - new_x, w + increase_w)
        new_h = min(img.shape[0] - new_y, h + increase_h)
        x, y, w, h = new_x, new_y, new_w, new_h

    detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
        
    facial_area = {
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'left_eye': left_eye,
        'right_eye': right_eye,
        'confidence': confidence,
        'nose': nose,
        'left_mouth': left_mouth,
        'right_mouth': right_mouth
    }
    
    DetectedFace = {
        'ori_img':img,
        'img':detected_face,
        'facial_area':facial_area,
        'confidence':confidence
    }
    return DetectedFace

def extract_faces(
    img_path,
    frame_name,
    detector_backend,
    crop_results_path,
    expand_percentage = 0,
    conf_thres_face_detect = 0.35,
    iou_thres_face_detect = 0.7,
    face_area_threshold = 3600,
    enforce_detection = True
):
    resp_objs = []
    if isinstance(img_path, str):
        img_obj_bgr = cv2.imread(img_path)
        ori_file_name =  os.path.splitext(os.path.basename(img_path))[0]
    else:
        img_obj_bgr = img_path
        ori_file_name = frame_name
     
    
    height, width, _ = img_obj_bgr.shape[0],img_obj_bgr.shape[1],img_obj_bgr.shape[2] 

    face_objs = detect_faces(
        face_detector=detector_backend,
        img=img_obj_bgr,
        conf_thres_face_detect = conf_thres_face_detect,
        iou_thres_face_detect = iou_thres_face_detect,
        face_area_threshold=face_area_threshold,
        expand_percentage=expand_percentage,
    )
    
    if len(face_objs) == 0 and enforce_detection is True:
        return []

    for index, face_obj in enumerate(face_objs):
        current_ori_img = face_obj['ori_img']
        current_img = face_obj['img']
        current_region = face_obj['facial_area']
        cropped_path = os.path.join(crop_results_path, '{}_{}.png'.format(ori_file_name,(index)))
        
        if current_img.shape[0] == 0 or current_img.shape[1] == 0:
            continue

        if DEBUG:
            cv2.imwrite(cropped_path, current_img)

        # cast to int for flask, and do final checks for borders
        x = max(0, int(current_region['x']))
        y = max(0, int(current_region['y']))
        w = min(width - x - 1, int(current_region['w']))
        h = min(height - y - 1, int(current_region['h']))

        facial_area = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "left_eye": current_region['left_eye'],
            "right_eye": current_region['right_eye'],
            'nose': current_region['nose'],
            'left_mouth': current_region['left_mouth'],
            'right_mouth': current_region['right_mouth']
        }

        resp_obj = {
            'face_ori_img': current_ori_img,
            "face": current_img,
            "facial_area": facial_area,
            "confidence": round(float(current_region['confidence'] or 0), 2),
            'cropped_path':cropped_path,
        }
        
        resp_objs.append(resp_obj)

    return resp_objs


# def resize_image(img, target_size):
#     """
#     Resize an image to expected size of a ml model with adding black pixels.
#     Args:
#         img (np.ndarray): pre-loaded image as numpy array
#         target_size (tuple): input shape of ml model
#     Returns:
#         img (np.ndarray): resized input image
#     """
#     factor_0 = target_size[0] / img.shape[0]
#     factor_1 = target_size[1] / img.shape[1]
#     factor = min(factor_0, factor_1)

#     dsize = (
#         int(img.shape[1] * factor),
#         int(img.shape[0] * factor),
#     )
#     img = cv2.resize(img, dsize)

#     diff_0 = target_size[0] - img.shape[0]
#     diff_1 = target_size[1] - img.shape[1]

#     # Put the base image in the middle of the padded image
#     img = np.pad(
#         img,
#         (
#             (diff_0 // 2, diff_0 - diff_0 // 2),
#             (diff_1 // 2, diff_1 - diff_1 // 2),
#             (0, 0),
#         ),
#         "constant",
#     )

#     # double check: if target image is not still the same size with target.
#     if img.shape[0:2] != target_size:
#         img = cv2.resize(img, target_size)

#     # make it 4-dimensional how ML models expect
#     img = np.array(img, dtype=np.float32)
#     img = np.expand_dims(img, axis=0)

#     if img.max() > 1:
#         img = (img.astype(np.float32) / 255.0).astype(np.float32)

#     return img

def represent(
    ori_img,
    img_path,
    img_region,
    model,
    warpAffine_path,
    detector_backend = "skip",
    expand_percentage = 0,
    max_faces = None,
):
    resp_objs = []
    target_size = [112,112]

    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
    else:
        img = img_path.copy()

    img_objs = [
        {
            "face_ori_img": ori_img,
            "face": img,
            "facial_area": {"x": img_region["x"], "y": img_region["y"], "w": img_region["w"], "h": img_region["h"], "left_eye":img_region["left_eye"], "right_eye":img_region["right_eye"], "nose":img_region["nose"], "left_mouth":img_region["left_mouth"], "right_mouth":img_region["right_mouth"]},
            "confidence": 0,
        }
    ]


    if max_faces is not None and max_faces < len(img_objs):
        # sort as largest facial areas come first
        img_objs = sorted(
            img_objs,
            key=lambda img_obj: img_obj["facial_area"]["w"] * img_obj["facial_area"]["h"],
            reverse=True,
        )
        # discard rest of the items
        img_objs = img_objs[0:max_faces]

    for index, img_obj in enumerate(img_objs):
        region = img_obj["facial_area"]
        confidence = img_obj["confidence"]
        loose_img = True
        FLIP = False
        use_CLAHE = True
        use_equalizeHist = False

        if not loose_img:
            img = img_obj["face"]
            landmark5 = np.array([
                np.array(img_region["left_eye"]) - np.array([img_region["x"], img_region["y"]]),
                np.array(img_region["right_eye"]) - np.array([img_region["x"], img_region["y"]]),
                np.array(img_region["nose"]) - np.array([img_region["x"], img_region["y"]]),
                np.array(img_region["left_mouth"]) - np.array([img_region["x"], img_region["y"]]),
                np.array(img_region["right_mouth"]) - np.array([img_region["x"], img_region["y"]])
            ], dtype=np.float32)
            # Compute similarity transform for alignment
            st = skimage.transform.SimilarityTransform()
            st.estimate(landmark5, SRC)
            img = cv2.warpAffine(img, st.params[0:2, :], (112, 112), borderValue=0.0)
        else:
            img = img_obj["face"]
            face_ori_img = img_obj['face_ori_img']

            x = img_region["x"]
            y = img_region["y"]
            w = img_region["w"]
            h = img_region["h"]
            # Compute expansion amounts
            enlarge_ratio = 0.5 
            expand_w = int(w * enlarge_ratio)
            expand_h = int(h * enlarge_ratio)
            new_x = max(0, x - expand_w // 2)
            new_y = max(0, y - expand_h // 2)
            desired_w = w + expand_w
            desired_h = h + expand_h
            img_height, img_width = face_ori_img.shape[:2]
            new_w = min(desired_w, img_width - new_x)
            new_h = min(desired_h, img_height - new_y)
            loose_img = face_ori_img[new_y:new_y+new_h, new_x:new_x+new_w]

            # --- Adjust landmark coordinates relative to the cropped loose image ---
            loose_landmarks = np.array([
                np.array(img_region["left_eye"]) - np.array([new_x, new_y]),
                np.array(img_region["right_eye"]) - np.array([new_x, new_y]),
                np.array(img_region["nose"]) - np.array([new_x, new_y]),
                np.array(img_region["left_mouth"]) - np.array([new_x, new_y]),
                np.array(img_region["right_mouth"]) - np.array([new_x, new_y])
            ], dtype=np.float32)

            # Compute similarity transform for alignment
            st = skimage.transform.SimilarityTransform()
            st.estimate(loose_landmarks, SRC)
            img = cv2.warpAffine(loose_img, st.params[0:2, :], (112, 112), borderValue=0.0)

        if use_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            if len(img.shape) == 3 and img.shape[2] == 3:
                # Convert to HSV color space and apply CLAHE on the V-channel
                use_hsv = False
                use_ycr = True
                if use_hsv:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(hsv)
                    v = clahe.apply(v)
                    hsv = cv2.merge((h, s, v))
                    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                if use_ycr:
                    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                    y, cr, cb = cv2.split(ycrcb)
                    y = clahe.apply(y)
                    ycrcb = cv2.merge((y, cr, cb))
                    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                img = clahe.apply(img)
                
        if use_equalizeHist:
            if len(img.shape) == 3 and img.shape[2] == 3:
                use_hsv = False
                use_ycr = True
                if use_hsv:
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(hsv)
                    v = cv2.equalizeHist(v)
                    hsv = cv2.merge((h, s, v))
                    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                if use_ycr:
                    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                    y, cr, cb = cv2.split(ycrcb)
                    y = cv2.equalizeHist(y)
                    ycrcb = cv2.merge((y, cr, cb))
                    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                img = cv2.equalizeHist(img)

        if DEBUG:
            cv2.imwrite(warpAffine_path, img)

        img_1 = img.copy()

        input_mean = input_std = 127.5
        img_1 = ((img_1 - input_mean) / input_std).astype(np.float32)
        img_1 = np.transpose(img_1, (2, 0, 1))
        img_1 = np.expand_dims(img_1, 0)
        if FLIP:
            img_2 = np.flip(img_1, axis=3)
        

        input_name = model.get_inputs()[0].name
        output1, output2 = None, None 
        output1 = model.run(None, {input_name: img_1})[0]
        
        if FLIP:
            output2 = model.run(None, {input_name: img_2})[0]
        if FLIP:
            embedding = output1 + output2 
        else:
            embedding = output1.copy()

        resp_objs.append(
            {
                "embedding": embedding,
                "facial_area": img_region, # region, # "facial_area": region,
                "face_confidence": confidence,
            }
        )

    return resp_objs

def find_bulk_embeddings(
    employees,
    detector_backend,
    facial_recognition,
    crop_results_path,
    warpAffine_path,
    expand_percentage: int = 0,
    conf_thres_face_detect = 0.35,
    iou_thres_face_detect = 0.7,
    face_area_threshold = 3600
):
    representations = []
    for employee in employees:
        file_hash = find_image_hash(employee)
        img_objs = extract_faces(
            img_path=employee,
            frame_name = None,
            detector_backend=detector_backend,
            crop_results_path = crop_results_path,
            expand_percentage=expand_percentage,
            conf_thres_face_detect = conf_thres_face_detect,
            iou_thres_face_detect = iou_thres_face_detect,
            face_area_threshold= face_area_threshold
        )
        
        if len(img_objs) == 0:
            representations.append(
                {
                    "identity": employee,
                    "hash": file_hash,
                    "embedding": None,
                    "target_x": 0,
                    "target_y": 0,
                    "target_w": 0,
                    "target_h": 0,
                    "target_left_eye":(0,0),
                    "target_right_eye":(0,0),
                    "target_nose":(0,0),
                    "target_left_mouth":(0,0),
                    "target_right_mouth":(0,0),
                }
            )
        else:

            for index, img_obj in enumerate(img_objs):
                ori_img = img_obj['face_ori_img']
                img_content = img_obj["face"]
                img_region = img_obj["facial_area"]
                
                ori_file_name =  os.path.splitext(os.path.basename(employee))[0]
                warpAffined_img = os.path.join(warpAffine_path, '{}_{}.png'.format(ori_file_name,(index)))
                embedding_obj = represent(
                    ori_img = ori_img,
                    img_path=img_content,
                    img_region = img_region,
                    model=facial_recognition,
                    warpAffine_path = warpAffined_img,
                    detector_backend="skip"
                )

                img_representation = embedding_obj[0]["embedding"]
                representations.append(
                    {
                        "identity": employee,
                        "hash": file_hash,
                        "embedding": img_representation,
                        "target_x": img_region["x"],
                        "target_y": img_region["y"],
                        "target_w": img_region["w"],
                        "target_h": img_region["h"],
                        "target_left_eye":img_region["left_eye"],
                        "target_right_eye":img_region["right_eye"],
                        "target_nose":img_region["nose"],
                        "target_left_mouth":img_region["left_mouth"],
                        "target_right_mouth":img_region["right_mouth"],
                        "cropped_path":img_obj['cropped_path']
                    }
                )

    return representations

def main(**args):

    if not os.path.exists(args['opts_dir']): os.makedirs(args['opts_dir'])
    crop_results_path = os.path.join(args['opts_dir'], 'crop_res') 
    mp4_res = os.path.join(args['opts_dir'], '{}.mp4'.format(os.path.basename(args['opts_dir'])))
    video_frames = os.path.join(args['opts_dir'], 'video_frames') 
    warpAffine_path = os.path.join(args['opts_dir'], 'warpAffine_res')
    res_frames = os.path.join(args['opts_dir'], 'res_frames')
    if DEBUG:
        if not os.path.exists(crop_results_path): os.makedirs(crop_results_path)
        if not os.path.exists(video_frames): os.makedirs(video_frames)
        if not os.path.exists(res_frames): os.makedirs(res_frames)
        if not os.path.exists(warpAffine_path): os.makedirs(warpAffine_path)
        
    file_parts = [
        "ds",
        "model",
        'MobileFaceNet',
        "detector",
        'scrfd',
        "expand",
        str(args['expand_percentage']),
    ]

    file_name = "_".join(file_parts) + ".pkl"
    file_name = file_name.replace("-", "").lower()
    datastore_path = os.path.join(args['db_path'], file_name)

    representations = []

    # required columns for representations
    df_cols = {
        "identity",
        "hash",
        "embedding",
        "target_x",
        "target_y",
        "target_w",
        "target_h",
        "target_left_eye",
        "target_right_eye",
        "target_nose",
        "target_left_mouth",
        "target_right_mouth"
    }

    # Ensure the proper pickle file exists
    if not os.path.exists(datastore_path):
        with open(datastore_path, "wb") as f:
            pickle.dump([], f, pickle.HIGHEST_PROTOCOL)

    # Load the representations from the pickle file
    with open(datastore_path, "rb") as f:
        representations = pickle.load(f)

    for i, current_representation in enumerate(representations):
        missing_keys = df_cols - set(current_representation.keys())
        if len(missing_keys) > 0:
            raise ValueError(
                f"{i}-th item does not have some required keys - {missing_keys}."
                f"Consider to delete {datastore_path}"
            )
    # Get the list of images on storage
    storage_images = {os.path.join(args['db_path'], img) for img in os.listdir(args['db_path']) if img.lower().endswith(('.png', '.jpg'))}

    if len(storage_images) == 0 and args['refresh_database'] is True:
        raise ValueError(f"No item found in {args['db_path']}")
    if len(representations) == 0 and args['refresh_database'] is False:
        raise ValueError(f"Nothing is found in {datastore_path}")

    must_save_pickle = False
    new_images, old_images, replaced_images = set(), set(), set()

    # Enforce data consistency amongst on disk images and pickle file
    if args['refresh_database']:
        # embedded images
        pickled_images = {
            representation["identity"] for representation in representations
        }

        new_images = storage_images - pickled_images  # images added to storage
        old_images = pickled_images - storage_images  # images removed from storage

        # detect replaced images
        for current_representation in representations:
            identity = current_representation["identity"]
            if identity in old_images:
                continue
            alpha_hash = current_representation["hash"]
            beta_hash = find_image_hash(identity)
            if alpha_hash != beta_hash:
                replaced_images.add(identity)

    if (len(new_images) > 0 or len(old_images) > 0 or len(replaced_images) > 0):
        print(
            f"Found {len(new_images)} newly added image(s)"
            f", {len(old_images)} removed image(s)"
            f", {len(replaced_images)} replaced image(s).")

    # append replaced images into both old and new images. these will be dropped and re-added.
    new_images.update(replaced_images)
    old_images.update(replaced_images)

    if len(old_images) > 0:
        representations = [rep for rep in representations if rep["identity"] not in old_images]
        must_save_pickle = True

    face_model = build_onnx_model(args['facial_recognition'])
    detector_backend = build_onnx_model(args['detector_backend'])

    if len(new_images) > 0:
        representations += find_bulk_embeddings(
            employees=new_images,
            detector_backend= detector_backend,
            facial_recognition = face_model,
            crop_results_path = crop_results_path,
            warpAffine_path = warpAffine_path,
            expand_percentage=args['expand_percentage'],
            conf_thres_face_detect = args['conf_thres_face_detect'],
            iou_thres_face_detect = args['iou_thres_face_detect'],
            face_area_threshold=args['face_area_threshold']
        )  # add new images

        must_save_pickle = True

    if must_save_pickle:
        with open(datastore_path, "wb") as f:
            pickle.dump(representations, f, pickle.HIGHEST_PROTOCOL)
        
    capture = cv2.VideoCapture(args['video_path'])
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
    visual_writer = cv2.VideoWriter(mp4_res, fourcc, fps, (width, height))
    frame_id = 0   
    frames = []
    all_frames_results = []  # Initialize list to store all frame results

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frame_name = f'frame_{frame_id:04d}'
        frame_filename = os.path.join(video_frames, f"{frame_name}.png")
        res_filename = os.path.join(res_frames, f"{frame_name}.png")
        if DEBUG:
            cv2.imwrite(frame_filename, frame)

        frames.append(frame)
        source_objs = extract_faces(
            img_path=frame,
            frame_name = frame_name,
            detector_backend=detector_backend,
            crop_results_path = crop_results_path,
            expand_percentage=args['expand_percentage'],
            conf_thres_face_detect = args['conf_thres_face_detect'],
            iou_thres_face_detect = args['iou_thres_face_detect'],
            face_area_threshold= args['face_area_threshold']
        )

        df = pd.DataFrame(representations)
        resp_obj = []
        all_results = []  # List to store all comparison results
        frame_results = []

        vis_frame = frame.copy()
        if source_objs:
            for source_obj_index, source_obj in enumerate(source_objs):
                source_ori_img = source_obj['face_ori_img']
                source_img = source_obj["face"]
                source_region = source_obj["facial_area"]
            
                # Draw bounding box
                x, y = source_region["x"], source_region["y"]
                w, h = source_region["w"], source_region["h"]
                left_eye, right_eye, nose, left_mouth, right_mouth = source_region["left_eye"], source_region["right_eye"], source_region["nose"], source_region["left_mouth"], source_region["right_mouth"]

                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if args.get('draw_kps', True):
                    # Draw keypoints if draw_kps is True
                    keypoints = [left_eye, right_eye, nose, left_mouth, right_mouth]
                    for point in keypoints:
                        if point is not None:
                            center = (int(round(point[0])), int(round(point[1])))
                            cv2.circle(vis_frame, center, 0, (255, 0, 0), 2)

                source_frame_idx = os.path.splitext(os.path.basename(source_obj['cropped_path']))[0]
                
                ori_file_name =  os.path.splitext(os.path.basename(frame_filename))[0]
                warpAffined_img = os.path.join(warpAffine_path, '{}.png'.format(source_frame_idx))
            
                target_embedding_obj = represent(
                    ori_img=source_ori_img,
                    img_path=source_img,
                    img_region = source_region,
                    model=face_model,
                    warpAffine_path=warpAffined_img,
                    detector_backend="skip"
                )

                result_df = df.copy() 
                result_df["source_x"] = source_region["x"]
                result_df["source_y"] = source_region["y"]
                result_df["source_w"] = source_region["w"]
                result_df["source_h"] = source_region["h"]
                result_df["source_left_eye"] = [source_region["left_eye"]] * len(result_df)
                result_df["source_right_eye"] = [source_region["right_eye"]] * len(result_df)
                result_df["source_nose"] = [source_region["nose"]] * len(result_df)
                result_df["source_left_mouth"] = [source_region["left_mouth"]] * len(result_df)
                result_df["source_right_mouth"] = [source_region["right_mouth"]] * len(result_df)
                distances = []
                comparisons = []
                # Keep track of face indices per database image
                face_indices = {}

                for index, instance in df.iterrows():
                    database_image = os.path.basename(instance['identity'])
                    # Initialize or increment face index for this database image
                    if database_image not in face_indices:
                        face_indices[database_image] = 0
                    current_face_index = face_indices[database_image]
                    face_indices[database_image] += 1

                    source_representation = instance["embedding"]
                    if source_representation is None:
                        distances.append(float("inf"))  # no representation for this image
                        continue
                    else:
                        distance_metric = 'cosine'
                        # Ensure embeddings are 1D arrays:
                        source_representation = np.squeeze(source_representation)
                        target_representation = np.squeeze(target_embedding_obj[0]["embedding"])

                        distance = find_distance(source_representation, target_representation, distance_metric)

                    comparison = {
                        'frame_id': frame_id,
                        'frame_name': source_frame_idx,
                        'query_face_id': len(frame_results),
                        'database_image': database_image,
                        'face_index': current_face_index,
                        'face_location': f"x:{instance['target_x']},y:{instance['target_y']},w:{instance['target_w']},h:{instance['target_h']}", 
                        'distance': distance,
                        'source_x': source_region["x"],
                        'source_y': source_region["y"],
                        'source_w': source_region["w"],
                        'source_h': source_region["h"],
                        'source_left_eye': source_region["left_eye"],
                        'source_right_eye': source_region["right_eye"],
                        'source_nose': source_region["nose"],
                        'source_left_mouth': source_region["left_mouth"],
                        'source_right_mouth': source_region["right_mouth"],
                        'target_x': instance['target_x'],
                        'target_y': instance['target_y'],
                        'target_w': instance['target_w'],
                        'target_h': instance['target_h'],
                        'target_left_eye': instance["target_left_eye"],
                        'target_right_eye': instance["target_right_eye"],
                        'target_nose': instance["target_nose"],
                        'target_left_mouth': instance["target_left_mouth"],
                        'target_right_mouth': instance["target_right_mouth"],

                    }
                    comparisons.append(comparison)
                    distances.append(distance)

                comparisons_df = pd.DataFrame(comparisons)
                thresholds = {"cosine": args['cosine_distanche_thres']} # thresholds = {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04}
                threshold = thresholds.get(distance_metric, 0.4)
                # comparisons_df['threshold'] = threshold
                comparisons_df['is_match'] = comparisons_df['distance'] <= threshold
                frame_results.append(comparisons_df)
                all_results.append(comparisons_df)
        else:
            print(f"Face could not be detected in {frame_name}")   

        
        if frame_results:
            frame_final_results = pd.concat(frame_results, ignore_index=True)
            all_frames_results.append(frame_final_results)


            print("\n=== Matched Results ===")
            # Get matches from the final results
            matches = frame_final_results[frame_final_results['is_match'] == True]
            if len(matches) > 0:
                # Convert distance to numeric first
                matches['distance'] = matches['distance'].astype(float)
                
                # Group matches by source face location and get the best match for each
                draw_matches = matches.groupby(['source_x', 'source_y', 'source_w', 'source_h'])
                for (source_x, source_y, source_w, source_h), group in draw_matches:
                    # Get the best match
                    best_match = group.loc[group['distance'].idxmin()]
                    distance = float(best_match['distance'])
                    if distance <= args['cosine_distanche_thres']:
                        # Draw label for best match
                        x = int(source_x)
                        y = int(source_y)
                        id_label = f"ID: {os.path.splitext(best_match['database_image'])[0]}"
                        dist_label = f"({distance:.2f})"
                        
                        # Draw ID on first line
                        cv2.putText(vis_frame, id_label, (x, y-25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  (0, 255, 0), 1)
                        # Draw distance on second line
                        cv2.putText(vis_frame, dist_label, (x, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  (0, 255, 0), 1)
                    
                # Group matches by database image
                grouped_matches = matches.groupby('database_image')
                for db_image, group in grouped_matches:
                    print(f"\nSource image: {source_frame_idx}")
                    print(f"Database image: {db_image}")
                    print(f"Number of faces detected: {len(group)}")
                    for _, match in group.iterrows():
                        face_index = match['face_index']
                        face_location = match['face_location']
                        kps = (match['target_left_eye'], match['target_right_eye'], match['target_nose'], match['target_left_mouth'], match['target_right_mouth'])
                        distance = float(match['distance'])
                        is_match = match['is_match']
                        
                        print(f"  Face #{face_index}:")
                        print(f"    Location: {face_location}")
                        print(f"    KPS: {kps}")
                        print(f"    Distance: {distance:.4f}")
                        print(f"    Is Match: {is_match}")
                        print("    ---")
            else:
                print(f"\nSource image: {frame_name} No matches found!")
            print("==================")
        
        
        
        if all_frames_results:
            final_results = pd.concat(all_frames_results, ignore_index=True)
            results_dir = os.path.join(args['opts_dir'], 'csv_results')
            os.makedirs(results_dir, exist_ok=True)
            output_filename = f"face_comparison_results_{os.path.basename(args['video_path']).split('.')[0]}.csv"
            output_path = os.path.join(results_dir, output_filename)
            final_results.to_csv(output_path, index=False)

            matched_results = final_results[final_results['is_match'] == True]
            if not matched_results.empty:
                true_filename = f"true_{os.path.basename(args['video_path']).split('.')[0]}.csv"
                true_path = os.path.join(results_dir, true_filename)
                matched_results.to_csv(true_path, index=False)

        frame_id += 1
        if DEBUG:
            cv2.imwrite(res_filename, vis_frame)
        visual_writer.write(vis_frame)

    capture.release()
    visual_writer.release() 

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    # Load file
    parser.add_argument("--detector_backend", type=str,default="scrfd.onnx", \
                        help='path to detection model')
    parser.add_argument("--facial_recognition", type=str,default="MobileFaceNet.onnx", \
                        help='path to face recognition model')
    parser.add_argument("--db_path", type=str, default="./database", \
                        help='Path to the folder containing image files. All detected faces in the database will be considered in the decision-making process')
    parser.add_argument('--refresh_database',  action='store_false', default=True,\
                        help='Synchronizes the images representation (pkl) file with the directory/db files, if set to false, it will ignore any file changes inside the db_path directory (default is True)')
    parser.add_argument("--expand_percentage", type=int, default=0.0, \
                        help='expand detected facial area with a percentage (default is 0).')
    parser.add_argument("--video_path", type=str, default="test_video.mp4", \
                        help='path to target video')
    parser.add_argument('--agnostic',  action='store_true', default=False,\
                        help='class-agnostic NMS, If True, the model is agnostic to the number of classes, and all classes will be considered as one.')
    parser.add_argument('--draw_kps',  action='store_false', default=True,\
                        help='draw kps on face.')
    parser.add_argument("--conf_thres_face_detect", type=float, default=0.25, \
                        help='define conf thres for face detection')
    parser.add_argument("--iou_thres_face_detect", type=float, default=0.7, \
                        help='define iou thres for face detection')
    parser.add_argument("--cosine_distanche_thres", type=float, default=0.5, \
                        help='define cosine_distanche_thres')
    parser.add_argument("--face_area_threshold", type=int, default=3600, \
                        help='The minimum area of the detected face')
    parser.add_argument("--opts_dir", type=str, default="./video_res_onnx", \
                        help='path of outputs ')
    argspar = parser.parse_args()     

    print("\n### Test model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))