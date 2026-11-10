# -*- coding: utf-8 -*-
import os
import tempfile

import cv2
import gradio as gr
import numpy as np


def function_hw1(input_image, mode, h_slider, s_slider, l_slider, brightness, contrast):
    if input_image is None:
        raise gr.Error('请先上传一张图像。', duration=5)
    # 作业1图像处理功能待实现。
    return [input_image] * 7


def function_hw2(input_image):
    if input_image is None:
        raise gr.Error('请先上传一张图像。', duration=5)
    # 作业2图像处理功能待实现。
    return input_image


def _homography_from_four_points(source_points, destination_points):
    """手动构造并求解 8 个自由度的单应矩阵。"""
    equations = []
    values = []
    for (x, y), (u, v) in zip(source_points, destination_points):
        equations.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        values.append(u)
        equations.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.append(v)
    coefficients = np.asarray(equations, dtype=np.float64)
    solution = np.linalg.solve(coefficients, np.asarray(values, dtype=np.float64))
    return np.append(solution, 1.0).reshape(3, 3)


def _parse_quad(text, width, height):
    """读取按左上、右上、右下、左下顺序输入的四个 (x,y) 像素坐标。"""
    try:
        pairs = [part.strip() for part in text.replace('\n', ';').split(';') if part.strip()]
        points = np.asarray([[float(value.strip()) for value in pair.split(',')]
                             for pair in pairs], dtype=np.float32)
    except (ValueError, TypeError):
        raise gr.Error('坐标格式错误。请按 x1,y1; x2,y2; x3,y3; x4,y4 输入。')
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise gr.Error('请按左上、右上、右下、左下顺序输入四组有效的 x,y 坐标。')
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width) or \
       np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
        raise gr.Error(f'坐标必须落在底图范围内：x 为 0–{width - 1}，y 为 0–{height - 1}。')
    contour = points.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour.astype(np.int32)) or abs(cv2.contourArea(contour)) < 1:
        raise gr.Error('四个点必须按顺时针或逆时针顺序构成非退化凸四边形。')
    return points


def _sample_bilinear(image, x, y):
    height, width = image.shape[:2]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x0 = np.floor(np.clip(x, 0, width - 1)).astype(np.int32)
    y0 = np.floor(np.clip(y, 0, height - 1)).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = (x - x0)[..., None]
    dy = (y - y0)[..., None]
    result = ((1 - dx) * (1 - dy) * image[y0, x0] +
              dx * (1 - dy) * image[y0, x1] +
              (1 - dx) * dy * image[y1, x0] + dx * dy * image[y1, x1])
    return np.where(valid[..., None], result, 0)


def _cubic_weight(distance, a=-0.5):
    distance = np.abs(distance)
    first = ((a + 2) * distance ** 3 - (a + 3) * distance ** 2 + 1)
    second = (a * distance ** 3 - 5 * a * distance ** 2 +
              8 * a * distance - 4 * a)
    return np.where(distance <= 1, first, np.where(distance < 2, second, 0))


def _sample_bicubic(image, x, y):
    """使用 4x4 邻域和 Keys 核手动执行双三次插值。"""
    height, width = image.shape[:2]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    safe_x = np.clip(x, 0, width - 1)
    safe_y = np.clip(y, 0, height - 1)
    base_x = np.floor(safe_x).astype(np.int32)
    base_y = np.floor(safe_y).astype(np.int32)
    result = np.zeros(x.shape + (image.shape[2],), dtype=np.float64)
    total_weight = np.zeros(x.shape, dtype=np.float64)
    for offset_y in (-1, 0, 1, 2):
        iy = np.clip(base_y + offset_y, 0, height - 1)
        wy = _cubic_weight(safe_y - (base_y + offset_y))
        for offset_x in (-1, 0, 1, 2):
            ix = np.clip(base_x + offset_x, 0, width - 1)
            weight = wy * _cubic_weight(safe_x - (base_x + offset_x))
            result += image[iy, ix] * weight[..., None]
            total_weight += weight
    result /= np.maximum(total_weight[..., None], 1e-12)
    return np.where(valid[..., None], result, 0)


def _composite_frame(background, foreground, destination_points, mode):
    """透视变换前景，并仅将四边形内部覆盖到底图。"""
    background = np.asarray(background)
    foreground = np.asarray(foreground)
    if background.ndim != 3 or foreground.ndim != 3:
        raise gr.Error('底图和前景图必须是彩色图像。')
    if background.shape[2] == 4:
        background = background[:, :, :3]
    if foreground.shape[2] == 4:
        foreground = foreground[:, :, :3]
    background = np.ascontiguousarray(background.astype(np.uint8))
    foreground = np.ascontiguousarray(foreground.astype(np.uint8))
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]
    destination_points = np.asarray(destination_points, dtype=np.float32)
    source_points = np.asarray([[0, 0], [fg_w - 1, 0],
                                [fg_w - 1, fg_h - 1], [0, fg_h - 1]], dtype=np.float64)
    matrix = _homography_from_four_points(source_points, destination_points)
    flag_by_mode = {
        'Nearest': cv2.INTER_NEAREST,
        'Bilinear': cv2.INTER_LINEAR,
        'Bicubic': cv2.INTER_CUBIC,
    }
    if mode not in flag_by_mode:
        raise gr.Error('请选择 Nearest、Bilinear 或 Bicubic 插值方式。')

    # OpenCV 透视变换及三种插值 flag；双线性/双三次结果随后由下方手写采样器生成。
    warped = cv2.warpPerspective(foreground, matrix, (bg_w, bg_h),
                                  flags=flag_by_mode[mode],
                                  borderMode=cv2.BORDER_CONSTANT)
    mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    polygon = np.rint(destination_points).astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, 255)
    if mode != 'Nearest':
        inverse = np.linalg.inv(matrix)
        min_x = max(0, int(np.floor(destination_points[:, 0].min())))
        max_x = min(bg_w - 1, int(np.ceil(destination_points[:, 0].max())))
        min_y = max(0, int(np.floor(destination_points[:, 1].min())))
        max_y = min(bg_h - 1, int(np.ceil(destination_points[:, 1].max())))
        grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x + 1),
                                     np.arange(min_y, max_y + 1))
        homogeneous = inverse @ np.stack((grid_x.ravel(), grid_y.ravel(),
                                          np.ones(grid_x.size)))
        denominator = homogeneous[2]
        safe = np.abs(denominator) > 1e-12
        source_x = np.zeros_like(denominator)
        source_y = np.zeros_like(denominator)
        source_x[safe] = homogeneous[0, safe] / denominator[safe]
        source_y[safe] = homogeneous[1, safe] / denominator[safe]
        source_x = source_x.reshape(grid_x.shape)
        source_y = source_y.reshape(grid_y.shape)
        if mode == 'Bilinear':
            sampled = _sample_bilinear(foreground.astype(np.float64), source_x, source_y)
        else:
            sampled = _sample_bicubic(foreground.astype(np.float64), source_x, source_y)
        region = warped[min_y:max_y + 1, min_x:max_x + 1]
        region[:] = np.clip(sampled, 0, 255).astype(np.uint8)

    result = background.copy()
    inside = mask != 0
    result[inside] = warped[inside]
    return result


def _feature_points(gray, quad):
    """提取屏幕四边形内部特征，供单应性估计使用。"""
    height, width = gray.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.rint(quad).astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, 255)
    # 略微收缩掩膜，尽量避免跟踪到屏幕外背景。
    mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
    points = cv2.goodFeaturesToTrack(
        gray, maxCorners=120, qualityLevel=0.01, minDistance=5,
        blockSize=7, mask=mask, useHarrisDetector=False)
    if points is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return points.astype(np.float32)


def _forward_backward_track(previous_gray, gray, points):
    """使用正反向 LK 和更大的金字塔搜索窗口筛除不可靠跟踪点。"""
    if points is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    params = dict(winSize=(31, 31), maxLevel=4, criteria=criteria,
                  minEigThreshold=1e-5)
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, gray, points, None, **params)
    if forward is None or forward_status is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, previous_gray, forward, None, **params)
    if backward is None or backward_status is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    old = points.reshape(-1, 2)
    new = forward.reshape(-1, 2)
    round_trip_error = np.linalg.norm(backward.reshape(-1, 2) - old, axis=1)
    valid = (forward_status.reshape(-1) == 1) & (backward_status.reshape(-1) == 1)
    valid &= np.isfinite(new).all(axis=1) & (round_trip_error < 2.5)
    return old[valid], new[valid]


def _valid_tracked_quad(quad, width, height):
    """检查跟踪结果仍是可用于透视变换的凸四边形。"""
    if quad is None or not np.isfinite(quad).all():
        return False
    contour = np.rint(quad).astype(np.int32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 16:
        return False
    # 允许屏幕部分离开画面，但拒绝明显失控的坐标。
    margin_x, margin_y = width * 0.25, height * 0.25
    return (np.all(quad[:, 0] >= -margin_x) and
            np.all(quad[:, 0] <= width - 1 + margin_x) and
            np.all(quad[:, 1] >= -margin_y) and
            np.all(quad[:, 1] <= height - 1 + margin_y))


def _track_screen_quad(previous_gray, gray, quad, features):
    """先用屏幕纹理估计单应性；纹理不足时退回四角的正反向 LK。"""
    # 将角点作为额外跟踪点，低纹理屏幕也有机会用角点直接跟踪。
    track_points = np.concatenate((features.reshape(-1, 1, 2),
                                   quad.reshape(-1, 1, 2)), axis=0)
    old_points, new_points = _forward_backward_track(previous_gray, gray, track_points)
    feature_count = len(features)
    tracked_quad = None

    if len(old_points) >= 4:
        matrix, inliers = cv2.findHomography(old_points, new_points,
                                             cv2.RANSAC, 4.0)
        if matrix is not None and inliers is not None:
            inlier_mask = inliers.reshape(-1).astype(bool)
            # 至少需要四个一致的对应点；对四角 LK 点则仍可估计屏幕整体运动。
            if np.count_nonzero(inlier_mask) >= 4:
                tracked_quad = cv2.perspectiveTransform(
                    quad.reshape(1, 4, 2), matrix).reshape(4, 2)

    if not _valid_tracked_quad(tracked_quad, gray.shape[1], gray.shape[0]):
        # 特征点不足或单应性不稳定时，尝试直接追踪四角。
        corner_old, corner_new = _forward_backward_track(
            previous_gray, gray, quad.reshape(-1, 1, 2))
        if len(corner_old) == 4:
            tracked_quad = corner_new.reshape(4, 2)
        else:
            tracked_quad = None

    if not _valid_tracked_quad(tracked_quad, gray.shape[1], gray.shape[0]):
        return None
    return tracked_quad.astype(np.float32)


def function_hw3(bg_image, fg_image, points_text, mode):
    if bg_image is None:
        raise gr.Error('请先上传底图。', duration=5)
    if fg_image is None:
        raise gr.Error('请先上传前景展示图。', duration=5)
    height, width = bg_image.shape[:2]
    points = _parse_quad(points_text, width, height)
    return _composite_frame(bg_image, fg_image, points, mode)


def function_hw3_video(video_file, fg_image, points_text, mode):
    """以正反向光流、屏幕特征单应性和四角回退追踪屏幕，并替换视频内容。"""
    if video_file is None:
        raise gr.Error('请先上传底图视频。')
    if fg_image is None:
        raise gr.Error('请先上传前景展示图。')
    video_path = video_file.get('name') if isinstance(video_file, dict) else video_file
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise gr.Error('无法读取所选视频文件。')
    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise gr.Error('视频没有可读取的首帧。')
    height, width = first_frame.shape[:2]
    points = _parse_quad(points_text, width, height)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    fd, output_path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (width, height))
    if not writer.isOpened():
        capture.release()
        os.remove(output_path)
        raise gr.Error('無法建立輸出影片，請確認 OpenCV 的 mp4v 編碼器可用。')

    previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    current_quad = points.astype(np.float32)
    features = _feature_points(previous_gray, current_quad)
    frame_index = 1
    try:
        frame = _composite_frame(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
                                 fg_image, current_quad, mode)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_index += 1
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            next_quad = _track_screen_quad(previous_gray, gray, current_quad, features)
            if next_quad is None:
                raise gr.Error(
                    f'第 {frame_index} 帧四角追踪失败；请检查画面是否被遮挡或屏幕已离开画面。')
            output = _composite_frame(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                                      fg_image, next_quad, mode)
            writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            current_quad = next_quad
            previous_gray = gray
            # 每帧重检屏幕纹理，避免旧特征丢失后无法恢复。
            features = _feature_points(previous_gray, current_quad)
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        capture.release()
        writer.release()
    return output_path


def function_hw4(input_image):
    if input_image is None:
        raise gr.Error('请先上传一张图像。', duration=5)
    return input_image


def function_hw5(input_image):
    if input_image is None:
        raise gr.Error('请先上传一张图像。', duration=5)
    return input_image
