# -*- coding: utf-8 -*-
from __future__ import annotations

import cv2
import gradio as gr


def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业一：图像色相/饱和度/亮度/对比度调整')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')
                mode = gr.Radio(['色相/饱和度', '亮度/对比度'], label='模式', value='色相/饱和度')
                h_slider = gr.Slider(-1, 1, label='色相', value=0, step=0.01)
                s_slider = gr.Slider(-1, 1, label='饱和度', value=0, step=0.01)
                l_slider = gr.Slider(-1, 1, label='亮度', value=0, step=0.01)
                brightness = gr.Slider(-1, 1, label='亮度', value=0, step=0.01, visible=False)
                contrast = gr.Slider(-1, 1, label='对比度', value=0, step=0.01, visible=False)
            with gr.Column():
                output_image = gr.Image(type='numpy', label='输出图像', interactive=False)
                with gr.Row():
                    r_channel = gr.Image(type='numpy', label='R 通道', interactive=False)
                    g_channel = gr.Image(type='numpy', label='G 通道', interactive=False)
                    b_channel = gr.Image(type='numpy', label='B 通道', interactive=False)
                with gr.Row():
                    h_channel = gr.Image(type='numpy', label='H 通道', interactive=False)
                    s_channel = gr.Image(type='numpy', label='S 通道', interactive=False)
                    l_channel = gr.Image(type='numpy', label='L 通道', interactive=False)
        inputs = [input_image, mode, h_slider, s_slider, l_slider, brightness, contrast]
        outputs = [output_image, r_channel, g_channel, b_channel, h_channel, s_channel, l_channel]
        input_image.change(fn=process, inputs=inputs, outputs=outputs)

        def update_mode(selected_mode):
            hsv_visible = selected_mode == '色相/饱和度'
            return tuple(gr.update(visible=hsv_visible) for _ in range(3)) + tuple(
                gr.update(visible=not hsv_visible) for _ in range(2))
        mode.change(update_mode, inputs=[mode], outputs=[h_slider, s_slider, l_slider, brightness, contrast])
        for slider in (h_slider, s_slider, l_slider, brightness, contrast):
            slider.release(fn=process, inputs=inputs, outputs=outputs)
    return demo


def create_demo_hw2(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业二：基于 GAN 的图像生成编辑工具')
        with gr.Row():
            input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')
            output_image = gr.Image(type='numpy', label='输出图像', interactive=False)
        gr.Button('运行').click(fn=process, inputs=[input_image], outputs=[output_image])
    return demo


def create_demo_hw3(process, process_video=None):
    with gr.Blocks() as demo:
        gr.Markdown('''## 作业三：透视变换与屏幕内容替换
上传底图和展示图，然后直接在底图上依次点击屏幕的左上、右上、右下、左下四个角；
无需手动输入像素坐标。选择最近邻、手动双线性或手动双三次插值后运行。''')
        with gr.Row():
            with gr.Column():
                bg_img = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy',
                                  label='底图（依次点击屏幕四角）', interactive=True)
                point_state = gr.State([])
                point_status = gr.Markdown('请先上传底图，再依次点击：左上 → 右上 → 右下 → 左下。')
                with gr.Row():
                    points = gr.Textbox(label='已选四角坐标 (x,y)', interactive=False,
                                        placeholder='点击底图选取四个角后自动填入', lines=2)
                    clear_points = gr.Button('清除并重选四角')
                fg_img = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='前景展示图')
                mode = gr.Radio(choices=['Nearest', 'Bilinear', 'Bicubic'], value='Nearest',
                                label='插值方式（Bilinear/Bicubic 为手动实现）')
                run_btn = gr.Button('透视变换并替换屏幕')
            with gr.Column():
                output_image = gr.Image(type='numpy', label='处理结果', interactive=False)
        run_btn.click(fn=process, inputs=[bg_img, fg_img, points, mode], outputs=[output_image])

        def add_corner(image, selected_points, evt: gr.SelectData, target='底图'):
            selected_points = selected_points or []
            if image is None:
                return selected_points, '', '请先上传' + target + '。'
            if len(selected_points) >= 4:
                return selected_points, '; '.join(
                    f'{x},{y}' for x, y in selected_points), '四个角已选好；如需修改，请点击“清除并重选四角”。'
            try:
                x, y = evt.index[:2]
                x, y = int(round(x)), int(round(y))
            except (TypeError, ValueError):
                return selected_points, '; '.join(
                    f'{x},{y}' for x, y in selected_points), '无法读取点击位置，请在预览图上点击。'
            height, width = image.shape[:2]
            if not (0 <= x < width and 0 <= y < height):
                return selected_points, '; '.join(
                    f'{px},{py}' for px, py in selected_points), '点击位置超出图像范围，请重新点击。'
            selected_points = list(selected_points) + [[x, y]]
            point_names = ['左上', '右上', '右下', '左下']
            coordinates = '; '.join(f'{px},{py}' for px, py in selected_points)
            if len(selected_points) == 4:
                status = ('视频首帧四角已选好，可以点击“追踪屏幕并生成视频”；顺序为左上、右上、右下、左下。'
                          if target == '视频首帧' else
                          '四个角已选好，可以运行区域替换；顺序为左上、右上、右下、左下。')
            else:
                status = f'已选{point_names[len(selected_points) - 1]}角；请继续点击{point_names[len(selected_points)]}角。'
            return selected_points, coordinates, status

        def add_video_corner(image, selected_points, evt: gr.SelectData):
            return add_corner(image, selected_points, evt, '视频首帧')

        def reset_corners():
            return [], '', '已清除，请依次点击：左上 → 右上 → 右下 → 左下。'

        def reset_on_new_background(_image):
            return reset_corners()

        bg_img.select(fn=add_corner, inputs=[bg_img, point_state],
                      outputs=[point_state, points, point_status])
        bg_img.change(fn=reset_on_new_background, inputs=[bg_img],
                      outputs=[point_state, points, point_status])
        clear_points.click(fn=reset_corners, inputs=[],
                           outputs=[point_state, points, point_status])

        if process_video is not None:
            gr.Markdown('### 视频扩展：四角光流追踪\n上传含屏幕的底图视频后，系统会显示首帧预览；请在该预览图上依次点击屏幕左上、右上、右下、左下四角。视频四角坐标与上方静态底图的坐标相互独立。')
            with gr.Row():
                input_video = gr.Video(label='底图视频')
                video_frame = gr.Image(type='numpy', label='底图视频首帧（点击四角选取）', interactive=True)
            video_point_state = gr.State([])
            video_point_status = gr.Markdown('上传底图视频后，首帧将显示在这里。')
            with gr.Row():
                video_points = gr.Textbox(label='首帧已选四角坐标 (x,y)', interactive=False,
                                          placeholder='点击上方首帧预览选取四个角后自动填入', lines=2)
                clear_video_points = gr.Button('清除并重选四角')
            with gr.Row():
                output_video = gr.Video(label='替换后的视频', interactive=False)
            video_btn = gr.Button('追踪屏幕并生成视频')

            def load_video_first_frame(video_file):
                if video_file is None:
                    return None, [], '', '请先上传底图视频。'
                video_path = video_file.get('name') or video_file.get('path') if isinstance(video_file, dict) else video_file
                capture = cv2.VideoCapture(video_path)
                try:
                    if not capture.isOpened():
                        raise gr.Error('无法读取所选视频文件。')
                    ok, first_frame = capture.read()
                    if not ok:
                        raise gr.Error('视频没有可读取的首帧。')
                    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                    return frame_rgb, [], '', '请在首帧预览上依次点击：左上 → 右上 → 右下 → 左下。'
                finally:
                    capture.release()

            input_video.change(fn=load_video_first_frame, inputs=[input_video],
                               outputs=[video_frame, video_point_state, video_points, video_point_status])
            video_frame.select(fn=add_video_corner, inputs=[video_frame, video_point_state],
                               outputs=[video_point_state, video_points, video_point_status])
            clear_video_points.click(fn=reset_corners, inputs=[],
                                     outputs=[video_point_state, video_points, video_point_status])
            video_btn.click(fn=process_video,
                            inputs=[input_video, fg_img, video_points, mode], outputs=[output_video])
    return demo


def create_demo_hw4(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业四：图像去噪')
        with gr.Row():
            input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')
            output_image = gr.Image(type='numpy', label='输出图像', interactive=False)
        gr.Button('运行').click(fn=process, inputs=[input_image], outputs=[output_image])
    return demo


def create_demo_hw5(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业五：图像提亮')
        with gr.Row():
            input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')
            output_image = gr.Image(type='numpy', label='输出图像', interactive=False)
        gr.Button('运行').click(fn=process, inputs=[input_image], outputs=[output_image])
    return demo
