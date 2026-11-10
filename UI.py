# -*- coding: utf-8 -*-
from __future__ import annotations

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
上传底图和展示图，按底图像素坐标输入四边形顶点，顺序为左上、右上、右下、左下；
坐标格式示例：`120,80; 500,90; 490,360; 110,350`。选择最近邻、手动双线性或手动双三次插值后运行。''')
        with gr.Row():
            with gr.Column():
                bg_img = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='底图')
                fg_img = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='前景展示图')
                points = gr.Textbox(label='底图四边形顶点 (x,y)',
                                    placeholder='120,80; 500,90; 490,360; 110,350', lines=2)
                mode = gr.Radio(choices=['Nearest', 'Bilinear', 'Bicubic'], value='Nearest',
                                label='插值方式（Bilinear/Bicubic 为手动实现）')
                run_btn = gr.Button('透视变换并替换屏幕')
            with gr.Column():
                output_image = gr.Image(type='numpy', label='处理结果', interactive=False)
        run_btn.click(fn=process, inputs=[bg_img, fg_img, points, mode], outputs=[output_image])

        if process_video is not None:
            gr.Markdown('### 视频扩展：四角光流追踪\n上传含屏幕的底图视频，坐标填写视频首帧的四角；程序以稀疏光流逐帧追踪四角。')
            with gr.Row():
                input_video = gr.Video(label='底图视频')
                output_video = gr.Video(label='替换后的视频', interactive=False)
            video_btn = gr.Button('追踪屏幕并生成视频')
            video_btn.click(fn=process_video, inputs=[input_video, fg_img, points, mode], outputs=[output_video])
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
