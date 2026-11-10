from __future__ import annotations

import gradio as gr
from UI import *
from functions import *


HTML_DESCRIPTION = '''
<div align=center>
<h1 style="font-weight: 900; margin-bottom: 7px;">图像处理网页演示工具</h1>
<p>使用方式：在浏览器中打开 http://127.0.0.1:8088/</p>
<p>作者：图小像</p>
<p>学号：2400000000</p>
</div>
'''
MD_DESCRIPTION = '''
## 此网页演示提供以下图像处理工具：
- 作业1：图像色相/饱和度/亮度/对比度调整工具
- 作业2：基于 GAN 的图像生成编辑工具
- 作业3：手动输入四边形顶点的透视变换、屏幕内容替换及视频四角光流追踪
- 作业4：图像去噪工具
- 作业5：图像提亮工具
'''


def main():
    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(HTML_DESCRIPTION)
        gr.Markdown(MD_DESCRIPTION)
        with gr.Tabs():
            with gr.TabItem('作业1：图像色相/饱和度/亮度/对比度调整工具'):
                create_demo_hw1(function_hw1)
            with gr.TabItem('作业2：基于 GAN 的图像生成编辑工具'):
                create_demo_hw2(function_hw2)
            with gr.TabItem('作业3：透视变换与屏幕内容替换'):
                create_demo_hw3(function_hw3, function_hw3_video)
            with gr.TabItem('作业4：图像去噪工具'):
                create_demo_hw4(function_hw4)
            with gr.TabItem('作业5：图像提亮工具'):
                create_demo_hw5(function_hw5)
    demo.launch(server_port=8088)


if __name__ == '__main__':
    main()
