"""Gradio Interactive Demo for SBIR.
Supports:
- Sketchpad (direct drawing)
- Image file upload (hand-drawn sketches on paper)
- Model selection: Baseline (E1), Frozen CLIP (E2), Prompt Learning (E4)
- Top-K retrieval display with cosine scores and latency
"""

import time
from typing import List, Tuple
import numpy as np
from PIL import Image

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


def mock_retrieval(sketch_image, model_choice: str, top_k: int = 5):
    """Placeholder retrieval function using mock data."""
    start_time = time.time()
    # Dummy results
    dummy_results = []
    for i in range(top_k):
        # Create a colored placeholder image
        img = Image.new("RGB", (256, 256), color=(50 + i * 30, 100, 150))
        score = 0.95 - i * 0.08
        dummy_results.append((img, f"Rank #{i+1} | Score: {score:.3f}"))
        
    latency = (time.time() - start_time) * 1000
    return dummy_results, f"Inference completed in {latency:.1f} ms using [{model_choice}]"


def build_app():
    if not GRADIO_AVAILABLE:
        print("[ERROR] Gradio is not installed. Please run `pip install gradio`.")
        return None

    with gr.Blocks(title="SBIR - Sketch-Based Image Retrieval") as demo:
        gr.Markdown("# 🎨 Sketch-Based Image Retrieval (SBIR)")
        gr.Markdown("Đồ án môn học CS406 – Xử lý ảnh và Ứng dụng")

        with gr.Row():
            with gr.Column(scale=1):
                sketch_input = gr.Sketchpad(
                    label="Vẽ hình phác thảo (Sketchpad)",
                    type="pil",
                    image_mode="L"
                )
                model_selector = gr.Radio(
                    choices=["E1: Baseline (Canny+HOG+BoVW)", "E2: Frozen CLIP ViT-B/32", "E4: CLIP + Deep Prompt Learning"],
                    value="E4: CLIP + Deep Prompt Learning",
                    label="Chọn mô hình truy vấn"
                )
                top_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Số lượng kết quả (Top-K)")
                search_btn = gr.Button("🔍 Tìm kiếm ảnh", variant="primary")

            with gr.Column(scale=2):
                status_box = gr.Textbox(label="Trạng thái & Thời gian truy vấn", interactive=False)
                gallery_output = gr.Gallery(label="Top-K Ảnh tương tự tìm được", columns=5, height="auto")

        search_btn.click(
            fn=mock_retrieval,
            inputs=[sketch_input, model_selector, top_k_slider],
            outputs=[gallery_output, status_box]
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    if app:
        app.launch(share=False)
