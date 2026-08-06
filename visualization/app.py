import os
import sys

# -------------------------------------------------------------------------
# 0. FFmpeg Setup (Ensures Gradio/OpenCV handles video uploads smoothly)
# -------------------------------------------------------------------------
try:
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_exe
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_exe)
except ImportError:
    pass

import gradio as gr
from extract import extract_landmarks_from_video
from preprocess import process_sequence, pad_or_truncate
from vision_model import SignVisionPredictor
from llm import ContextualRescorer

# -------------------------------------------------------------------------
# 1. Paths & Model Initialization
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, 'best_convbilstm_model.pth')
MAPPING_PATH = os.path.join(BASE_DIR, 'class_mapping_100.json')

# Initialize Vision Predictor and LLM Rescorer
print("Initializing Vision Predictor...")
vision_predictor = SignVisionPredictor(model_path=MODEL_PATH, mapping_path=MAPPING_PATH)

print("Initializing Contextual Rescorer...")
llm_rescorer = ContextualRescorer()


# -------------------------------------------------------------------------
# 2. Live Pipeline Execution Function
# -------------------------------------------------------------------------
def run_realtime_pipeline(video_path, context_sentence):
    # Safely extract file path if Gradio passes a dict or object
    if isinstance(video_path, dict):
        video_path = video_path.get("path") or video_path.get("name")
    elif hasattr(video_path, "path"):
        video_path = video_path.path

    if not video_path:
        return "<div style='color: #ef4444; font-weight: bold; padding: 20px; text-align: center;'>❌ Please upload a video or record one using your webcam.</div>"
    if "___" not in context_sentence:
        return "<div style='color: #f59e0b; font-weight: bold; padding: 20px; text-align: center;'>⚠️ Please include '___' in your context sentence.</div>"

    try:
        # Step 1: Extract Keypoints
        raw_landmarks = extract_landmarks_from_video(video_path)
        if len(raw_landmarks) < 15:
            return "<div style='color: #ef4444; font-weight: bold; padding: 20px; text-align: center;'>⚠️ Video is too short (< 15 frames) to extract reliable landmarks.</div>"

        # Step 2: Preprocess Sequence
        processed_seq = process_sequence(raw_landmarks.astype("float32"))
        padded_seq = pad_or_truncate(processed_seq, max_frames=40)

        # Step 3: Run ConvBiLSTM Vision Inference
        top10_vision_predictions = vision_predictor.predict_top_k(padded_seq, top_k=10)
        raw_top1_word, raw_top1_conf = top10_vision_predictions[0]

        # Step 4: Run GPT-2 Context Rescoring
        rescored_results = llm_rescorer.rescore_candidates(context_sentence, top10_vision_predictions)
        winner = rescored_results[0]
        final_winner_word = winner["word"]

        # Step 5: Render Enhanced HTML Output Card
        html_output = f"""
        <div style="background-color: #ffffff; padding: 25px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
            <h2 style="color: #0f172a; margin-top: 0; margin-bottom: 20px; font-weight: 800; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;">🎯 Pipeline Diagnostics</h2>

            <div style="display: flex; gap: 20px; margin-bottom: 25px;">
                <div style="flex: 1; background: #f8fafc; padding: 18px; border-radius: 10px; border-left: 5px solid {'#10b981' if raw_top1_word == final_winner_word else '#ef4444'}; border: 1px solid #e2e8f0;">
                    <span style="font-size: 12px; text-transform: uppercase; color: #475569; font-weight: 800;">Raw Vision Prediction</span>
                    <div style="font-size: 24px; font-weight: 900; color: #0f172a; margin: 4px 0;">{raw_top1_word.upper()} <span style="font-size: 14px; font-weight: 700; color: #64748b;">({raw_top1_conf:.2f}%)</span></div>
                    <span style="font-size: 13px; font-weight: 700; color: {'#10b981' if raw_top1_word == final_winner_word else '#ef4444'};">
                        {'✅ Exact Match' if raw_top1_word == final_winner_word else '⚠️ Visual Ambiguity Detected'}
                    </span>
                </div>

                <div style="flex: 1; background: #ecfdf5; padding: 18px; border-radius: 10px; border-left: 5px solid #10b981; border: 1px solid #a7f3d0;">
                    <span style="font-size: 12px; text-transform: uppercase; color: #065f46; font-weight: 800;">LLM Contextual Winner</span>
                    <div style="font-size: 24px; font-weight: 900; color: #065f46; margin: 4px 0;">{final_winner_word.upper()} 🏆</div>
                    <span style="font-size: 13px; font-weight: 700; color: #047857;">✅ Context Resolved</span>
                </div>
            </div>

            <h3 style="color: #0f172a; margin-bottom: 15px; font-size: 18px; font-weight: 800;">📊 Candidate Ranking Breakdown</h3>
            <div style="border-radius: 8px; border: 1px solid #cbd5e1; overflow: hidden;">
                <table style="width: 100%; border-collapse: collapse; text-align: left; background-color: white;">
                    <thead>
                        <tr style="background-color: #f1f5f9; border-bottom: 2px solid #cbd5e1;">
                            <th style="padding: 12px 16px; color: #0f172a !important; font-weight: 800; font-size: 13px; text-transform: uppercase;">Rank</th>
                            <th style="padding: 12px 16px; color: #0f172a !important; font-weight: 800; font-size: 13px; text-transform: uppercase;">Predicted Sign</th>
                            <th style="padding: 12px 16px; color: #0f172a !important; font-weight: 800; font-size: 13px; text-transform: uppercase;">Vision Confidence</th>
                            <th style="padding: 12px 16px; color: #0f172a !important; font-weight: 800; font-size: 13px; text-transform: uppercase;">Context Loss <span style="font-size: 11px; font-weight: normal; color: #475569 !important;">(Lower=Better)</span></th>
                        </tr>
                    </thead>
                    <tbody style="font-size: 14px;">
            """

        for rank, item in enumerate(rescored_results, 1):
            is_winner = item["word"] == final_winner_word
            row_style = "background-color: #f0fdf4;" if is_winner else (
                "background-color: #ffffff;" if rank % 2 != 0 else "background-color: #f8fafc;")
            badge = "🥇" if is_winner else f"<span style='color: #64748b !important; font-weight: 700;'>#{rank}</span>"
            word_color = "color: #047857 !important; font-weight: 800;" if is_winner else "color: #0f172a !important; font-weight: 700;"

            html_output += f"""
                        <tr style="{row_style}">
                            <td style="padding: 12px 16px; border-bottom: 1px solid #e2e8f0; font-weight: bold; color: #0f172a !important;">{badge}</td>
                            <td style="padding: 12px 16px; border-bottom: 1px solid #e2e8f0; {word_color}">{item['word'].upper()}</td>
                            <td style="padding: 12px 16px; border-bottom: 1px solid #e2e8f0; color: #334155 !important; font-weight: 600;">{item['vision_conf']:.2f}%</td>
                            <td style="padding: 12px 16px; border-bottom: 1px solid #e2e8f0; font-weight: 700; color: {'#047857' if is_winner else '#475569'} !important;">{item['context_loss']:.2f}</td>
                        </tr>
            """

        html_output += """
                    </tbody>
                </table>
            </div>
        </div>
        """
        return html_output

    except Exception as e:
        return f"<div style='color: #ef4444; font-weight: bold; padding: 20px; border: 1px solid #fca5a5; background-color: #fef2f2; border-radius: 8px;'>❌ Error during execution: {str(e)}</div>"


# -------------------------------------------------------------------------
# 3. Gradio Interface Layout
# -------------------------------------------------------------------------
css = """
body, .gradio-container {
    background-color: #f1f5f9 !important;
    max-width: 100% !important;
    padding: 20px 40px !important;
}

/* Force video player to maintain natural aspect ratio */
video {
    object-fit: contain !important;
    max-height: 500px !important;
    width: auto !important;
    margin: 0 auto !important;
    display: block !important;
}

/* Force the main title to be dark and visible */
.main-title-container h1 {
    color: #0f172a !important;
    font-weight: 900 !important;
    margin-bottom: 5px !important;
    font-size: 2.2rem !important;
}
.main-title-container p {
    color: #475569 !important;
    font-size: 1.1rem !important;
}

/* Button Styling */
.gr-button-primary {
    background: linear-gradient(to right, #4f46e5, #6366f1) !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.4) !important;
    transition: all 0.2s ease !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 8px -1px rgba(79, 70, 229, 0.5) !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=css) as demo:
    gr.Markdown(
        """
        # 🤟 Real-Time Multi-Modal Sign Language Decoder
        Select **Upload** or **Webcam** below to capture a sign video. The system extracts landmarks, generates vision candidates, and rescores them via LLM context.
        """,
        elem_classes="main-title-container"
    )

    with gr.Row(equal_height=True):
        # Left Side: Video & Sentence Inputs
        with gr.Column(scale=1):
            video_input = gr.Video(
                sources=["upload", "webcam"],
                label="Video Input Method (Upload file or use Webcam)",
                include_audio=False
            )
            context_input = gr.Textbox(
                label="Context Sentence (Use '___' for the signed word)",
                value="The sun was shining directly in my eyes, so I wore a ___ on my head.",
                placeholder="e.g., The sun was shining directly in my eyes, so I wore a ___ on my head.",
                lines=2
            )
            btn = gr.Button("🚀 Decode Sign Sequence", variant="primary", size="lg")

        # Right Side: Interactive Output Render
        with gr.Column(scale=1):
            results_output = gr.HTML(
                value="<div style='background: #ffffff; padding: 50px 20px; text-align: center; border-radius: 12px; border: 2px dashed #cbd5e1; color: #64748b; font-weight: 500; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1); height: 100%; display: flex; align-items: center; justify-content: center;'>Upload a video and click 'Decode Sign Sequence' to view real-time results here.</div>",
                label="System Analysis Output"
            )

    btn.click(
        fn=run_realtime_pipeline,
        inputs=[video_input, context_input],
        outputs=results_output
    )

if __name__ == "__main__":
    demo.launch(share=False)