import gc
import tqdm
import gradio as gr
import ffmpeg
import numpy as np
import time
import torch
import os
import dotenv
import utils
import json

dotenv.load_dotenv()


def process(files, device, model, lang, allign, diarization, batch_size, output_format, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Loading models...")
    import whisperx

    if files is None:
        raise gr.Error("Please upload a file to transcribe")

    # if lang == "":
    #     raise gr.Error("Please provide a language hint")

    results = []
    tmp_results = []
    whisper_model = whisperx.load_model(model, device=device)

    for file in tqdm.tqdm(files, desc="Transcribing", position=0, leave=True):
        result = whisper_model.transcribe(file.name, batch_size=batch_size)
        results.append((result, file.name))

    del whisper_model
    gc.collect()
    torch.cuda.empty_cache()

    # load whisperx model
    if allign:
        tmp_results = results

        results = []
        align_model, align_metadata = whisperx.load_align_model(model_name="WAV2VEC2_ASR_LARGE_LV60K_960H" if lang == "en" else None, language_code=lang, device=device)

        for result, audio_path in tqdm.tqdm(tmp_results, desc="Alligning", position=0, leave=True):
            input_audio = audio_path

            if align_model is not None and len(result["segments"]) > 0:
                if result.get('language') != align_metadata["language"]:
                    # load new model
                    print(f"Loading new model for {result['language']}")
                    align_model, align_metadata = whisperx.load_align_model(result["language"], device=device)
                result = whisperx.align(result["segments"], align_model, align_metadata, input_audio, device, return_char_alignments=False)
            results.append((result, audio_path))

        del align_model
        gc.collect()
        torch.cuda.empty_cache()

    if diarization:
        if os.getenv("hf_token") is None:
            print("Please provide a huggingface token to use speaker diarization")
        else:
            tmp_res = results
            results = []
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv("hf_token"), device=device)
            for result, input_audio_path in tqdm.tqdm(tmp_res, desc="Diarizing", position=0, leave=True):
                diarize_segments = diarize_model(input_audio_path, min_speakers=None, max_speakers=None)
                result = whisperx.diarize.assign_word_speakers(diarize_segments, result)
                results.append((result, input_audio_path))


    writer_args = {"max_line_width": None, "max_line_count": None, "highlight_words": False}
    for res, audio_path in tqdm.tqdm(results, desc="Writing", position=0, leave=True):
        slug = os.path.basename(audio_path).replace(os.path.basename(audio_path).split("_")[-1], "")

        if not os.path.exists(os.getcwd() + "/output/" + slug):
            os.mkdir(os.getcwd() + "/output/" + slug)

        writer = utils.get_writer(output_format, os.getcwd() + "/output/" + slug)
        writer(res, audio_path, writer_args)


with gr.Blocks() as ui:
    with gr.Tab(label="Transcribe"):
        with gr.Row():
            # # input field for audio / video file
            input_files = gr.Files(label="Input Files")

            def clear():
                return None

            with gr.Column():
                with gr.Row():
                    btn_run = gr.Button()
                    btn_reset = gr.Button(value="Reset").click(fn=clear, outputs=[input_files])

                with gr.Row():
                    # model selection dropdown
                    model = gr.Dropdown(label="Model", choices=["tiny", "base", "small", "medium", "large", "large-v2"], value="base")
                    # langaue hint input
                    lang = gr.Text(label="Language Hint", placeholder="en")

                with gr.Row():
                    with gr.Group():
                        allign = gr.Checkbox(label="Allign Text", value=True)
                        diarization = gr.Checkbox(label="Speaker Diarization")
                        with gr.Row():
                            min_speakers = gr.Number(label="Min Speakers", max_value=10, value=1, interactive=True)
                            max_speakers = gr.Number(label="Max Speakers", max_value=10, value=1, interactive=True)

                    with gr.Group():
                        # device add cuda to dropdown if available
                        device = gr.Dropdown(
                            label="Device", choices=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], value="cuda" if torch.cuda.is_available() else "cpu"
                        )
                        batch_size = gr.Slider(label="Batch Size", min_value=1, maximum=100, step=1, value=8, interactive=True)

                # output format
                output_format = gr.Dropdown(label="Output Format", choices=["all", "json", "txt", "srt", "vtt", "tsv"], value="all")

    with gr.Tab(label="History"):
        # history
        history = gr.Textbox(label="History")

    btn_run.click(process, inputs=[input_files, device, model, lang, allign, diarization, batch_size, output_format], outputs=[input_files])

if __name__ == "__main__":

    # chekc if output folder exists
    if not os.path.exists(os.getcwd() + "/output"):
        os.mkdir(os.getcwd() + "/output")

    ui.queue(concurrency_count=10).launch()
