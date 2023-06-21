import gc
import tqdm
import gradio as gr
import ffmpeg
import numpy as np
import time
import torch
import os
import dotenv
from whisperx_custom import utils
import json
import pandas as pd

dotenv.load_dotenv()


def process(
    files,
    device,
    model,
    lang,
    allign,
    diarization,
    batch_size,
    output_format,
    min_speakers,
    max_speakers,
    max_line_count,
    max_line_width,
    interpolate_method,
    return_char_alignments,
    vad_onset,
    vad_offset,
    compute_type,
    beam_size,
    patience,
    length_penalty,
    temperature,
    compression_ratio_threshold,
    logprob_threshold,
    no_speech_threshold,
    initial_prompt,
    progress=gr.Progress(track_tqdm=True),
):
    progress(0, desc="Loading models...")
    import whisperx_custom

    if files is None:
        raise gr.Error("Please upload a file to transcribe")

    asr_options = {
        "beam_size": beam_size,
        "patience": None if patience == 0 else patience,
        "length_penalty": None if length_penalty == 0 else length_penalty,
        "temperatures": temperature,
        "compression_ratio_threshold": compression_ratio_threshold,
        "log_prob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "condition_on_previous_text": False,
        "initial_prompt": None if initial_prompt == "" else initial_prompt,
    }

    results = []
    tmp_results = []
    whisper_model = whisperx_custom.load_model(
        model,
        device=device,
        compute_type=compute_type,
        language=None if lang == "" else lang,
        asr_options=asr_options,
        vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset},
    )

    for file in tqdm.tqdm(files, desc="Transcribing", position=0, leave=True, unit="files"):
        result = whisper_model.transcribe(file.name, batch_size=batch_size)
        results.append((result, file.name))

    del whisper_model
    gc.collect()
    torch.cuda.empty_cache()

    # load whisperx model
    if allign:
        tmp_results = results

        if lang == "":
            lang = "en"

        results = []
        align_model, align_metadata = whisperx_custom.load_align_model(model_name="WAV2VEC2_ASR_LARGE_LV60K_960H" if lang == "en" else None, language_code=lang, device=device)

        for result, audio_path in tqdm.tqdm(tmp_results, desc="Alligning", position=0, leave=True, unit="files"):
            input_audio = audio_path

            if align_model is not None and len(result["segments"]) > 0:
                if result.get("language") != align_metadata["language"]:
                    # load new model
                    print(f"Loading new model for {result['language']}")
                    align_model, align_metadata = whisperx_custom.load_align_model(result["language"], device=device)
                result = whisperx_custom.align(
                    result["segments"], align_model, align_metadata, input_audio, device, interpolate_method=interpolate_method, return_char_alignments=return_char_alignments
                )
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
            diarize_model = whisperx_custom.DiarizationPipeline(use_auth_token=os.getenv("hf_token"), device=device)
            for result, input_audio_path in tqdm.tqdm(tmp_res, desc="Diarizing", position=0, leave=True, unit="files"):
                diarize_segments = diarize_model(input_audio_path, min_speakers=min_speakers, max_speakers=max_speakers)
                result = whisperx_custom.diarize.assign_word_speakers(diarize_segments, result)
                results.append((result, input_audio_path))

    writer_args = {"max_line_width": None if max_line_width == 0 else max_line_width, "max_line_count": None if max_line_count == 0 else max_line_count, "highlight_words": False}

    for res, audio_path in tqdm.tqdm(results, desc="Writing", position=0, leave=True, unit="files"):

        filename_alpha_numeric = "".join([c for c in os.path.basename(audio_path) if c.isalpha() or c.isdigit() or c == " "]).rstrip()

        if not os.path.exists(os.getcwd() + "/output/" + filename_alpha_numeric):
            os.mkdir(os.getcwd() + "/output/" + filename_alpha_numeric)

        writer = utils.get_writer(output_format, os.getcwd() + "/output/" + filename_alpha_numeric)
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
                        max_line_width = gr.Slider(label="Max Line Width (0 for default)", minimum=0, step=1, maximum=10000, value=0)
                        max_line_count = gr.Slider(label="Max number of lines in a segment (0 for default)", minimum=0, step=1, maximum=10000, value=0, visible=False)

                        def change_interactive1(max_line_width, max_line_count, val):
                            return [
                                gr.Number.update(visible=val, value=0 if not val else max_line_width),
                                gr.Number.update(visible=not val, value=0 if val else max_line_count),
                            ]

                        allign.change(fn=change_interactive1, inputs=[max_line_width, max_line_count, allign], outputs=[max_line_width, max_line_count])
                        diarization = gr.Checkbox(label="Speaker Diarization")
                        with gr.Row():
                            min_speakers = gr.Slider(label="Min Speakers", minimum=1, maximum=100, step=1, value=1, visible=False)
                            max_speakers = gr.Slider(label="Max Speakers", minimum=1, maximum=100, step=1, value=1, visible=False)

                            # enable min and max speakers if diarization is enabled
                            def change_interactive2(min, max, val):
                                return [
                                    gr.Number.update(visible=val),
                                    gr.Number.update(visible=val),
                                ]

                            diarization.change(fn=change_interactive2, inputs=[min_speakers, max_speakers, diarization], outputs=[min_speakers, max_speakers])

                    with gr.Group():
                        # device add cuda to dropdown if available
                        device = gr.Dropdown(
                            label="Device", choices=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], value="cuda" if torch.cuda.is_available() else "cpu"
                        )
                        batch_size = gr.Slider(label="Batch Size", min_value=1, maximum=100, step=1, value=8, interactive=True)
                        compute_type = gr.Dropdown(label="Compute Type", choices=["int8", "float32", "float16"], value="float16")

                with gr.Group():
                    advanced = gr.Checkbox(label="Advanced Options", value=False)

                    def change_visible1(advanced):
                        return [
                            gr.Dropdown.update(visible=advanced),
                            gr.Checkbox.update(visible=advanced),
                            gr.Slider.update(visible=advanced),
                            gr.Slider.update(visible=advanced),
                            gr.Slider.update(visible=advanced),
                            gr.Slider.update(visible=advanced),
                            gr.Slider.update(visible=advanced),
                            gr.Slider.update(visible=advanced),
                            gr.Slider.update(visible=advanced),
                            gr.Textbox.update(visible=advanced),
                            gr.Slider.update(visible=advanced),
                            gr.Slider.update(visible=advanced),
                        ]

                    with gr.Row():
                        interpolate_method = gr.Dropdown(label="Interpolate Method", choices=["nearest", "linear", "ignore"], value="nearest", visible=False)
                        return_char_alignments = gr.Checkbox(label="Return Char Alignments", value=False, visible=False)
                    with gr.Row():
                        beam_size = gr.Slider(label="Beam Size (only when temperature is 0)", minimum=1, maximum=100, step=1, value=5, visible=False)
                        patience = gr.Slider(label="Patience (0 default)", minimum=0, maximum=100, step=0.01, value=0, visible=False)
                    with gr.Row():
                        length_penalty = gr.Slider(label="Length Penalty (0 default)", minimum=0, maximum=100, step=0.01, value=0, visible=False)
                        temperature = gr.Slider(label="Temperature", minimum=0, maximum=100, step=0.01, value=0, visible=False)
                    with gr.Row():
                        compression_ratio_threshold = gr.Slider(label="Compression Ratio Threshold", minimum=0, maximum=100, step=0.01, value=2.4, visible=False)
                        logprob_threshold = gr.Slider(label="Logprob Threshold", minimum=-10, maximum=10, step=0.01, value=-1, visible=False)
                    with gr.Row():
                        no_speech_threshold = gr.Slider(label="No Speech Threshold", minimum=0, maximum=1, step=0.001, value=0.6, visible=False)
                        initial_prompt = gr.Textbox(label="Initial Prompt", placeholder="Enter initial prompt", visible=False)
                    with gr.Row():
                        vad_onset = gr.Slider(label="VAD Onset Threshold", minimum=0, maximum=1, step=0.0001, value=0.5, visible=False)
                        vad_offset = gr.Slider(label="VAD Offset Threshold", minimum=0, maximum=1, step=0.0001, value=0.363, visible=False)
                        advanced.change(
                            fn=change_visible1,
                            inputs=[advanced],
                            outputs=[
                                interpolate_method,
                                return_char_alignments,
                                beam_size,
                                patience,
                                length_penalty,
                                temperature,
                                compression_ratio_threshold,
                                logprob_threshold,
                                no_speech_threshold,
                                initial_prompt,
                                vad_onset,
                                vad_offset,
                            ],
                        )

                # output format
                output_format = gr.Dropdown(label="Output Format", choices=["all", "json", "txt", "srt", "vtt", "tsv"], value="all")

        gr.Label(value="Output Folder: " + os.getcwd() + "\output")

    with gr.Tab(label="History"):

        def fill_dropdown():
            folders = os.listdir(os.getcwd() + "/output")
            return gr.Dropdown.update(choices=folders)

        history_dropdown = gr.Dropdown(label="Folder", choices=os.listdir(os.getcwd() + "/output"), interactive=True, value="")
        btn_refresh = gr.Button(value="Refresh output list")
        btn_refresh.click(fill_dropdown, inputs=None, outputs=history_dropdown)

        def set_file_type(selected):
            if selected == "":
                return gr.Dropdown.update(choices=["select a file"])
            files = [os.path.splitext(x)[1] for x in os.listdir(os.getcwd() + "/output/" + selected)]
            return gr.Dropdown.update(choices=files, interactive=True)

        file_type = gr.Dropdown(label="File Type", choices=[], value="select a file", interactive=False)
        history_dropdown.change(set_file_type, inputs=history_dropdown, outputs=file_type)

        def fill_output(folder, type):
            if folder == "" or type == "select a file":
                return gr.TextArea.update(value="")
            file = [x for x in os.listdir(os.getcwd() + "/output/" + folder) if os.path.splitext(x)[1] == type][0]
            with open(os.getcwd() + "/output/" + folder + "/" + file, "r", encoding="utf-8") as f:
                text = f.read()
            return gr.TextArea.update(value=text)

        output_text_field = gr.TextArea(label="Output (changes made wont be saved - files are also in the output folder)", value="", interactive=True)
        file_type.change(fill_output, inputs=[history_dropdown, file_type], outputs=output_text_field)

    btn_run.click(
        process,
        inputs=[
            input_files,
            device,
            model,
            lang,
            allign,
            diarization,
            batch_size,
            output_format,
            min_speakers,
            max_speakers,
            max_line_count,
            max_line_width,
            interpolate_method,
            return_char_alignments,
            vad_onset,
            vad_offset,
            compute_type,
            beam_size,
            patience,
            length_penalty,
            temperature,
            compression_ratio_threshold,
            logprob_threshold,
            no_speech_threshold,
            initial_prompt,
        ],
        outputs=[input_files],
    )

if __name__ == "__main__":
    ui.queue(concurrency_count=10).launch()
