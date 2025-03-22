import importlib

benchmark_modules = [
    "benchmarks.text_to_text.gsm8k.eval",
    "benchmarks.text_to_text.mmlu.eval",
    "benchmarks.text_to_text.HumanEval.eval",
    "benchmarks.text_to_text.AGIEval.eval",
    "benchmarks.text_to_text.TruthfulQA.eval",
    "benchmarks.text_to_text.ARC.eval",
    "benchmarks.text_to_text.CMMLU.eval",
    "benchmarks.text_to_text.MMLUPRO.eval",
    "benchmarks.text_to_text.CEval.eval",
    "benchmarks.text_image_to_text.mmmu.eval",
    "benchmarks.text_audio_to_text.mmau.eval",
    "benchmarks.text_video_to_text.mmvu.eval",
]

for module in benchmark_modules:
    importlib.import_module(module)