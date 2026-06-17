import os

# Choose implementation: "phoneme" (F.pad, default) or "original" (F.interpolate)
_SKETCH_MODE = os.environ.get("SKETCH_MODE", "phoneme")

if _SKETCH_MODE == "original":
    from drawspeech.dataset_plugin_original import get_preprocessed_meta
else:
    from drawspeech.dataset_plugin_phoneme import get_preprocessed_meta
