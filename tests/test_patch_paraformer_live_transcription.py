from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_paraformer_live_transcription.py"
SPEC = importlib.util.spec_from_file_location("patch_paraformer_live_transcription", SCRIPT)
patcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(patcher)


class ParaformerPatchTest(unittest.TestCase):
    def test_patch_adds_partial_for_progressive_and_is_idempotent(self) -> None:
        source = (
            "from speech_to_speech.pipeline.messages import Transcription\n"
            "\n"
            "class ParaformerSTTHandler:\n"
            "    def process(self, vad_audio):\n"
            "        pred_text = '今天天气不错'\n"
            "        yield Transcription(text=pred_text)\n"
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paraformer_handler.py"
            path.write_text(source, encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                changed = patcher.patch_file(path, backup=False)
            self.assertTrue(changed)
            first_patch = path.read_text(encoding="utf-8")
            self.assertIn("PartialTranscription, Transcription", first_patch)
            self.assertIn('getattr(vad_audio, "mode", None) == "progressive"', first_patch)
            self.assertIn("yield PartialTranscription(text=pred_text)", first_patch)
            self.assertIn("yield Transcription(text=pred_text)", first_patch)

            with redirect_stdout(io.StringIO()):
                changed = patcher.patch_file(path, backup=False)
            self.assertFalse(changed)
            self.assertEqual(first_patch, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
