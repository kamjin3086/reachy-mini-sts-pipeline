from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_qwen3_tts_inline_instruct.py"
SPEC = importlib.util.spec_from_file_location("patch_qwen3_tts_inline_instruct", SCRIPT)
patcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(patcher)


class Qwen3InlineInstructPatchTest(unittest.TestCase):
    def test_patch_extracts_inline_instruct_and_is_idempotent(self) -> None:
        source = (
            "from time import perf_counter\n"
            "\n"
            "class Qwen3TTSHandler:\n"
            "    def process(self, tts_input):\n"
            "        coalesced_text = '（开心地）你好'\n"
            "        runtime_config = None\n"
            "        response = None\n"
            "        text = coalesced_text or \"Hello.\"\n"
            "\n"
            "        model_type = self._model_type()\n"
            "        self._apply_session_voice_override(model_type, runtime_config, response)\n"
            "\n"
            "        console.print(f\"[green]ASSISTANT: {text}\")\n"
            "\n"
            "        try:\n"
            "            if self.ref_audio:\n"
            "                yield from self._process_voice_clone(text)\n"
            "            elif model_type == \"custom_voice\":\n"
            "                yield from self._process_custom_voice(text)\n"
            "            elif model_type == \"voice_design\":\n"
            "                yield from self._process_voice_design(text)\n"
            "            else:\n"
            "                raise ValueError(\n"
            "                    \"Qwen3-TTS Base model requires ref_audio for voice cloning. \"\n"
            "                    \"Provide qwen3_tts_ref_audio or use a CustomVoice/VoiceDesign model.\"\n"
            "                )\n"
            "        except Exception as e:\n"
            "            logger.error(f\"Error during Qwen3-TTS generation: {e}\", exc_info=True)\n"
            "\n"
            "    def _process_voice_clone(self, text: str) -> Iterator[bytes | np.ndarray]:\n"
            "        pass\n"
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen3_tts_handler.py"
            path.write_text(source, encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                changed = patcher.patch_file(path, backup=False)
            self.assertTrue(changed)
            first_patch = path.read_text(encoding="utf-8")
            self.assertIn("import re", first_patch)
            self.assertIn("def _extract_inline_instruct", first_patch)
            self.assertIn("text, inline_instruct = self._extract_inline_instruct(text)", first_patch)
            self.assertIn("finally:\n            self.instruct = original_instruct", first_patch)

            with redirect_stdout(io.StringIO()):
                changed = patcher.patch_file(path, backup=False)
            self.assertFalse(changed)
            self.assertEqual(first_patch, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
