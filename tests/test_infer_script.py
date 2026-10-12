import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InferScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workdir = Path(self.temp.name)
        (self.workdir / "input song.wav").touch()
        helper = self.workdir / "capture-python"
        helper.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "audio = args[args.index('--input') + 1]\n"
            "output = args[args.index('--output-dir') + 1]\n"
            "print(json.dumps({'cwd': os.getcwd(), 'args': args, "
            "'input_exists': Path(audio).is_file(), "
            "'output': str(Path(output).resolve()), "
            "'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}))\n"
        )
        helper.chmod(0o700)
        self.env = dict(os.environ, PYTHON_BIN=str(helper), CUDA_VISIBLE_DEVICES="2")
        self.env.pop("CKPT_PATH", None)
        self.env.pop("MODEL_CONFIG", None)

    def invoke(self, *extra, env=None):
        result = subprocess.run(
            ["bash", str(PROJECT_ROOT / "scripts/infer.sh"),
             "input song.wav", "output directory", *extra],
            cwd=self.workdir, env=self.env if env is None else env,
            text=True, capture_output=True, check=True,
        )
        return json.loads(result.stdout)

    def test_external_working_directory_and_paths_with_spaces(self):
        result = self.invoke("0")
        self.assertEqual(result["cwd"], str(self.workdir))
        self.assertTrue(result["input_exists"])
        self.assertEqual(result["output"], str(self.workdir / "output directory"))
        args = result["args"]
        self.assertEqual(args[args.index("--model-config") + 1],
                         str(PROJECT_ROOT / "src/config/vae/config.yaml"))
        self.assertTrue(Path(args[args.index("--model-config") + 1]).is_file())
        self.assertEqual(args[args.index("--ckpt-path") + 1], str(PROJECT_ROOT / "model.pt"))

    def test_flags_without_gpu_preserve_environment_gpu(self):
        result = self.invoke("--latent-mode", "mean", "--seed", "7")
        self.assertEqual(result["gpu"], "2")
        self.assertEqual(result["args"][-4:], ["--latent-mode", "mean", "--seed", "7"])

    def test_omitted_gpu_defaults_to_zero(self):
        env = dict(self.env)
        env.pop("CUDA_VISIBLE_DEVICES")
        self.assertEqual(self.invoke(env=env)["gpu"], "0")

    def test_explicit_gpu_overrides_environment(self):
        for gpu in ("3", "1,2", "-1"):
            with self.subTest(gpu=gpu):
                result = self.invoke(gpu, "--latent-mode", "mean")
                self.assertEqual(result["gpu"], gpu)
                self.assertEqual(result["args"][-2:], ["--latent-mode", "mean"])

    def test_relative_environment_paths_preserve_caller_context(self):
        (self.workdir / "config.yml").touch()
        (self.workdir / "model.pt").touch()
        env = dict(self.env, PYTHON_BIN="./capture-python", MODEL_CONFIG="config.yml", CKPT_PATH="model.pt")
        result = self.invoke(env=env)
        self.assertEqual(result["cwd"], str(self.workdir))
        args = result["args"]
        self.assertTrue((self.workdir / args[args.index("--model-config") + 1]).is_file())
        self.assertTrue((self.workdir / args[args.index("--ckpt-path") + 1]).is_file())


if __name__ == "__main__":
    unittest.main()
