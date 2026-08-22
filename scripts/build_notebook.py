"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install the `arc-agi` wheel from the offline competition dataset.
  Cell 2: write `my_agent.py` to /kaggle/working/ — its body is THIS file.
  Cell 3: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 4: otherwise (during commit / save-and-run-all), write a dummy
          submission.parquet so Kaggle accepts the commit.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "t4"

# Internal mapping; don't edit unless Kaggle adds new options.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text()

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )
    # --- INJECT OLLAMA SETUP CELL ---
    ollama_cell_source = dedent(
        '''\
        import os
        import sys
        import time
        import glob
        import shutil
        import subprocess
        import urllib.request
        import urllib.error
        import json

        print("=== DEWMA Full Ollama Distribution Test ===")

        OLLAMA_HOST = "127.0.0.1:11434"
        MODEL_TAG = "qwen2.5-coder:7b"
        OLLAMA_LOG = "/tmp/ollama.log"
        MODELFILE_PATH = "/tmp/Modelfile"


        def print_ollama_log(max_characters=10000):
            print("\\n=== Ollama Server Log ===")

            if not os.path.exists(OLLAMA_LOG):
                print("Ollama log does not exist.")
                return

            try:
                with open(
                    OLLAMA_LOG,
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as log_file:
                    content = log_file.read()

                print(content[-max_characters:])

            except Exception as log_error:
                print(f"Could not read Ollama log: {log_error}")


        def find_library_directories(root_directory):
            """
            Find every directory containing at least one Linux shared library.
            """
            library_directories = []

            for root, _, files in os.walk(root_directory):
                contains_shared_library = any(
                    filename.endswith(".so") or ".so." in filename
                    for filename in files
                )

                if contains_shared_library:
                    library_directories.append(root)

            return sorted(set(library_directories))


        def restore_shared_library_symlinks(library_root):
            """
            Restore missing SONAME symbolic links.

            Kaggle datasets may not preserve symbolic links. For example:

                libllama-common.so.0.1.0

            may exist while the required SONAME link is missing:

                libllama-common.so.0 -> libllama-common.so.0.1.0
            """
            created_links = []

            for root, _, files in os.walk(library_root):
                for filename in files:
                    file_path = os.path.join(root, filename)

                    if not os.path.isfile(file_path):
                        continue

                    if ".so." not in filename:
                        continue

                    readelf_result = subprocess.run(
                        ["readelf", "-d", file_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        check=False,
                    )

                    soname = None

                    for line in readelf_result.stdout.splitlines():
                        if (
                            "(SONAME)" in line
                            and "[" in line
                            and "]" in line
                        ):
                            soname = (
                                line.split("[", 1)[1]
                                .split("]", 1)[0]
                                .strip()
                            )
                            break

                    if not soname:
                        continue

                    soname_path = os.path.join(root, soname)

                    if os.path.lexists(soname_path):
                        continue

                    # Relative target keeps the copied distribution portable.
                    os.symlink(filename, soname_path)

                    created_links.append(
                        {
                            "link": soname_path,
                            "target": filename,
                        }
                    )

            return created_links


        def wait_for_server(timeout_seconds=30):
            deadline = time.time() + timeout_seconds
            last_error = None

            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(
                        f"http://{OLLAMA_HOST}/api/tags",
                        timeout=2,
                    ) as response:
                        if response.status == 200:
                            return True

                except Exception as error:
                    last_error = error
                    time.sleep(1)

            print(f"Last server connection error: {last_error}")
            return False


        def list_unresolved_dependencies(executable_path, environment):
            dependency_check = subprocess.run(
                ["ldd", executable_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
                check=False,
            )

            print("\\n=== llama-server Dependency Check ===")
            print(dependency_check.stdout)

            unresolved = [
                line.strip()
                for line in dependency_check.stdout.splitlines()
                if "not found" in line
            ]

            return unresolved


        server_process = None
        log_handle = None

        try:
            # -------------------------------------------------------------
            # 0. Stop processes left behind by previous Kaggle cells
            # -------------------------------------------------------------
            subprocess.run(
                ["pkill", "-x", "ollama"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

            subprocess.run(
                ["pkill", "-x", "llama-server"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

            time.sleep(2)

            # Remove stale log output from previous executions.
            with open(OLLAMA_LOG, "w"):
                pass

            # -------------------------------------------------------------
            # 1. Locate the complete offline Ollama distribution
            # -------------------------------------------------------------
            all_candidates = glob.glob("/kaggle/input/**/ollama", recursive=True)
            offline_binaries = [
                p for p in all_candidates
                if os.path.isfile(p) and not p.endswith(".py") and not p.endswith(".sh") and not p.endswith(".gguf")
            ]

            if not offline_binaries:
                print("=== Debug: Listing all files under /kaggle/input ===")
                for root_dir, _, files in os.walk("/kaggle/input"):
                    for file_name in files:
                        print(os.path.join(root_dir, file_name))
                raise FileNotFoundError(
                    "Ollama binary was not found under /kaggle/input. "
                    "Ensure dataset containing ollama binary is attached."
                )

            source_binary = offline_binaries[0]
            print(f"Found Ollama binary at: {source_binary}")

            if os.path.basename(os.path.dirname(source_binary)) == "bin":
                source_distribution = os.path.dirname(os.path.dirname(source_binary))
            else:
                source_distribution = os.path.dirname(source_binary)

            destination_distribution = "/tmp/dist/linux-amd64"

            print(f"Source distribution: {source_distribution}")

            if os.path.exists(destination_distribution):
                shutil.rmtree(destination_distribution)

            shutil.copytree(
                source_distribution,
                destination_distribution,
                symlinks=True,
            )

            # Locate copied binary and library root
            copied_candidates = glob.glob(f"{destination_distribution}/**/ollama", recursive=True)
            copied_files = [p for p in copied_candidates if os.path.isfile(p) and not p.endswith(".py") and not p.endswith(".sh") and not p.endswith(".gguf")]
            ollama_binary = copied_files[0] if copied_files else os.path.join(destination_distribution, "ollama")

            lib_candidates = glob.glob(f"{destination_distribution}/**/lib/ollama", recursive=True) + glob.glob(f"{destination_distribution}/**/runners", recursive=True)
            ollama_library_root = lib_candidates[0] if lib_candidates else destination_distribution

            print(f"Copied distribution to: {destination_distribution}")
            print(f"Ollama binary: {ollama_binary}")
            print(f"Ollama library root: {ollama_library_root}")

            # -------------------------------------------------------------
            # 2. Set file permissions
            # -------------------------------------------------------------
            os.chmod(ollama_binary, 0o755)

            for root, directories, files in os.walk(ollama_library_root):
                os.chmod(root, 0o755)

                for directory in directories:
                    directory_path = os.path.join(root, directory)
                    os.chmod(directory_path, 0o755)

                for filename in files:
                    file_path = os.path.join(root, filename)

                    # Do not replace a library symlink's target permissions.
                    if os.path.islink(file_path):
                        continue

                    try:
                        os.chmod(file_path, 0o755)
                    except FileNotFoundError:
                        pass

            # -------------------------------------------------------------
            # 3. Restore library links lost during Kaggle dataset upload
            # -------------------------------------------------------------
            print("\\nRestoring shared-library SONAME links...")

            restored_links = restore_shared_library_symlinks(
                ollama_library_root
            )

            if restored_links:
                print(f"Restored {len(restored_links)} library links:")

                for restored_link in restored_links:
                    print(
                        f"  - {os.path.basename(restored_link['link'])}"
                        f" -> {restored_link['target']}"
                    )
            else:
                print("No missing SONAME links were detected.")

            # -------------------------------------------------------------
            # 4. Build LD_LIBRARY_PATH
            # -------------------------------------------------------------
            library_directories = find_library_directories(
                ollama_library_root
            )

            library_directories.insert(0, ollama_library_root)
            library_directories = list(
                dict.fromkeys(library_directories)
            )

            existing_ld_library_path = os.environ.get(
                "LD_LIBRARY_PATH",
                "",
            )

            combined_library_path = os.pathsep.join(
                library_directories
            )

            if existing_ld_library_path:
                combined_library_path += (
                    os.pathsep + existing_ld_library_path
                )

            os.environ["LD_LIBRARY_PATH"] = combined_library_path
            os.environ["OLLAMA_RUNNERS_DIR"] = ollama_library_root
            os.environ["OLLAMA_HOST"] = OLLAMA_HOST
            os.environ["OLLAMA_NUM_PARALLEL"] = "1"
            os.environ["OLLAMA_CONTEXT_LENGTH"] = "2048"
            os.environ["OLLAMA_KEEP_ALIVE"] = "0"

            os.environ["PATH"] = (
                os.path.join(destination_distribution, "bin")
                + os.pathsep
                + os.environ.get("PATH", "")
            )

            server_environment = os.environ.copy()

            print(
                f"Library directories configured: "
                f"{len(library_directories)}"
            )

            for library_directory in library_directories:
                print(f"  - {library_directory}")

            # -------------------------------------------------------------
            # 5. Verify required libraries
            # -------------------------------------------------------------
            required_library_names = [
                "libllama-common.so.0",
                "libmtmd.so.0",
                "libllama.so.0",
                "libggml.so.0",
                "libggml-base.so.0",
            ]

            print("\\nChecking required Ollama libraries...")

            missing_required_libraries = []

            for required_name in required_library_names:
                matches = glob.glob(
                    os.path.join(
                        ollama_library_root,
                        "**",
                        required_name,
                    ),
                    recursive=True,
                )

                if matches:
                    print(f"  ✅ {required_name}: {matches[0]}")
                else:
                    print(f"  ❌ {required_name}: missing")
                    missing_required_libraries.append(required_name)

            if missing_required_libraries:
                raise FileNotFoundError(
                    "Required shared libraries are missing: "
                    + ", ".join(missing_required_libraries)
                )

            # -------------------------------------------------------------
            # 6. Locate llama-server
            # -------------------------------------------------------------
            llama_server_candidates = glob.glob(
                os.path.join(
                    ollama_library_root,
                    "**",
                    "llama-server",
                ),
                recursive=True,
            )

            if not llama_server_candidates:
                raise FileNotFoundError(
                    "No llama-server executable was found under lib/ollama."
                )

            llama_server_binary = llama_server_candidates[0]
            os.chmod(llama_server_binary, 0o755)

            print(f"\\nllama-server: {llama_server_binary}")

            unresolved_dependencies = list_unresolved_dependencies(
                llama_server_binary,
                server_environment,
            )

            if unresolved_dependencies:
                print("❌ Unresolved llama-server dependencies:")

                for dependency in unresolved_dependencies:
                    print(f"  {dependency}")

                raise RuntimeError(
                    "llama-server still has unresolved shared-library "
                    "dependencies."
                )

            print("llama-server dependency check passed ✅")

            # -------------------------------------------------------------
            # 7. Locate the GGUF model
            # -------------------------------------------------------------
            gguf_candidates = glob.glob(
                "/kaggle/input/**/*.gguf",
                recursive=True,
            )

            if not gguf_candidates:
                raise FileNotFoundError(
                    "No GGUF model was found under /kaggle/input."
                )

            preferred_candidates = [
                path
                for path in gguf_candidates
                if "qwen" in path.lower()
                or "coder" in path.lower()
                or "gemma4" in path.lower()
                or "e4b" in path.lower()
            ]

            gguf_path = (
                preferred_candidates[0]
                if preferred_candidates
                else gguf_candidates[0]
            )

            gguf_size_gib = os.path.getsize(gguf_path) / (1024 ** 3)

            print(f"\\nSelected GGUF: {gguf_path}")
            print(f"GGUF size: {gguf_size_gib:.2f} GiB")

            # -------------------------------------------------------------
            # 8. Start Ollama using the corrected environment
            # -------------------------------------------------------------
            print("\\nStarting Ollama server...")

            log_handle = open(
                OLLAMA_LOG,
                "w",
                encoding="utf-8",
            )

            server_process = subprocess.Popen(
                [ollama_binary, "serve"],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=server_environment,
            )

            if not wait_for_server(timeout_seconds=30):
                if server_process.poll() is not None:
                    print(
                        f"Ollama process exited with code "
                        f"{server_process.returncode}."
                    )

                if log_handle:
                    log_handle.flush()

                print_ollama_log()

                raise RuntimeError(
                    "Ollama did not become ready within 30 seconds."
                )

            print("Ollama daemon connected on port 11434 ✅")

            # -------------------------------------------------------------
            # 9. Create the local Ollama model
            # -------------------------------------------------------------
            with open(
                MODELFILE_PATH,
                "w",
                encoding="utf-8",
            ) as model_file:
                model_file.write(f'FROM "{gguf_path}"\\n')
                model_file.write("PARAMETER num_ctx 2048\\n")
                model_file.write("PARAMETER temperature 0\\n")

            print(f"\\nCreating model tag '{MODEL_TAG}'...")

            create_result = subprocess.run(
                [
                    ollama_binary,
                    "create",
                    MODEL_TAG,
                    "-f",
                    MODELFILE_PATH,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=server_environment,
                check=False,
            )

            print(create_result.stdout)

            if create_result.returncode != 0:
                if log_handle:
                    log_handle.flush()

                print_ollama_log()

                raise RuntimeError(
                    f"ollama create failed with exit code "
                    f"{create_result.returncode}."
                )

            print(f"Ollama model '{MODEL_TAG}' registered ✅")

            # -------------------------------------------------------------
            # 10. Test real model loading and generation
            # -------------------------------------------------------------
            print("\\nLoading model and testing generation...")

            request_body = {
                "model": MODEL_TAG,
                "prompt": (
                    "Reply with exactly the single word CONNECTED "
                    "and nothing else."
                ),
                "stream": False,
                "keep_alive": 0,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 2048,
                    "num_predict": 10,
                },
            }

            request = urllib.request.Request(
                f"http://{OLLAMA_HOST}/api/generate",
                data=json.dumps(request_body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            # Initial CPU model loading may take several minutes.
            with urllib.request.urlopen(
                request,
                timeout=300,
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

            generated_text = result.get("response", "").strip()

            print(f"✅ LLM response: {generated_text}")
            print(f"Load duration: {result.get('load_duration')}")
            print(f"Prompt tokens: {result.get('prompt_eval_count')}")
            print(f"Generated tokens: {result.get('eval_count')}")
            print("\\n=== FULL OLLAMA TEST PASSED ===")

        except urllib.error.HTTPError as error:
            print(f"\\n❌ HTTP Error {error.code}: {error.reason}")

            try:
                error_body = error.read().decode(
                    "utf-8",
                    errors="replace",
                )
                print(f"API error body: {error_body}")
            except Exception:
                pass

            if log_handle:
                log_handle.flush()

            print_ollama_log()

        except urllib.error.URLError as error:
            print(f"\\n❌ Connection error: {error}")

            if log_handle:
                log_handle.flush()

            print_ollama_log()

        except FileNotFoundError as error:
            print(f"\\n❌ Required file or command missing: {error}")

            if log_handle:
                log_handle.flush()

            print_ollama_log()

        except Exception as error:
            print(f"\\n❌ Ollama test failed: {error}")
            if log_handle:
                log_handle.flush()
            print_ollama_log()

        finally:
            # Keep Ollama running after a successful test.
            # Only close this notebook's Python handle to the log file.
            if log_handle:
                log_handle.flush()
                log_handle.close()
        '''
    )
    ollama_setup_cell = code_cell(ollama_cell_source)

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # Register MyAgent in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        DEWMA_MODEL_TAG=qwen2.5-coder:7b
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                python main.py --agent myagent

            # Ensure submission.parquet is placed at root working directory
            !if [ -f /kaggle/working/ARC-AGI-3-Agents/submission.parquet ]; then \\
                cp /kaggle/working/ARC-AGI-3-Agents/submission.parquet /kaggle/working/submission.parquet; \\
                echo "Copied submission.parquet to /kaggle/working/ ✅"; \\
            fi

            import glob, shutil
            sub_files = glob.glob('/kaggle/working/**/submission.parquet', recursive=True)
            for sf in sub_files:
                if sf != '/kaggle/working/submission.parquet':
                    shutil.copy(sf, '/kaggle/working/submission.parquet')
                    print(f"Python fallback: copied {sf} -> /kaggle/working/submission.parquet ✅")
        """
    )
    run_cell = code_cell(run_cell_source)

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN') or not os.path.exists('/kaggle/working/submission.parquet'):
                # Save-and-run-all (commit) mode: emit dummy submission so notebook commit succeeds.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# GeminiAgent_V1\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."
            ),
            install_cell,
            write_agent_cell,
            ollama_setup_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[build_notebook] Synced enable_gpu={wanted} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
