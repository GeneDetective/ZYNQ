import argparse
import asyncio
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import aiofiles
import aiofiles.os
from dotenv import load_dotenv

# ensure local package import path (keeps the previous behavior)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lightweight logging setup — use a simple formatter here to avoid importing project utils
root_logger = logging.getLogger()
root_logger.handlers.clear()
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
root_logger.addHandler(console_handler)
logging.basicConfig(level=logging.INFO)

load_dotenv()

banner = r"""

          _____                _____                    _____                   _______         
         /\    \              |\    \                  /\    \                 /::\    \        
        /::\    \             |:\____\                /::\____\               /::::\    \       
        \:::\    \            |::|   |               /::::|   |              /::::::\    \      
         \:::\    \           |::|   |              /:::::|   |             /::::::::\    \     
          \:::\    \          |::|   |             /::::::|   |            /:::/~~\:::\    \    
           \:::\    \         |::|   |            /:::/|::|   |           /:::/    \:::\    \   
            \:::\    \        |::|   |           /:::/ |::|   |          /:::/    / \:::\    \  
             \:::\    \       |::|___|______    /:::/  |::|   | _____   /:::/____/   \:::\____\ 
              \:::\    \      /::::::::\    \  /:::/   |::|   |/\    \ |:::|    |     |:::|    |
_______________\:::\____\    /::::::::::\____\/:: /    |::|   /::\____\|:::|____|     |:::|____|
\::::::::::::::::::/    /   /:::/~~~~/~~      \::/    /|::|  /:::/    / \:::\   _\___/:::/    / 
 \::::::::::::::::/____/   /:::/    /          \/____/ |::| /:::/    /   \:::\ |::| /:::/    /  
  \:::\~~~~\~~~~~~        /:::/    /                   |::|/:::/    /     \:::\|::|/:::/    /   
   \:::\    \            /:::/    /                    |::::::/    /       \::::::::::/    /    
    \:::\    \           \::/    /                     |:::::/    /         \::::::::/    /     
     \:::\    \           \/____/                      |::::/    /           \::::::/    /      
      \:::\    \                                       /:::/    /             \::::/____/       
       \:::\____\                                     /:::/    /               |::|    |        
        \::/    /                                     \::/    /                |::|____|        
         \/____/                                       \/____/                  ~~              
                                                                                                
ZYNQ — Zero-Knowledge Intelligence Quotient
"""

logger = logging.getLogger(__name__)

# ---------- Helper: lazy import ----------
import importlib


def _lazy(module_name: str):
    """Import module by name on demand."""
    return importlib.import_module(module_name)


# ---------- argparse Action ----------
class LoadFromFile(argparse.Action):
    """
    Loads CLI arguments from a JSON file. Keeps original behavior.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: Optional[str] = None,
    ) -> None:
        cli_args: str = str()
        with values as f:
            json_data = json.loads(f.read())
            for k, v in json_data.items():
                if isinstance(v, list):
                    for item in v:
                        cli_args += f"--{k} {item} "
                elif isinstance(v, bool):
                    if v:
                        cli_args += f"--{k} "
                else:
                    cli_args += f"--{k} {v} "
            parser.parse_args(shlex.split(cli_args), namespace)


# ---------- CLI command implementations (lazy imports inside functions) ----------
async def run_fuzzer(args: argparse.Namespace) -> None:
    """
    Run the fuzzer using the in-process `fuzz_and_return_json` helper.
    Heavy modules are imported lazily here.
    """
    # lazy imports used by the fuzzer flow
    z_consts = _lazy("zynq.consts")
    WIKI_LINK = getattr(z_consts, "WIKI_LINK", None)

    # utils and fuzzer
    utils = _lazy("zynq.utils.utils")
    CURRENT_TIMESTAMP = getattr(utils, "CURRENT_TIMESTAMP", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    generate_report = getattr(utils, "generate_report", None)
    print_report = getattr(utils, "print_report", None)
    run_ollama_list_command = getattr(utils, "run_ollama_list_command", None)

    fuzzer_module = _lazy("zynq.fuzzer")
    Fuzzer = getattr(fuzzer_module, "Fuzzer")

    # attack and classifier registries
    attacks_enum_mod = _lazy("zynq.handlers.attacks.enums")
    FuzzerAttackMode = getattr(attacks_enum_mod, "FuzzerAttackMode")

    classifiers_mod = _lazy("zynq.handlers.classifiers.base")
    classifiers_fm = getattr(classifiers_mod, "classifiers_fm")

    classifiers_enum_mod = _lazy("zynq.handlers.classifiers.enums")
    Classifier = getattr(classifiers_enum_mod, "Classifier")

    if args.verbose:
        logger.info("Verbose logging ON")
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().propagate = True

    if getattr(args, "ollama_list", False):
        # run the helper (if available)
        if callable(run_ollama_list_command):
            run_ollama_list_command()
        else:
            logger.error("ollama list helper not available.")
        return

    # basic validations
    for req_arg in ["attack_modes"]:
        if not getattr(args, req_arg):
            logger.error(f"Required argument: --{req_arg}")
            return

    for attack_mode in args.attack_modes:
        if attack_mode not in set(FuzzerAttackMode):
            logger.error(f"Attack method '{attack_mode}' not found")
            return

    for req_arg in ["model"]:
        if not getattr(args, req_arg):
            logger.error(f"Required argument: --{req_arg}")
            return

    if not args.target_prompt and not args.target_prompts_file:
        logger.error("Please provide a target prompt (-t) or a file with target prompts (-T)")
        return

    # parse extras into a dict
    try:
        if hasattr(args, "extra") and args.extra:
            extras = {k: v for k, v in (x.split("=", 1) for x in args.extra)}
            del args.extra
            extras.update(**vars(args))
        else:
            extras = vars(args)
    except Exception:
        raise ValueError(
            f"Error adding extra argument, please make sure you use the correct format, i.e -e key=value. For further help, please check the wiki: {WIKI_LINK}"
        )

    # load prompts
    if hasattr(args, "target_prompts_file") and args.target_prompts_file:
        with open(args.target_prompts_file, "r", encoding="utf-8") as f:
            prompts = [p.strip() for p in f.readlines() if p.strip()]
    else:
        prompts = args.target_prompt

    # optional benign prompts
    if hasattr(args, "benign_prompts") and args.benign_prompts:
        logger.info(f"Adding {args.benign_prompts} benign prompts to the attack")
        with open("resources/benign_prompts.txt", "r", encoding="utf-8") as f:
            benign_prompts = f.readlines()[: args.benign_prompts]
        prompts += [p.strip() for p in benign_prompts if p.strip()]

    # instantiate Fuzzer
    fuzzer = Fuzzer(db_address=args.db_address)

    # classifiers
    if hasattr(args, "classifier") and args.classifier:
        for classifier in args.classifier:
            if classifier not in set(Classifier):
                logger.error(f"Classifier {classifier} not found")
                return
            logger.debug(f"Adding classifier {classifier}")
            fuzzer.add_classifier(classifiers_fm[classifier](**extras))
    else:
        logger.info('No classifier defined - using default HAR classifier ("Is the following content harmful?")')
        fuzzer.add_classifier(classifiers_fm[Classifier.HARMFUL_LLM](**extras))

    # prepare models and seeds
    extra_copy = extras.copy()
    extra_copy.pop("model", None)

    args.model = [x.lower() for x in args.model]
    args.auxiliary_model = [x.lower() for x in args.auxiliary_model]

    if args.max_workers < len(args.model):
        logger.warning("Number of workers is less than number of models. Is this intentional?")

    for seed, qualified_model_name in enumerate(args.model + args.auxiliary_model):
        provider_seed: int = seed if args.max_workers > 1 else 0
        fuzzer.add_llm(qualified_model_name, seed=provider_seed, **extra_copy)

    if hasattr(args, "classifier_model") and args.classifier_model:
        if args.classifier_model not in args.model + args.auxiliary_model:
            logger.info("Adding auxiliary model for classifier: " + args.classifier_model)
            fuzzer.add_llm(args.classifier_model, **extra_copy)

    extras["model"] = set(extras.get("model", []))

    # Now run fuzz_and_return_json (deterministic)
    try:
        canon_json_str, report, raw_results = await fuzzer.fuzz_and_return_json(
            attack_modes=[FuzzerAttackMode(a) for a in args.attack_modes],
            model=args.model,
            prompts=prompts,
            **extras,
        )
    except Exception as e:
        logger.error(f"Error during attack: {str(e)}.\nFor further help, please check the wiki: {WIKI_LINK}")
        await fuzzer.cleanup()
        return

    # write results to disk
    ts = CURRENT_TIMESTAMP
    await aiofiles.os.makedirs(f"results/{ts}", exist_ok=True)
    # raw jsonl
    if raw_results:
        logger.info(f"Dumping raw results to results/{ts}/raw.jsonl")
        async with aiofiles.open(f"results/{ts}/raw.jsonl", "w", encoding="utf-8") as f:
            for raw_result in raw_results:
                await f.write((raw_result.model_dump_json() if hasattr(raw_result, "model_dump_json") else json.dumps(raw_result)) + "\n")

    # report
    if report:
        logger.info(f"Dumping report to results/{ts}/report.json")
        async with aiofiles.open(f"results/{ts}/report.json", "w", encoding="utf-8") as f:
            await f.write(report.model_dump_json() if hasattr(report, "model_dump_json") else json.dumps(report))

    # write canonical JSON + hash file
    async with aiofiles.open(f"results/{ts}/canonical.json", "w", encoding="utf-8") as f:
        await f.write(canon_json_str)

    # compute hash using the fuzzer helper if available
    result_hash = getattr(fuzzer, "get_result_hash", None)
    if callable(result_hash):
        fingerprint = fuzzer.get_result_hash(canon_json_str)
    else:
        import hashlib

        fingerprint = "0x" + hashlib.sha256(canon_json_str.encode("utf-8")).hexdigest()

    # Print summary to console
    logger.info("=== ZYNQ audit finished ===")
    logger.info(f"Results saved to results/{ts}/")
    logger.info(f"Canonical JSON length: {len(canon_json_str)} bytes")
    logger.info(f"Result fingerprint (sha256): {fingerprint}")
    print(canon_json_str)
    print(f"\nFINGERPRINT:{fingerprint}\n")

    # cleanup
    await fuzzer.cleanup()


async def run_webui(args: argparse.Namespace) -> None:
    """
    Run the Streamlit web UI (launcher).
    """
    port = getattr(args, "port", 8080)
    ui_path = Path("src") / "zynq" / "webui.py"
    if not ui_path.exists():
        logger.error(f"Web UI entrypoint not found at {ui_path}. Ensure webui.py exists.")
        return

    # Use subprocess to run streamlit in a separate process
    process = subprocess.Popen(
        ["streamlit", "run", str(ui_path), "--server.port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    await asyncio.sleep(2)
    print(f"Web UI is running at http://localhost:{port}, Use Ctrl+C to exit")
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        await asyncio.sleep(0.2)
        print("Web UI terminated")


# ---------- CLI builder / entry ----------
async def run_cli() -> None:
    """
    The command-line interface.
    Commands:
      - webui: launches the streamlit UI
      - fuzz : runs the fuzzer in-process and outputs canonical JSON + fingerprint
    """
    # Now that we're running the CLI, we can safely import light project utilities
    # (like the custom formatter) without causing the heavy model imports during module load.
    try:
        custom_formatter_mod = _lazy("zynq.utils.custom_logging_formatter")
        CustomFormatter = getattr(custom_formatter_mod, "CustomFormatter")
        # replace console handler formatter with the project's nicer formatter
        console_handler.setFormatter(CustomFormatter())
    except Exception:
        # ignore formatter replacement errors — keep the simple formatter
        pass

    print(banner)

    # Lazy import project constants and provider registry used to build help text.
    z_consts = _lazy("zynq.consts")
    DEFAULT_SYSTEM_PROMPT = getattr(z_consts, "DEFAULT_SYSTEM_PROMPT", "")
    PARAMETER_MAX_TOKENS = getattr(z_consts, "PARAMETER_MAX_TOKENS", "max_tokens")
    WIKI_LINK = getattr(z_consts, "WIKI_LINK", "")

    # Build argparser
    parser = argparse.ArgumentParser(
        prog="zynq", formatter_class=argparse.RawTextHelpFormatter, description="ZYNQ - Web-first model red-teaming toolkit (web + CLI)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # webui command
    webui_parser = subparsers.add_parser("webui", help="Run the web UI")
    webui_parser.add_argument("--port", type=int, default=8080, help="Port for streamlit web UI")
    webui_parser.set_defaults(func=run_webui)

    # fuzz parser (defer heavy provider inspection until now)
    fuzz_parser = subparsers.add_parser("fuzz", help="Run the fuzzer (in-process) and output deterministic JSON")
    fuzz_parser.set_defaults(func=run_fuzzer)

    fuzz_parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    fuzz_parser.add_argument("-d", "--db_address", help="MongoDB address (default: 127.0.0.1)", type=str, default="127.0.0.1")
    fuzz_parser.add_argument("-w", "--max_workers", help="Max workers (default: 1)", type=int, default=1)
    fuzz_parser.add_argument("-i", "--attack_id", help="Load previous attack id", type=str, default=None)
    fuzz_parser.add_argument("-C", "--configuration_file", help="Load fuzzer arguments from JSON configuration file", type=open, action=LoadFromFile)

    # retrieve providers/models lazily to build the help string (this can still import provider modules
    # but it happens during CLI invocation rather than module import)
    try:
        providers_mod = _lazy("zynq.llm.providers.base")
        llm_provider_fm = getattr(providers_mod, "llm_provider_fm")
        enums_mod = _lazy("zynq.llm.providers.enums")
        LLMProvider = getattr(enums_mod, "LLMProvider")

        models: dict[LLMProvider, list[str]] = {}
        for provider in LLMProvider:
            supported_models = llm_provider_fm[provider].get_supported_models()
            if isinstance(supported_models, str):
                models.setdefault(provider, []).append(supported_models)
                continue
            for model in llm_provider_fm[provider].get_supported_models():
                models.setdefault(provider, []).append(model)

    except Exception:
        # if provider registry can't be loaded (e.g., missing heavy deps), provide a fall-back help.
        models = {}
    models_help = ""
    for provider_name, model_name in models.items():
        for model in model_name:
            models_help += f"{provider_name.value}/{model}\n"
        models_help += "\n"

    fuzz_parser.add_argument("-m", "--model", help=f"Model(s) to attack, any of:\n\n{models_help}", action="append", type=str, default=[])
    fuzz_parser.add_argument("-a", "--attack_modes", help="Add attack mode any of", action="append", type=str, default=[])
    fuzz_parser.add_argument("-c", "--classifier", help="Add a classifier (default: har)", action="append", type=str, default=[])
    fuzz_parser.add_argument("-cm", "--classifier_model", help="Which model to use for classification (optional)", type=str, default=None)
    fuzz_parser.add_argument("-tc", "--truncate-cot", help="Remove CoT when classifying results (default: true)", action="store_true", default=True)
    fuzz_parser.add_argument(f"-N", f"--{PARAMETER_MAX_TOKENS}", help="Max tokens to generate when generating LLM response (default: 100)", type=int, default=100)
    fuzz_parser.add_argument("-b", f"--benign_prompts", help="Adds n benign prompts to the attack (default: 0)", type=int, default=0)

    group = fuzz_parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--target-prompt", help="Prompt to attack (One or more)", action="append", type=str, default=[])
    group.add_argument("-T", "--target-prompts-file", help="Prompts to attack (from file, line separated)", type=str, default=None)

    fuzz_parser.add_argument("-s", "--system-prompt", help=f"System prompt to use (default: {DEFAULT_SYSTEM_PROMPT})", type=str, default=DEFAULT_SYSTEM_PROMPT)
    fuzz_parser.add_argument("-e", "--extra", help="Extra parameters (for providers/attack handlers) in form of key=value", action="append", type=str, default=[])
    fuzz_parser.add_argument("-E", "--list-extra", help="List extra arguments for selected attack method(s)", action="store_true", default=False)
    fuzz_parser.add_argument("-x", "--auxiliary_model", help="Add auxiliary models", action="append", type=str, default=[])
    fuzz_parser.add_argument("-I", "--improve-attempts", help="Attempts to refine the LLM response up to n times following a successful jailbreak.", type=int, default=0)
    fuzz_parser.add_argument("-ol", "--ollama-list", action="store_true", help="Shows all the ollama models that are installed on the station")

    args = parser.parse_args()
    if asyncio.iscoroutinefunction(args.func):
        await args.func(args)
    else:
        args.func(args)


def main() -> None:
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        logger.info("Exiting...")
        exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=False)
        exit(1)


if __name__ == "__main__":
    main()
