# type: ignore
import os
import sys
import subprocess
from pathlib import Path
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

# Allow nested asyncio loops inside Streamlit (apply early)
import nest_asyncio
nest_asyncio.apply()

# ensure repo src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# safe imports that do NOT import provider modules
from zynq.enums import EnvironmentVariables
from zynq.handlers.attacks.base import attack_handler_fm
from zynq.handlers.attacks.enums import FuzzerAttackMode
from zynq.handlers.classifiers.base import classifiers_fm
from zynq.handlers.classifiers.enums import Classifier
from zynq.utils.utils import get_ollama_models  # safe helper; handles missing ollama CLI internally

# load any existing .env (do not override existing os.environ)
load_dotenv(override=False)


# -----------------------
# Helpers
# -----------------------
def maybe_rerun() -> None:
    """Try experimental_rerun if available, otherwise st.stop()."""
    rerun = getattr(st, "experimental_rerun", None)
    if callable(rerun):
        try:
            rerun()
            return
        except Exception:
            pass
    st.stop()


def save_env_var(key: str, value: str) -> None:
    """
    Save environment variable into process env, session_state, and persist to .env.
    value is quoted when written to .env.
    """
    os.environ[key] = value
    st.session_state.env_vars[key] = value

    env_path = Path(".") / ".env"
    lines: List[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f'{key}="{value}"'
            found = True
            break
    if not found:
        lines.append(f'{key}="{value}"')
    env_path.write_text("\n".join(lines), encoding="utf-8")
    # reload dotenv so code that reads it on-demand picks it up
    load_dotenv(override=True)


def try_import_provider_registry() -> Tuple[Optional[Dict], Optional[List[str]]]:
    """
    Attempt to import provider registry lazily.
    Returns (registry_map, provider_keys) on success, or (None, fallback_keys) on failure.
    """
    try:
        from zynq.llm.providers.base import llm_provider_fm  # local import
        from zynq.llm.providers.enums import LLMProvider
        model_map = {}
        keys = []
        for p in LLMProvider:
            try:
                supported = llm_provider_fm[p].get_supported_models()
            except Exception:
                supported = []
            model_map[p.value] = supported
            keys.append(p.value)
        return model_map, keys
    except Exception:
        # provider registry not importable at this moment (often because env keys not set)
        fallback = ["gemini", "openai", "ollama"]
        return None, fallback


def ensure_env_vars_in_process() -> None:
    """Copy UI-saved env vars into os.environ (defensive)."""
    for k, v in st.session_state.get("env_vars", {}).items():
        if k and v:
            os.environ[k] = v
    load_dotenv(override=True)


def run_onchain_anchor(proof_path: Path, public_path: Path, onchain_script: Path, show_output=True, timeout=300) -> dict:
    """
    Run the onchain anchoring script and return a result dict with stdout/stderr/rc.
    Uses current process python executable to run the script.
    """
    cmd = [sys.executable, str(onchain_script), "--proof", str(proof_path), "--public", str(public_path)]
    env = os.environ.copy()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        result = {"rc": proc.returncode, "stdout": proc.stdout or "", "stderr": proc.stderr or ""}
    except subprocess.TimeoutExpired as e:
        result = {"rc": -1, "stdout": getattr(e, "stdout", "") or "", "stderr": f"Timeout after {timeout}s"}
    except Exception as e:
        result = {"rc": -2, "stdout": "", "stderr": str(e)}
    return result


# -----------------------
# UI appearance (single injection)
# -----------------------
st.set_page_config(page_title="ZYNQ — Zero-Knowledge Intelligence Quotient", layout="wide", initial_sidebar_state="expanded")
CSS = """
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<style>
:root { --bg-color:#000; --accent:#00c6ff; --panel-bg:#050505; }
html,body,.streamlit-container { background: #000; color: #dff6ff; font-family:'Orbitron', system-ui, sans-serif; }
.card{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:16px; border-radius:10px; border:1px solid rgba(0,198,255,0.06); }
.stButton>button{ background: linear-gradient(90deg, rgba(0,198,255,1), rgba(0,120,255,0.9)); color:black; font-weight:700; border-radius:8px; }
.stSidebar .stButton>button{ background:transparent; border:1px solid rgba(0,198,255,0.12); color:var(--accent); }
.big-title{ font-size:30px; font-weight:800; color:white; }
.muted{ color:#7eaec7; }
.particle-bg { position: fixed; inset:0; pointer-events:none; z-index:0; background-image: radial-gradient(rgba(0,198,255,0.04) 1px, transparent 1px); background-size: 36px 36px; opacity:0.55; animation: drift 20s linear infinite; mix-blend-mode:overlay; }
@keyframes drift { from{transform:translate(0,0)} to{transform:translate(200px,-200px)} }
.main .block-container{position:relative; z-index:1;}
.sidebar .block-container{position:relative; z-index:1;}
</style>
<div class="particle-bg"></div>
"""
st.markdown(CSS, unsafe_allow_html=True)


# -----------------------
# Defaults & session state
# -----------------------
DEFAULTS = {
    "env_vars": {},
    "verbose": False,
    "db_address": "127.0.0.1",
    "max_workers": 1,
    "max_tokens": 1000,
    "extra_params": "",
    "selected_models": [],
    "selected_models_aux": [],
    "selected_attacks": [],
    "selected_classifiers": [],
    "classifier_model": None,
    "prompt": "",
    "step": 1,
    # model options cache
    "model_options": {},
    "provider_keys": ["gemini", "openai", "ollama"],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# populate from process env if available
for ev in EnvironmentVariables:
    if ev.value in os.environ and ev.value not in st.session_state.env_vars:
        st.session_state.env_vars[ev.value] = os.environ[ev.value]


# -----------------------
# Sidebar: Environment + settings
# -----------------------
st.sidebar.header("Environment Settings")
api_keys = [x.value for x in EnvironmentVariables]
new_env_key = st.sidebar.selectbox("Name", options=api_keys)
st.session_state.setdefault("_new_env_value", "")
new_env_value = st.sidebar.text_input("Value (hidden)", value=st.session_state.get("_new_env_value", ""), key="_new_env_value", type="password")

if st.sidebar.button("Save Variable"):
    v = st.session_state.get("_new_env_value", "")
    if new_env_key and v:
        try:
            save_env_var(new_env_key, v)
            st.sidebar.success(f"Saved {new_env_key}")
            # after saving a key, reattempt provider registry import (user may want to fetch models)
            maybe_rerun()
        except Exception as e:
            st.sidebar.error(f"Error saving key: {e}")
    else:
        st.sidebar.warning("Please choose a key and enter a value.")

# show saved envs in a small table (masked)
with st.sidebar.container():
    if st.session_state.env_vars:
        cols = st.columns([2, 2, 1])
        cols[0].markdown("**Key**")
        cols[1].markdown("**Value**")
        cols[2].markdown("**Action**")
        for key, value in dict(st.session_state.env_vars).items():
            c1, c2, c3 = st.columns([2, 2, 1])
            c1.text(key)
            masked = (value[:8] + "...") if value else ""
            if c2.button("👁️", key=f"reveal_{key}"):
                c2.text(value)
            else:
                c2.text(masked)
            if c3.button("❌", key=f"del_{key}"):
                st.session_state.env_vars.pop(key, None)
                try:
                    env_path = Path(".") / ".env"
                    if env_path.exists():
                        lines = env_path.read_text(encoding="utf-8").splitlines()
                        lines = [l for l in lines if not l.strip().startswith(f"{key}=")]
                        env_path.write_text("\n".join(lines), encoding="utf-8")
                    if key in os.environ:
                        os.environ.pop(key, None)
                except Exception:
                    pass
                maybe_rerun()

st.sidebar.header("ZYNQ Settings")
st.session_state.verbose = st.sidebar.checkbox("Verbose logging", value=st.session_state.verbose)
st.session_state.db_address = st.sidebar.text_input("MongoDB Address", value=st.session_state.db_address)
st.session_state.max_workers = st.sidebar.number_input("Max Workers", min_value=1, value=st.session_state.max_workers)
st.session_state.max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, value=st.session_state.max_tokens)


# -----------------------
# Try to import provider registry at load (safe attempt)
# -----------------------
if not st.session_state.get("model_options"):
    # Use a lightweight attempt — provider import may fail if keys absent. That's OK.
    model_map, provider_keys = try_import_provider_registry()
    if model_map is not None:
        st.session_state.model_options = model_map
        st.session_state.provider_keys = provider_keys
    else:
        st.session_state.model_options = {k: [] for k in st.session_state.provider_keys}


# -----------------------
# Main UI flow
# -----------------------
st.markdown("<div class='big-title'>ZYNQ</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Zero-Knowledge Intelligence Quotient — Verifiable model red-teaming</div>", unsafe_allow_html=True)
st.write("")

# navigation step
step = st.session_state.step

# Step 1: Model selection (dropdowns like original)
if step == 1:
    st.header("Step 1 — Model Selection")
    st.subheader("Choose target and auxiliary models")

    # provider category choose
    provider = st.selectbox("Select Model Category", options=st.session_state.provider_keys, index=0)

    # show warning if provider registry empty for this provider
    models_for_provider = st.session_state.model_options.get(provider, [])
    if not models_for_provider:
        st.warning(f"No models available for {provider} (provider registry may not be importable). Use 'Refresh providers' after saving keys or use manual entry.")
    # allow fetching provider models on demand (this will copy UI keys -> os.environ first)
    if st.button(f"Refresh providers / fetch models for {provider}"):
        ensure_env_vars_in_process()
        try:
            # try to import provider registry now
            model_map, provider_keys = try_import_provider_registry()
            if model_map is None:
                st.error("Provider registry still unavailable. Make sure API keys are set in the process environment and restart Streamlit if you used setx, or save them above.")
                st.session_state.model_options = {k: [] for k in provider_keys}
                st.session_state.provider_keys = provider_keys
            else:
                st.session_state.model_options = model_map
                st.session_state.provider_keys = provider_keys
                st.success("Fetched provider models (registry imported).")
            maybe_rerun()
        except Exception as e:
            st.error("Failed to refresh providers.")
            st.expander("Traceback").write(str(e))

    # model selection UI (if provider models present)
    if st.session_state.model_options.get(provider):
        selected_model = st.selectbox(f"Select {provider} model", options=st.session_state.model_options[provider], index=0, key=f"sel_model_{provider}")
        if st.button("Add selected model"):
            st.session_state.selected_models.append(f"{provider}/{selected_model}")
            st.success(f"Added {provider}/{selected_model}")
            maybe_rerun()

    # manual model entry (keeps previous behaviour)
    manual = st.text_input("Or enter model identifier manually (provider/model-name)", placeholder="e.g. gemini/gemini-1.5-flash")
    if st.button("Add manual model"):
        if manual:
            st.session_state.selected_models.append(manual.strip())
            st.success(f"Added {manual.strip()}")
            maybe_rerun()
        else:
            st.warning("Enter model identifier first.")

    st.write("Selected Models:", st.session_state.selected_models)

    st.markdown("---")
    st.subheader("Auxiliary models (optional)")
    provider_aux = st.selectbox("Select Model Category (aux)", options=st.session_state.provider_keys, index=0, key="provider_aux")
    if st.session_state.model_options.get(provider_aux):
        sel_aux = st.selectbox(f"Pick {provider_aux} model (aux)", options=st.session_state.model_options[provider_aux], key=f"aux_model_choice")
        if st.button("Add aux model"):
            st.session_state.selected_models_aux.append(f"{provider_aux}/{sel_aux}")
            st.success(f"Added aux {provider_aux}/{sel_aux}")
            maybe_rerun()

    manual_aux = st.text_input("Or enter auxiliary model manually", placeholder="e.g. openai/gpt-4-turbo", key="manual_aux")
    if st.button("Add manual aux"):
        if manual_aux:
            st.session_state.selected_models_aux.append(manual_aux.strip())
            st.success(f"Added {manual_aux.strip()}")
            maybe_rerun()

    st.write("Selected Auxiliary Models:", st.session_state.selected_models_aux)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Back (reset)"):
            st.session_state.step = 1
            maybe_rerun()
    with c2:
        if st.button("Next"):
            if not st.session_state.selected_models:
                st.error("Please select at least one model")
            else:
                st.session_state.step = 2
                maybe_rerun()

# Step 2 — Attack selection (same as original)
elif step == 2:
    st.header("Step 2 — Attack Selection")
    attack_modes = {mode.value: attack_handler_fm[mode].description() for mode in FuzzerAttackMode}
    selected_attacks = st.multiselect("Select Attack Modes", options=list(attack_modes.keys()), format_func=lambda x: f"{x} - {attack_modes[x]}")
    if st.button("List attack extra"):
        if not selected_attacks:
            st.error("Please select at least one attack mode")
        else:
            extra_info = {}
            for attack in selected_attacks:
                try:
                    handler_cls = attack_handler_fm[FuzzerAttackMode(attack)]
                    extra_info[attack] = handler_cls.extra_args() if hasattr(handler_cls, "extra_args") else {}
                except Exception as e:
                    extra_info[attack] = {"error": str(e)}
            st.json(extra_info)

    st.session_state.selected_attacks = selected_attacks
    st.session_state.extra_params = st.text_area("Extra Attack Parameters (line-separated key=value pairs)", placeholder="KEY1=VALUE1\nKEY2=VALUE2")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back"):
            st.session_state.step = 1
            maybe_rerun()
    with col2:
        if st.button("Next"):
            if not selected_attacks:
                st.error("Please select at least one attack mode")
            else:
                st.session_state.step = 3
                maybe_rerun()

# Step 3 — Classifier selection
elif step == 3:
    st.header("Step 3 — Classifier Selection")
    classifiers = {classifier.value: classifiers_fm[classifier].description() for classifier in Classifier}
    selected_classifiers = st.multiselect("Select Classifiers", options=classifiers.keys(), format_func=lambda x: f"{x} - {classifiers[x]}")
    st.session_state.selected_classifiers = selected_classifiers

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back"):
            st.session_state.step = 2
            maybe_rerun()
    with col2:
        if st.button("Next"):
            st.session_state.step = 4
            maybe_rerun()

# Step 4 — Prompt
elif step == 4:
    st.header("Step 4 — Prompt")
    st.subheader("Enter the target prompt (or small instruction)")
    prompt_text = st.text_area("Enter prompt", value=st.session_state.get("prompt", ""), height=140)
    st.session_state.prompt = prompt_text

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back"):
            st.session_state.step = 3
            maybe_rerun()
    with col2:
        if st.button("Next"):
            st.session_state.step = 5
            maybe_rerun()

# Step 5 — Run audit (safe)
elif step == 5:
    st.header("Step 5 — Execute Audit")
    st.markdown("<div class='card'>Review settings and run the ZYNQ audit. Results will be shown below.</div>", unsafe_allow_html=True)

    st.write("**Models:**", st.session_state.selected_models)
    st.write("**Aux Models:**", st.session_state.selected_models_aux)
    st.write("**Attacks:**", st.session_state.selected_attacks)
    st.write("**Classifiers:**", st.session_state.selected_classifiers)
    st.write("**Prompt:**", st.session_state.prompt or "<empty>")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Back"):
            st.session_state.step = 4
            maybe_rerun()
    with col2:
        run_audit = st.button("Run ZYNQ Audit")
    with col3:
        if st.button("Restart"):
            st.session_state.step = 1
            maybe_rerun()

    if run_audit:
        # copy UI-saved env vars to process env (very important)
        ensure_env_vars_in_process()

        # quick debug expander so you can confirm keys visible to process
        with st.expander("Debug: env keys presence (toggle)"):
            st.write({k: (k in os.environ) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")})

        # parse extra params to dict
        ep: Dict[str, Any] = {}
        if st.session_state.extra_params:
            for kvp in st.session_state.extra_params.splitlines():
                if "=" in kvp:
                    k, v = kvp.split("=", 1)
                    ep[k.strip()] = v.strip()

        # NOW import Fuzzer (after env vars are in os.environ)
        try:
            from zynq.fuzzer import Fuzzer  # lazy import
        except Exception as e:
            st.error(f"Could not import Fuzzer: {e}")
            st.expander("Import traceback").write(str(e))
            st.stop()

        try:
            fuzzer = Fuzzer(db_address=st.session_state.db_address, max_workers=st.session_state.max_workers)
        except Exception as e:
            st.error(f"Could not instantiate Fuzzer: {e}")
            st.expander("Traceback").write(str(e))
            st.stop()

        # Add classifiers
        if st.session_state.selected_classifiers:
            for c in st.session_state.selected_classifiers:
                try:
                    fuzzer.add_classifier(classifiers_fm[Classifier(c)](**ep))
                except Exception as e:
                    st.warning(f"Could not add classifier {c}: {e}")

        # Add models — abort if any model fails to add (and show traceback)
        models_to_add = list(set(st.session_state.selected_models + st.session_state.selected_models_aux))
        failed_models: List[Tuple[str, Exception]] = []
        for model in models_to_add:
            try:
                fuzzer.add_llm(model, **ep)
            except Exception as e:
                failed_models.append((model, e))
                st.error(f"Failed to add model {model}: {e}")
                import traceback
                st.expander(f"Traceback for {model}").write(traceback.format_exc())

        if failed_models:
            st.error("One or more models failed to be added. Fix API keys or start Streamlit from a shell where keys are present, then try again.")
            st.stop()

        # prepare attack modes
        try:
            attacks = [FuzzerAttackMode(a) for a in st.session_state.selected_attacks]
        except Exception as e:
            st.error(f"Invalid attack mode(s): {e}")
            attacks = []

        # run fuzz (use nest_asyncio and safe loop handling)
        progress = st.progress(0)
        try:
            with st.spinner("Running ZYNQ audit — this may take a while …"):
                try:
                    loop = asyncio.get_event_loop()
                except Exception:
                    loop = None

                if loop and loop.is_running():
                    coro = fuzzer.fuzz(attack_modes=attacks, model=list(set(st.session_state.selected_models)), prompts=st.session_state.prompt or "", **ep)
                    task = asyncio.ensure_future(coro)
                    tick = 0
                    while not task.done():
                        try:
                            progress.progress(min(100, 10 + tick))
                        except Exception:
                            pass
                        tick = (tick + 5) % 90
                        loop.run_until_complete(asyncio.sleep(0.1))
                    result_pair = task.result()
                else:
                    result_pair = asyncio.run(fuzzer.fuzz(attack_modes=attacks, model=list(set(st.session_state.selected_models)), prompts=st.session_state.prompt or "", **ep))
            st.success("Fuzzing completed")
        except Exception as e:
            st.error(f"Audit run failed: {e}")
            import traceback
            st.expander("Traceback").write(traceback.format_exc())
            try:
                if asyncio.get_event_loop().is_running():
                    asyncio.get_event_loop().run_until_complete(fuzzer.cleanup())
                else:
                    asyncio.run(fuzzer.cleanup())
            except Exception:
                pass
            st.stop()

        # unpack results
        if isinstance(result_pair, tuple) and len(result_pair) == 2:
            report, raw_results = result_pair
        else:
            report = result_pair
            raw_results = []

        combined = {
            "attack_id": getattr(fuzzer, "_attack_id", None),
            "attack_time": getattr(fuzzer, "_attack_time", None),
            "report": report.model_dump() if hasattr(report, "model_dump") else (json.loads(report) if isinstance(report, str) else report),
            "raw_results": [r.model_dump() if hasattr(r, "model_dump") else (json.loads(r) if isinstance(r, str) else r) for r in (raw_results or [])]
        }
        combined_json_str = json.dumps(combined, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

        # save (best-effort)
        try:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            outdir = Path("results") / ts
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "report.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        st.subheader("Summary Report")
        try:
            st.json(combined["report"])
        except Exception:
            st.write(combined["report"])

        st.subheader("Raw results (sample)")
        if combined["raw_results"]:
            try:
                st.json(combined["raw_results"][0])
            except Exception:
                st.write(combined["raw_results"][0])
        else:
            st.write("No raw results returned — this can mean the attack found no vulnerabilities or that classifiers filtered results out.")

        st.download_button(label="Download full audit (JSON)", data=combined_json_str, file_name=f"zynq_audit_{combined['attack_id'] or datetime.utcnow().isoformat()}.json", mime="application/json")

        # -----------------------
        # ZK proof anchoring hook (NEW)
        # - If verify/Janus_proof.json and verify/Janus_public.json exist AND onchain.py is present
        #   we will attempt to run the onchain anchoring script and show its output.
        # - A manual button is also provided to force anchoring.
        # -----------------------
        verify_dir = Path("verify")
        proof_file = verify_dir / "Janus_proof.json"
        public_file = verify_dir / "Janus_public.json"
        onchain_script = Path(__file__).parent.parent / "onchain.py"  # repo root/onchain.py

        # Auto-run attempt (safe): only if both files exist and onchain.py is found
        if proof_file.exists() and public_file.exists():
            if onchain_script.exists():
                with st.expander("ZK Proof found — Anchor to chain (auto-run output)"):
                    st.write("Found proof and public files in `verify/`.")
                    res = run_onchain_anchor(proof_file, public_file, onchain_script, timeout=300)
                    if res["stdout"]:
                        st.code(res["stdout"])
                    if res["stderr"]:
                        st.code("STDERR:\n" + res["stderr"])
                    if res["rc"] == 0:
                        st.success("Anchoring script finished (rc 0). Check the transaction in the script output above.")
                    else:
                        st.warning(f"Anchoring script returned code {res['rc']} — inspect output above.")
            else:
                with st.expander("ZK Proof files found (manual anchor)"):
                    st.info("Proof files are present but onchain.py script was not found at repo root. Please place your `onchain.py` at the repository root or set the correct path.")

        # Manual anchor button
        with st.expander("Manual anchor controls"):
            st.write("If automatic anchoring didn't run (or you prefer manual control), press the button below.")
            if st.button("Anchor proof now (manual)"):
                if not proof_file.exists() or not public_file.exists():
                    st.error("Cannot anchor: proof or public files missing. Expected at: verify/Janus_proof.json and verify/Janus_public.json")
                elif not onchain_script.exists():
                    st.error(f"Cannot anchor: onchain script not found at {onchain_script}")
                else:
                    with st.spinner("Running onchain anchor script..."):
                        res = run_onchain_anchor(proof_file, public_file, onchain_script, timeout=300)
                    if res["stdout"]:
                        st.code(res["stdout"])
                    if res["stderr"]:
                        st.code("STDERR:\n" + res["stderr"])
                    if res["rc"] == 0:
                        st.success("Anchoring script finished (rc 0).")
                    else:
                        st.error(f"Anchoring script returned code {res['rc']}. See output above.")

        # cleanup
        try:
            if asyncio.get_event_loop().is_running():
                asyncio.get_event_loop().run_until_complete(fuzzer.cleanup())
            else:
                asyncio.run(fuzzer.cleanup())
        except Exception:
            pass

# EOF
