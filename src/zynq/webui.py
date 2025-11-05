# type: ignore
import os
import sys
from pathlib import Path
import json
import asyncio
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# ensure repo src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# keep internal imports as fuzzyai.* for now so this runs without renaming package folder
from zynq.enums import EnvironmentVariables
from zynq.handlers.attacks.base import attack_handler_fm
from zynq.handlers.attacks.enums import FuzzerAttackMode
from zynq.handlers.classifiers.base import classifiers_fm
from zynq.handlers.classifiers.enums import Classifier
from zynq.llm.providers.base import llm_provider_fm
from zynq.llm.providers.enums import LLMProvider
from zynq.utils.utils import get_ollama_models
from zynq.fuzzer import Fuzzer

load_dotenv()

# -----------------------
# THEME / STYLING
# -----------------------
st.set_page_config(
    page_title="ZYNQ — Zero-Knowledge Intelligence Quotient",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject dark neon theme CSS
# Note: Streamlit doesn't support full theming programmatically in all versions,
# so we inject some custom CSS to make the UI dark and neon-blue-ish.
st.markdown(
    """
    <style>
    :root {
        --bg-color: #0b0f12;
        --panel-bg: #071019;
        --accent: #00c6ff;
        --muted: #9aa7b0;
        --card: #071419;
        --glass: rgba(255,255,255,0.02);
    }
    .stApp {
        background: linear-gradient(180deg, var(--bg-color) 0%, #020407 100%);
        color: #dff6ff;
    }
    .css-1d391kg { /* main container */
        background-color: transparent;
    }
    .stButton>button {
        background: linear-gradient(90deg, rgba(0,198,255,1), rgba(0,120,255,0.8));
        color: black;
        font-weight: 600;
        border: none;
        box-shadow: 0 6px 14px rgba(0,198,255,0.08);
    }
    .stSidebar .css-1d391kg, .css-1v3fvcr {
        background-color: var(--panel-bg) !important;
    }
    .stSidebar .stButton>button {
        background: transparent;
        border: 1px solid rgba(0,198,255,0.12);
        color: var(--accent);
    }
    .reportview-container .markdown-text-container p {
        color: #9fdff8;
    }
    .big-title {
        font-size: 34px;
        font-weight: 800;
        color: white;
        letter-spacing: 0.4px;
    }
    .muted {
        color: #7eaec7;
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02));
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(0,198,255,0.06);
    }
    .neon {
        color: var(--accent);
        text-shadow: 0 0 12px rgba(0,198,255,0.35);
    }
    .small {
        font-size: 12px;
        color: var(--muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# LOGO / SIDEBAR
# -----------------------
# Primary logo path provided by user; if missing, fall back to repository resources
USER_LOGO_PATH = Path(r"C:\Users\HP\Downloads\ZYNQ LOGO.png")
FALLBACK_LOGO = Path(__file__).parent.parent / "resources" / "logo.png"

logo_path = None
if USER_LOGO_PATH.exists():
    logo_path = str(USER_LOGO_PATH)
elif FALLBACK_LOGO.exists():
    logo_path = str(FALLBACK_LOGO)

if logo_path:
    st.sidebar.image(logo_path, width=175)
else:
    # fallback: a simple styled title
    st.sidebar.markdown("<h2 class='neon'>ZYNQ</h2>", unsafe_allow_html=True)

# -----------------------
# DEFAULTS / SESSION STATE
# -----------------------
defaults = {
    "env_vars": {},
    "verbose": False,
    "db_address": "127.0.0.1",
    "max_workers": 1,
    "max_tokens": 1000,
    "truncate_cot": True,
    "extra_params": {},
    "selected_models": [],
    "selected_models_aux": [],
    "selected_attacks": [],
    "selected_classifiers": [],
    "classifier_model": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Environment settings sidebar
st.sidebar.header("Environment Settings")
api_keys = [x.value for x in EnvironmentVariables]
new_env_key = st.sidebar.selectbox("Name", options=api_keys)
new_env_value = st.sidebar.text_input("Value")
if st.sidebar.button("Add Variable"):
    if new_env_key and new_env_value:
        st.session_state.env_vars[new_env_key] = new_env_value

for x in EnvironmentVariables:
    if x.value in os.environ:
        st.session_state.env_vars[x.value] = os.environ[x.value]

# container for env key table
with st.sidebar.container():
    if st.session_state.env_vars:
        cols = st.columns([2, 2, 1])
        cols[0].markdown("**Key**")
        cols[1].markdown("**Value**")
        cols[2].markdown("**Action**")
        for key, value in dict(st.session_state.env_vars).items():
            col1, col2, col3 = st.columns([2, 2, 1])
            col1.text(key)
            masked_value = value[:8] + "..."
            col2.text(masked_value)
            if col3.button("❌", key=f"delete_{key}"):
                del st.session_state.env_vars[key]
                st.rerun()

# classifier model selector
st.sidebar.header("Classifier Model")
if st.session_state.selected_models_aux:
    classifier_model = st.sidebar.selectbox(
        "Select Classifier Model (optional)",
        options=st.session_state.selected_models_aux,
        index=None if st.session_state.classifier_model is None
        else st.session_state.selected_models_aux.index(st.session_state.classifier_model)
    )
    st.session_state.classifier_model = classifier_model
else:
    st.sidebar.selectbox(
        "Select Classifier Model (optional)",
        options=["No aux models available"],
        disabled=True
    )
    st.session_state.classifier_model = None

# fuzzy (zynq) settings
st.sidebar.header("ZYNQ Settings")
st.session_state.verbose = st.sidebar.checkbox("Verbose Logging", value=st.session_state.verbose)
st.session_state.db_address = st.sidebar.text_input("MongoDB Address", value=st.session_state.db_address)
st.session_state.max_workers = st.sidebar.number_input("Max Workers", min_value=1, value=st.session_state.max_workers)
st.session_state.max_tokens = st.sidebar.number_input("Max Tokens", min_value=1, value=st.session_state.max_tokens)

# -----------------------
# APP FLOW (multi-step)
# -----------------------
if 'step' not in st.session_state:
    st.session_state.step = 1

# Step 1: Model selection
if st.session_state.step == 1:
    ollama_models: list[str] = []

    def on_model_select(category, select_key, models: str):
        def on_change():
            st.session_state[models].append(f"{category}/{st.session_state[select_key]}")
        return on_change

    st.markdown("<div class='big-title'>ZYNQ</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted small'>Zero-Knowledge Intelligence Quotient — Verifiable model red-teaming</div>", unsafe_allow_html=True)
    st.write("")
    st.header("Step 1 — Model Selection")
    st.subheader("Choose target and auxiliary models")

    model_options = {provider.value: llm_provider_fm[provider].get_supported_models() for provider in LLMProvider}

    category = st.selectbox("Select Model Category", options=model_options.keys(), index=None)
    if category == "ollama":
        ollama_models = get_ollama_models()
        model_options[category] = ollama_models

    if category:
        st.selectbox(f"Select {category} Models", options=model_options[category], index=None,
                     key='model', on_change=on_model_select(category, 'model', 'selected_models'))

    st.session_state.selected_models = st.multiselect(
        "Selected Models",
        options=st.session_state.selected_models,
        default=st.session_state.selected_models
    )

    st.subheader("Auxiliary models (optional)")
    category_aux = st.selectbox("Select Model Category", options=model_options.keys(), key="cat_aux", index=None)

    if category_aux == "ollama" and not ollama_models:
        model_options[category_aux] = get_ollama_models()

    if category_aux:
        st.selectbox(f"Select {category_aux} Models", options=model_options[category_aux],
                     index=None, key='model_aux', on_change=on_model_select(category_aux, 'model_aux', 'selected_models_aux'))

    st.session_state.selected_models_aux = st.multiselect(
        "Selected Auxiliary Models",
        options=st.session_state.selected_models_aux,
        default=st.session_state.selected_models_aux
    )

    if st.button("Next"):
        if not st.session_state.selected_models:
            st.error("Please select at least one model")
            st.stop()
        st.session_state.step = 2
        st.rerun()

# Step 2: attack selection
elif st.session_state.step == 2:
    st.header("Step 2 — Attack Selection")
    attack_modes = {mode.value: attack_handler_fm[mode].description() for mode in FuzzerAttackMode}
    selected_attacks = st.multiselect("Select Attack Modes", options=attack_modes.keys(), format_func=lambda x: f"{x} - {attack_modes[x]}")

    # "List attack extra" now calls into attack handler classes (no subprocess)
    if st.button("List attack extra"):
        if not selected_attacks:
            st.error("Please select at least one attack mode")
            st.stop()
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
            st.session_state.step -= 1
            st.rerun()
    with col2:
        if st.button("Next"):
            if not selected_attacks:
                st.error("Please select at least one attack mode")
                st.stop()
            # validate extra params
            if st.session_state.extra_params:
                try:
                    for kvp in st.session_state.extra_params.split("\n"):
                        if "=" not in kvp:
                            st.error("Invalid extra parameters format")
                            st.stop()
                        k, v = kvp.split("=", 1)
                except Exception:
                    st.error("Invalid extra parameters format")
                    st.stop()
            st.session_state.step = 3
            st.rerun()

# Step 3: classifiers
elif st.session_state.step == 3:
    st.header("Step 3 — Classifier Selection")
    classifiers = {classifier.value: classifiers_fm[classifier].description() for classifier in Classifier}
    selected_classifiers = st.multiselect("Select Classifiers", options=classifiers.keys(), format_func=lambda x: f"{x} - {classifiers[x]}")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back"):
            st.session_state.step -= 1
            st.rerun()
    with col2:
        if st.button("Next"):
            st.session_state.selected_classifiers = selected_classifiers
            st.session_state.step = 4
            st.rerun()

# Step 4: prompt selection
elif st.session_state.step == 4:
    st.header("Step 4 — Prompt")
    st.subheader("Enter the target prompt (or a small instruction)")
    prompt = st.text_area("Enter prompt", height=140)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back"):
            st.session_state.step -= 1
            st.rerun()
    with col2:
        if st.button("Next"):
            st.session_state.prompt = prompt
            st.session_state.step = 5
            st.rerun()

# Step 5: execution — direct call to Fuzzer (no CLI)
elif st.session_state.step == 5:
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
            st.session_state.step -= 1
            st.rerun()
    with col2:
        run_audit = st.button("Run ZYNQ Audit")
    with col3:
        if st.button("Restart"):
            st.session_state.step = 1
            st.rerun()

    if run_audit:
        # prepare extra params
        ep = {}
        if st.session_state.extra_params:
            for kvp in st.session_state.extra_params.split("\n"):
                if "=" in kvp:
                    k, v = kvp.split("=", 1)
                    ep[k.strip()] = v.strip()

        # instantiate Fuzzer
        db_addr = st.session_state.db_address
        fuzzer = Fuzzer(db_address=db_addr, max_workers=st.session_state.max_workers)

        # add classifiers
        if st.session_state.selected_classifiers:
            for c in st.session_state.selected_classifiers:
                try:
                    fuzzer.add_classifier(classifiers_fm[Classifier(c)](**ep))
                except Exception as e:
                    st.warning(f"Could not add classifier {c}: {e}")

        # add models (target + aux)
        for model in list(set(st.session_state.selected_models + st.session_state.selected_models_aux)):
            try:
                fuzzer.add_llm(model, **ep)
            except Exception as e:
                st.warning(f"Failed to add model {model}: {e}")

        # prepare attack modes
        try:
            attacks = [FuzzerAttackMode(a) for a in st.session_state.selected_attacks]
        except Exception as e:
            st.error(f"Invalid attack mode selected: {e}")
            attacks = []

        # run the fuzzer (async)
        spinner = st.spinner("Running ZYNQ audit — this may take a while …")
        progress = st.progress(0)
        try:
            # run fuzz in a thread-safe way — asyncio.run should be OK for one-off
            report, raw_results = asyncio.run(
                fuzzer.fuzz(attack_modes=attacks, model=list(set(st.session_state.selected_models)), prompts=st.session_state.prompt, **ep)
            )

            spinner.success("Fuzzing completed")

            # create canonical result JSON (deterministic) for future proofing
            combined = {
                "attack_id": getattr(fuzzer, "_attack_id", None),
                "attack_time": getattr(fuzzer, "_attack_time", None),
                "report": report.model_dump() if hasattr(report, "model_dump") else json.loads(report),
                "raw_results": [r.model_dump() if hasattr(r, "model_dump") else json.loads(r) for r in raw_results]
            }
            combined_json_str = json.dumps(combined, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

            # show summary & raw results
            st.subheader("Summary Report")
            st.json(combined["report"])

            st.subheader("Raw results (sample)")
            if combined["raw_results"]:
                st.json(combined["raw_results"][0])
            else:
                st.write("No raw results returned.")

            # download button for full JSON
            st.download_button(
                label="Download full audit (JSON)",
                data=combined_json_str,
                file_name=f"zynq_audit_{combined['attack_id'] or datetime.utcnow().isoformat()}.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"Audit run failed: {e}")
        finally:
            # cleanup LLM connections
            try:
                asyncio.run(fuzzer.cleanup())
            except Exception:
                pass

