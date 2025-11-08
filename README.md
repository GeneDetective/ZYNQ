# ZYNQ — Zero-Knowledge Intelligence Quotient

**Verifiable model red-teaming + on-chain anchor**

A reproducible, end-to-end system that:

1. Runs automated red-team fuzzing audits (ZYNQ / ZK-JBFuzz style) against LLMs.
2. Produces deterministic audit outputs, generates a ZK proof (Circom / snarkjs), and publishes a compact proof anchor on-chain (Hardhat / Solidity).
3. Provides a Streamlit UI to run audits, save results, and optionally anchor proofs to a blockchain.

> This README tells you everything to get the repo running, how to compile & deploy the smart contracts, how to run the audit UI, how to generate/verify proofs and anchor them on Sepolia, plus troubleshooting notes.

---

## Table of contents

1. Project summary
2. Requirements & tools
3. Recommended repo layout (you already have this)
4. Quickstart — clone & install
5. Environment variables (`.env`) — example and explanation
6. Compile & deploy smart contracts (Hardhat)
7. Proof generation & verification (snarkjs / Circom helpers)
8. Run the Streamlit UI and perform audits
9. Anchor proof on-chain (how it works + commands)
10. Troubleshooting & common errors
11. Security & privacy notes
12. License & credits

---

## 1 — Project summary (short)

ZYNQ runs adversarial prompt discovery against a target model, serializes the deterministic audit results, generates a zero-knowledge proof that the audit took place (without revealing secret prompt content), and optionally anchors a proof/public-signal pair to an EVM chain for public, cryptographic verification.

You can:

* run a fuzz audit locally via Streamlit UI,
* generate/verify ZK proofs (scripts provided),
* deploy verifier & anchor contracts to Sepolia (Hardhat),
* push proof anchors on-chain via an `onchain.py` / deploy script.

---

## 2 — Requirements & tools

### System

* OS: Windows / Linux / macOS (commands shown use PowerShell / bash variants as noted)
* Node >= 16 (the user environment above used Node v22.x but Node 18+ is typical)
* npm (comes with Node) — or pnpm if preferred
* Python 3.10+ (the project used 3.12 in your environment)

### Node packages (dev)

* hardhat (we used `hardhat@2.27.0` for compatibility in this repo)
* @nomiclabs/hardhat-ethers
* ethers (v5)
* snarkjs (for proving / verifying)
* optionally: pnpm (if you prefer)

We install locally in the repo.

### Python packages

* streamlit
* nest_asyncio
* aiohttp
* motor (async MongoDB driver) — only if using local MongoDB
* pydantic
* python-dotenv
* backoff
* any other packages referenced in `pyproject.toml` or `requirements.txt` in the repo

> If there is a `pyproject.toml` or `requirements.txt` already in the repo, install from that.

---

## 3 — Recommended repo layout (what you already copied)

```
ZYNQ/
├─ circuits/                       # Circom source
│  └─ Janus.circom
├─ outputs/                         # compiled wasm, zkey, r1cs, etc (optional precomputed)
│  ├─ Janus_js/Janus.wasm
│  ├─ keys/Janus_final.zkey
│  └─ verify/Janus_proof.json
│  └─ verify/Janus_public.json
├─ scripts-circom/                 # scripts to generate & verify proofs (node)
│  └─ main.js
│  └─ utils/generateProof.js
│  └─ utils/verify.js
├─ contracts/                       # solidity contracts
│  └─ Janus_Verifier.sol
│  └─ ProofAnchor.sol (anchor contract)
├─ src/                            # Python app (zynq)
│  └─ zynq/ ...
│  └─ webui.py
│  └─ fuzzer.py
├─ scripts/                         # hardhat deploy scripts
│  └─ deploy_verifier.js
│  └─ deploy_anchor.js
├─ verify/                          # proof outputs (expected paths)
│  └─ Janus_proof.json
│  └─ Janus_public.json
├─ package.json
├─ hardhat.config.cjs
├─ .env
└─ README.md   <-- (you are editing this)
```

---

## 4 — Quickstart — clone & install

```bash
# clone
git clone https://github.com/GeneDetective/ZYNQ.git
cd ZYNQ
```

### Node side (smart contract & proof scripts)

Install node deps (from repo root):

```bash
# using npm (preferred if you used npm above)
npm install
# If package.json uses a specific local hardhat version, this will install it locally.
```

If your project expects a specific hardhat version (we used `hardhat@2.27.0` earlier) you can install that explicitly:

```bash
npm install --save-dev hardhat@2.27.0 @nomiclabs/hardhat-ethers ethers dotenv snarkjs
```

### Python side (UI & fuzzing)

Create a virtualenv and install Python deps:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
# if there is no requirements.txt, run:
pip install streamlit nest_asyncio aiohttp motor pydantic python-dotenv backoff
```

If the repo uses Poetry, you can use `poetry install` instead.

---

## 5 — Environment variables (`.env`) — example

Create a `.env` file in the repo root. Example contents:

```ini
# .env (example)
# For circom
CIRCUIT_NAME=circom_startup
TYPE_OF_SNARK=groth16
POWER_OF_TAU=12
ELLIPTIC_CURVE=bn128
BEACON_TAU=0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f
ITERATIONS_TAU=10
BEACON_ZKEY=0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f
ITERATIONS_ZKEY=10

# For Foundry
RPC_URL=https://sepolia.infura.io/v3/
SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/
PRIVATE_KEY=
SENDER_ADDRESS=
ETHERSCAN_API_KEY=
VERIFIER_ADDRESS=
ANCHOR_ADDRESS=
```

**Important**:

* NEVER commit `.env` with real keys to git. Keep it private.
* `PRIVATE_KEY` is the deployer wallet private key used by Hardhat scripts. Provide the hex private key string with `0x` prefix.

---

## 6 — Compile & deploy smart contracts (Hardhat)

### compile contracts

From repo root:

```bash
# ensure local hardhat is installed
npx hardhat compile --config hardhat.config.cjs
```

If `hardhat` complains about ESM/CJS, follow the repo commands:

* remove `"type":"module"` from `package.json` or use `.cjs` for hardhat config (we used `hardhat.config.cjs`).
* If you previously hit `ERR_PACKAGE_PATH_NOT_EXPORTED` install a compatible `hardhat` version (we used local `hardhat@2.27.0`).

### deploy contracts to Sepolia

Make sure `.env` includes `RPC_URL` and `PRIVATE_KEY` and your network is set to `sepolia` in `hardhat.config.cjs`.

Deploy verifier:

```bash
npx hardhat run scripts/deploy_verifier.js --network sepolia
```

This will compile and deploy `Janus_Verifier.sol` and print the deployed address. Save that address in `.env` as `VERIFIER_ADDRESS` or similar if your code references it.

Deploy anchor contract:

```bash
npx hardhat run scripts/deploy_anchor.js --network sepolia
```

This should print the deployed anchor contract address. Save it as `ANCHOR_ADDRESS` in `.env` if desired.

> You already deployed and observed addresses (e.g. `Janus_Verifier` and `ProofAnchor`). Add those addresses to `.env` (the UI / onchain script reads them).

---

## 7 — Proof generation & verification (snarkjs / circom scripts)

If you already have the compiled circuit and zkey (in `outputs/`), you can generate a proof with the provided Node scripts:

### Generate a proof (example)

From repo root:

```bash
# set CIRCUIT_NAME in .env first (e.g., Janus)
# run the proof generation helper (this uses snarkjs.groth16.fullProve)
node scripts-circom/main.js
# or if using npx/node:
node scripts-circom/main.js
```

`main.js` calls `generateProof(inputs)` which writes `outputs/verify/Janus_proof.json` and `outputs/verify/Janus_public.json`.

If you need to recompile circuits (only if you changed `Janus.circom`):

* install `circom` and `snarkjs` and follow Circom compilation steps (out of scope for a minimal README). If you already have `*.wasm` and `*.zkey` in `outputs/`, you are good.

### Verify a proof (locally)

```bash
node scripts-circom/utils/verify.js  # or run a small driver that reads outputs/verify/* and runs verifyProof
```

---

## 8 — Run the Streamlit UI and do an audit

Start Streamlit UI:

```bash
# ensure .env contains keys you intend to use (e.g. GEMINI_API_KEY), or use the UI sidebar Save Variable to persist them
streamlit run src/zynq/webui.py --server.port 8080 --server.address 127.0.0.1
```

UI flow:

1. Step 1 — Model selection: add `gemini/gemini-2.5-flash` (or other provider/model). If your provider registry fails to load, add model manually with `provider/model`.
2. Step 2 — Attack selection (e.g., MANYSHOT)
3. Step 3 — Classifier selection (optional)
4. Step 4 — Prompt: enter the target prompt
5. Step 5 — Run ZYNQ Audit: click Run ZYNQ Audit

What happens:

* `webui` instantiates `Fuzzer`, adds LLM provider(s), runs the attacks, collects results, serializes a deterministic `report` and `raw_results`.
* The UI saves a local `results/<TIMESTAMP>/report.json`.
* If `verify/Janus_proof.json` and `verify/Janus_public.json` are available, the UI will attempt to run `onchain.py` (if present) to anchor the proof. If not found, you can manually click **Anchor proof now** in the UI.

---

## 9 — Anchor proof on-chain (how and commands)

### How it works (mechanism)

1. Fuzzer runs and produces deterministic JSON result. You can canonicalize that JSON, hash it (sha256), and optionally use the hash as a public signal to the ZK proof.
2. The Circom circuit `Janus.circom` takes a private input (the secret adversarial prompt) and produces public signals (e.g., a hash of the prompt or a hash combining prompt and result). The proof proves knowledge of the preimage without revealing it.
3. `snarkjs` produces `proof.json` and `public.json` (public signals). The Solidity `Janus_Verifier.sol` contract verifies the Groth16 proof with public signals.
4. The `ProofAnchor` contract stores or emits an on-chain record: e.g., it might accept the proof and public signal(s), call the verifier, then emit an event with the canonical audit hash or store it in state.
5. The UI runs `onchain.py` (or `scripts/deploy_anchor.js`) which:

   * reads `verify/Janus_proof.json` and `verify/Janus_public.json`,
   * calls the anchor contract (via ethers.py or web3.py or a Node script),
   * transaction result and tx hash printed back to UI.

### Anchor via Node/Hardhat (example)

If you have a Node script called `scripts/anchor_proof.js`, run:

```bash
npx hardhat run scripts/anchor_proof.js --network sepolia
```

Or if using the Python helper `onchain.py`:

```bash
python onchain.py --proof verify/Janus_proof.json --public verify/Janus_public.json
```

(Your `onchain.py` should be coded to read `.env` for RPC_URL, PRIVATE_KEY and `ANCHOR_ADDRESS`).

**After anchoring**:

* Hardhat script / `onchain.py` will print a tx hash and contract response (e.g., event log with anchor id).
* Verify on-chain via Etherscan by pasting the tx hash.

---

## 10 — Troubleshooting & common errors

### 1) `RuntimeError: no running event loop` when creating `aiohttp.ClientSession` in **init**

**Cause**: Creating `aiohttp.ClientSession()` synchronously in constructor without an async loop.
**Fix**: Use a lazy `async _get_session()` method that creates session inside an async function (we updated `gemini/handler.py` earlier). If you see this error, ensure provider sessions are created lazily.

### 2) `ValueError: 'resources/...' must be only a file name`

**Cause**: `importlib.resources.open_text()` expects package + file name (not a path).
**Fix**: Use `resources.open_text("zynq.resources", "llama2-uncensored-prompt-response-t0.jsonl")` or place file in `zynq/resources` and reference only filename.

### 3) `GeminiProviderException: status=429 Quota exceeded`

**Cause**: Google Generative API quota exhausted for your project/region (not a code bug).
**Fix**: Request higher quota in Google Cloud Console, use a different project, or throttle calls. Check your API key and project quotas.

### 4) `models/gemini-1.5-flash is not found for API version`

**Cause**: Wrong API endpoint / model name or using v1beta vs v1. Use the correct model string (e.g., `gemini-2.5-flash` if available) or call ListModels to check supported models for your API key/project.

### 5) Hardhat errors like `No Hardhat config file found`

**Cause**: Hardhat can't find config or package.json type mismatch.
**Fix**: Ensure `hardhat.config.cjs` exists, `package.json` doesn't force ESM (remove `"type": "module"`), and hardhat installed locally. We fixed this earlier by changing package.json and using `.cjs` config.

### 6) `curl` problems on Windows PowerShell

PowerShell treats `\` continuation and `-H` differently. Use `curl.exe` with full JSON in one line or use `Invoke-WebRequest` or run the command in Git Bash.

---

## 11 — Security & privacy notes

* **Do not commit** `.env`, private keys, or proof private inputs to git. Keep them locally.
* The zero-knowledge proof generation step must run in a secure environment if the secret contains real vulnerabilities; `public.json` reveals only public signals (hashes), not secrets.
* The anchor transaction uses your `PRIVATE_KEY` to sign; protect it (hardware wallet or ephemeral key recommended for production).

---

## 12 — Files you might add / ensure exist

* `onchain.py` — Python helper to publish proof/public signals to `ProofAnchor` contract (reads `.env` RPC_URL, PRIVATE_KEY, ANCHOR_ADDRESS/VERIFIER_ADDRESS). If not present, the Streamlit UI looks for it but will not anchor automatically.
* `scripts/anchor_proof.js` — Node/Hardhat script alternative to anchor proofs.
* `outputs/` (optional) — binary artifacts: `.wasm`, `.zkey`, `r1cs` if you plan to avoid compiling circuits locally.

---

## Appendix — Example commands (cheat sheet)

```bash
# Node & Hardhat
npm install
npx hardhat compile --config hardhat.config.cjs
npx hardhat run scripts/deploy_verifier.js --network sepolia
npx hardhat run scripts/deploy_anchor.js --network sepolia

# Proofs (Node)
node scripts-circom/main.js      # generate proof -> outputs/verify/*.json
node scripts-circom/utils/verify.js

# Streamlit UI
streamlit run src/zynq/webui.py --server.port 8080 --server.address 127.0.0.1

# Anchor with python helper (if exists)
python onchain.py --proof verify/Janus_proof.json --public verify/Janus_public.json
```

---
