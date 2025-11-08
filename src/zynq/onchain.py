# src/zynq/onchain.py
# Usage:
#   python src/zynq/onchain.py --proof outputs/verify/Janus_proof.json --public outputs/verify/Janus_public.json

import json
import os
import argparse
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account

load_dotenv()

RPC_URL = os.getenv("RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")  # hex private key, 0x...
VERIFIER_ADDRESS = os.getenv("VERIFIER_ADDRESS")  # deployed JanusVerifier
ANCHOR_ADDRESS = os.getenv("ANCHOR_ADDRESS")      # deployed ProofAnchor (after deploy)

if not RPC_URL or not PRIVATE_KEY:
    raise SystemExit("Set RPC_URL and PRIVATE_KEY in .env")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
acct = Account.from_key(PRIVATE_KEY)
CHAIN_ID = w3.eth.chain_id

# Minimal ABI for the Janus verifier (view) - replace with full ABI if desired
VERIFIER_ABI = [
    {
        "inputs": [
            {"internalType": "uint256[2]", "name": "_pA", "type": "uint256[2]"},
            {"internalType": "uint256[2][2]", "name": "_pB", "type": "uint256[2][2]"},
            {"internalType": "uint256[2]", "name": "_pC", "type": "uint256[2]"},
            {"internalType": "uint256[1]", "name": "_pubSignals", "type": "uint256[1]"}
        ],
        "name": "verifyProof",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# Minimal ABI for the ProofAnchor contract
ANCHOR_ABI = [
    {
        "inputs": [{"internalType": "bytes32", "name": "proofHash", "type": "bytes32"}],
        "name": "anchorProof",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "proofHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "anchorer", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "blockNumber", "type": "uint256"}
        ],
        "name": "ProofAnchored",
        "type": "event"
    }
]

def load_snarkjs_proof(proof_path):
    j = json.load(open(proof_path, "r", encoding="utf-8"))
    # Accept top-level proof or { "proof": { ... } } shape
    proof_obj = j.get("proof", j)
    # snarkjs typical keys: pi_a, pi_b, pi_c each containing arrays of strings or numbers
    pi_a = proof_obj.get("pi_a") or proof_obj.get("piA") or proof_obj.get("a")
    pi_b = proof_obj.get("pi_b") or proof_obj.get("piB") or proof_obj.get("b")
    pi_c = proof_obj.get("pi_c") or proof_obj.get("piC") or proof_obj.get("c")

    def to_int(x):
        if isinstance(x, str):
            # snarkjs sometimes includes decimal or hex strings
            return int(x, 0)
        return int(x)

    # Normalize A
    a = [to_int(pi_a[0]), to_int(pi_a[1])]

    # Normalize B -- expect [[b00,b01],[b10,b11]] (snarkjs outputs nested arrays)
    if isinstance(pi_b[0], list):
        b = [[to_int(pi_b[0][0]), to_int(pi_b[0][1])], [to_int(pi_b[1][0]), to_int(pi_b[1][1])]]
    else:
        # fallback flatten case
        b = [[to_int(pi_b[0]), to_int(pi_b[1])], [to_int(pi_b[2]), to_int(pi_b[3])]]

    c = [to_int(pi_c[0]), to_int(pi_c[1])]
    return a, b, c, j

def load_public_signals(pub_path):
    j = json.load(open(pub_path, "r", encoding="utf-8"))
    # Expect an array of numbers
    return [int(x) for x in j]

def compute_proof_hash(*filepaths):
    m = hashlib.sha256()
    for p in filepaths:
        with open(p, "rb") as fh:
            m.update(fh.read())
    return m.hexdigest()

def main(args):
    proof_path = Path(args.proof)
    public_path = Path(args.public)

    if not proof_path.exists() or not public_path.exists():
        raise SystemExit("Proof or public JSON not found at given paths")

    print("Loading proof...")
    a, b, c, raw_proof_json = load_snarkjs_proof(str(proof_path))
    pub_signals = load_public_signals(str(public_path))

    # compute hash to anchor
    proof_hash_hex = compute_proof_hash(str(proof_path), str(public_path))
    proof_hash_bytes32 = Web3.toBytes(hexstr=proof_hash_hex)

    print("Connecting to chain:", RPC_URL)
    verifier_addr = Web3.toChecksumAddress(VERIFIER_ADDRESS) if VERIFIER_ADDRESS else None
    anchor_addr = Web3.toChecksumAddress(ANCHOR_ADDRESS) if ANCHOR_ADDRESS else None

    if not verifier_addr:
        print("WARNING: VERIFIER_ADDRESS not set in .env - skipping onchain verification step.")
    else:
        verifier = w3.eth.contract(address=verifier_addr, abi=VERIFIER_ABI)
        # For solidity verifyProof(uint256[2],uint256[2][2],uint256[2],uint256[1])
        # SOME verifier circuits expect different public input lengths. Adjust if needed.
        print("Calling verifier.verifyProof (view)...")
        try:
            is_valid = verifier.functions.verifyProof(a, b, c, pub_signals).call()
            print("Verifier returned:", is_valid)
        except Exception as e:
            print("Verifier call failed:", e)
            is_valid = False

    if not anchor_addr:
        raise SystemExit("ANCHOR_ADDRESS not set in .env — deploy ProofAnchor and set ANCHOR_ADDRESS")
    anchor = w3.eth.contract(address=anchor_addr, abi=ANCHOR_ABI)

    # If verifier exists and returned True, proceed. Otherwise confirm user wants to anchor anyway.
    if verifier_addr and not is_valid:
        print("Verifier did NOT accept the proof. Aborting anchor.")
        return

    # Build tx to anchor
    tx_nonce = w3.eth.get_transaction_count(acct.address)
    try:
        gas_est = anchor.functions.anchorProof(proof_hash_bytes32).estimateGas({"from": acct.address})
    except Exception:
        gas_est = 200000  # fallback

    tx = anchor.functions.anchorProof(proof_hash_bytes32).buildTransaction({
        "chainId": CHAIN_ID,
        "from": acct.address,
        "nonce": tx_nonce,
        "gas": int(gas_est * 1.2),
        "gasPrice": w3.eth.gas_price
    })

    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    print("Anchor tx sent:", tx_hash.hex())
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=600)
    print("Receipt:", receipt)
    print(f"Anchored proof hash {proof_hash_hex} in tx {tx_hash.hex()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof", required=True, help="Path to proof JSON (snarkjs output)")
    parser.add_argument("--public", required=True, help="Path to public signals JSON")
    args = parser.parse_args()
    main(args)
