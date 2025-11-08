// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title ProofAnchor
/// @notice Lightweight contract to anchor a proof hash on-chain (emits an event).
contract ProofAnchor {
    /// Emitted when a proof hash is anchored
    event ProofAnchored(bytes32 indexed proofHash, address indexed anchorer, uint256 blockNumber);

    /// Anchor a proof hash on-chain. Emits ProofAnchored.
    /// @param proofHash - SHA256 (or other) of the proof/public JSON (or combined).
    function anchorProof(bytes32 proofHash) external returns (bool) {
        emit ProofAnchored(proofHash, msg.sender, block.number);
        return true;
    }
}
