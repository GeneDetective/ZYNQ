// scripts/deploy_anchor.js
const hre = require("hardhat");

async function main() {
  console.log("Deploying ProofAnchor...");
  const ProofAnchor = await hre.ethers.getContractFactory("ProofAnchor");
  const anchor = await ProofAnchor.deploy();
  await anchor.deployed();

  console.log("ProofAnchor deployed at:", anchor.address);
  console.log("Add this address as ANCHOR_ADDRESS in your .env if needed.");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
