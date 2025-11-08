// scripts/deploy_verifier.js
const hre = require("hardhat");
const fs = require("fs");
async function main() {
  console.log("Network:", hre.network.name);
  const Verifier = await hre.ethers.getContractFactory("Groth16Verifier"); // contract name in file is Groth16Verifier
  const verifier = await Verifier.deploy();
  await verifier.deployed();
  console.log("Deployed Janus_Verifier at:", verifier.address);
  fs.writeFileSync("deployed_verifier_address.txt", verifier.address, "utf8");
}
main().catch((e) => { console.error(e); process.exit(1); });
