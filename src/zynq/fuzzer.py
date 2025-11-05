import asyncio
import logging
import os
import time
import json
import hashlib
from datetime import datetime
from typing import Any, Optional, overload
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

from zynq.consts import DATETIME_FORMAT
from zynq.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from zynq.handlers.attacks.enums import FuzzerAttackMode
from zynq.handlers.attacks.proto import AttackSummary, BaseAttackTechniqueHandlerProto
from zynq.handlers.classifiers.base import BaseClassifier
from zynq.handlers.db.adv_attacks import AdversarialAttacksHandler
from zynq.handlers.db.adv_prompts import AdversarialPromptDTO, AdversarialPromptsHandler
from zynq.handlers.db.adv_suffixes import AdversarialSuffixesHandler
from zynq.llm.providers.base import BaseLLMProvider
from zynq.llm.providers.enums import LLMProvider
from zynq.models.fuzzer_result import FuzzerResult
from zynq.utils.utils import llm_provider_factory

logger = logging.getLogger(__name__)


class Fuzzer:
    """
    Core orchestration class for running model fuzzing / red-team attacks.

    Notes:
      - This class remains mostly identical to the original FuzzyAI implementation,
        with a few convenience helpers added for deterministic export and hashing
        (useful later for zero-knowledge / audit flows).
      - The added helpers are non-destructive: they only read returned objects and
        serialize them into deterministic JSON.
    """

    def __init__(self, db_address: str, cleanup: bool = True, **extra: Any) -> None:
        """
        Initialize the Fuzzer class.

        Args:
            db_address (str): Address of the MongoDB (host or IP).
            cleanup (bool): whether to close attack handler / llm resources after run.
            **extra: additional parameters forwarded to attack handlers / providers.
        """
        self._extra = extra
        self._classification_batch_size = 150

        # note: AsyncIOMotorClient will connect lazily; port is 27017 by default here
        self._mongo_client: AsyncIOMotorClient = AsyncIOMotorClient(db_address, 27017)  # ignore: type
        self._adv_prompts_handler = AdversarialPromptsHandler(self._mongo_client)
        self._adv_suffixes_handler = AdversarialSuffixesHandler(self._mongo_client)
        self._adv_attacks_handler = AdversarialAttacksHandler(self._mongo_client)

        self._llms: list[BaseLLMProvider] = []
        self._classifiers: list[BaseClassifier] = []
        
        self._attack_id = str(uuid4())
        self._attack_time = datetime.now().strftime(DATETIME_FORMAT)
        self._cleanup = cleanup
        
        logger.info(f"ZYNQ: Initiating Attack ID: {self._attack_id}, Attack Time: {self._attack_time}, DB Address: {db_address}")

    def add_classifier(self, classifier: BaseClassifier, **extra: Any) -> None:
        """
        Add a new classifier instance to the fuzzer.
        """
        self._classifiers.append(classifier)

    def add_llm(self, provider_and_model: str, **extra: Any) -> None:
        """
        Add a new LLM provider.

        Args:
            provider_and_model (str): provider/model_name (e.g., 'ollama/qwen2.5' or 'openai/gpt-4').
        """
        if '/' not in provider_and_model:
            raise RuntimeError(f"Model {provider_and_model} not in correct format, please use provider/model_name format")

        provider_name, model = provider_and_model.split('/', 1)

        # NOTE: original code expects provider_name membership checks against LLMProvider enum
        if provider_name not in LLMProvider.__members__.values():
            # keep behavior consistent with upstream; log and continue
            # (this check is not always accurate - it mirrors upstream code)
            logger.debug(f"Provider membership check skipped or failed for: {provider_name}")

        is_valid_model: bool = True

        if provider_name == LLMProvider.LOCAL:
            is_valid_model = os.path.isdir(model)
        elif provider_name == LLMProvider.REST:
            is_valid_model = os.path.isfile(model)

        if not is_valid_model:
            logger.error(f"Model {model} not found for {provider_name.lower()} provider")
            return

        logger.debug(f"Adding {model} model on {provider_name}")
        llm = llm_provider_factory(provider=LLMProvider(provider_name), model=model, **extra)
        self._llms.append(llm)

    def get_llm(self, model: str) -> Optional[BaseLLMProvider]:
        """
        Get the LLM provider for the given model.
        """
        for llm in self._llms:
            if llm.qualified_model_name == model:
                return llm
        return None
    
    # ----------------
    # Typing overloads for fuzz
    # ----------------
    @overload
    async def fuzz(self, attack_modes: list[FuzzerAttackMode], model: list[str], prompts: str, **extra: Any) -> tuple[FuzzerResult, list[AttackSummary]]:
        ...

    @overload
    async def fuzz(self, attack_modes: list[FuzzerAttackMode], model: list[str], **extra: Any) -> tuple[FuzzerResult, list[AttackSummary]]:
        ...

    async def fuzz(self, attack_modes: list[FuzzerAttackMode], model: list[str], prompts: Optional[list[str] | str] = None, **extra: Any) -> tuple[FuzzerResult, list[AttackSummary]]:
        """
        Fuzz multiple prompts.

        If prompts is None, prompts are retrieved from the configured prompts collection in the DB.
        If prompts is a str, it's treated as a single prompt.
        If prompts is a list of strings, each entry is treated as a prompt.
        """
        prompts_dto: list[AdversarialPromptDTO]  = []

        if prompts is None:
            prompts_dto = await self._adv_prompts_handler.retrieve()
        elif isinstance(prompts, str):
            prompts_dto = [AdversarialPromptDTO(prompt=prompts)]
        elif isinstance(prompts, list):
            prompts_dto = [AdversarialPromptDTO(prompt=prompt) for prompt in prompts]

        return await self._fuzz(prompts_dto, model, attack_modes, **extra)
    
    async def cleanup(self) -> None:
        """
        Cleanup the fuzzer: close any LLM providers.
        """
        await asyncio.gather(*[llm.close() for llm in self._llms])

    def _attack_technique_factory(self, attack_mode: FuzzerAttackMode, model: str, **extra: Any) -> BaseAttackTechniqueHandlerProto:
        """
        Factory method to create an instance of the attack technique handler.
        """
        handler_cls: type[BaseAttackTechniqueHandler[BaseModel]] = attack_handler_fm[attack_mode]
        if (auxiliary_models := handler_cls.default_auxiliary_models()) is not None:
            for auxiliary_model in auxiliary_models:
                if not any(llm.qualified_model_name == auxiliary_model for llm in self._llms):
                    logger.info(f"Attack mode {attack_mode} defines a default auxiliary model {auxiliary_model}. Automatically adding it - see wiki for more details.")
                    self.add_llm(auxiliary_model)

        return attack_handler_fm[attack_mode](llms=self._llms, model=model,
                                              classifiers=self._classifiers, **extra)
    
    async def _fuzz(self, prompts: list[AdversarialPromptDTO], models: list[str],
                    attack_modes: list[FuzzerAttackMode], **extra: Any) -> tuple[FuzzerResult, list[AttackSummary]]:
        """
        Fuzz the given prompts.

        Returns:
            (FuzzerResult, list[AttackSummary])
        """
        raw_results: list[AttackSummary] = []
        attack_handler: Optional[BaseAttackTechniqueHandlerProto] = None
        
        logger.info('ZYNQ: Starting fuzzer...')
        
        start_time = time.time()

        # Verify that all models has been added by add_llm
        for model in models:
            if not any(llm.qualified_model_name == model for llm in self._llms):
                logger.error(f"Model {model} has not been added to the fuzzer, skipping...")
                continue

            for attack_mode in attack_modes:
                logger.info(f'Attacking {len(prompts)} prompts with attack mode: {attack_mode} for model: {model}...')
                extra.update(**self._extra)
                attack_handler = self._attack_technique_factory(attack_mode, model=model, **extra)

                attack_result: Optional[AttackSummary] = await attack_handler.attack(prompts)
                if attack_result:
                    attack_result.attack_mode = attack_mode
                    attack_result.model = model
                    attack_result.system_prompt = extra.get('system_prompt', 'No system prompt set')
                    logger.info(f'Finished attacking {len(prompts)} prompts for attack mode {attack_mode}')
                    raw_results.append(attack_result)
                else:
                    logger.error(f'Failed to attack {len(prompts)} prompts for attack mode {attack_mode}')
        
        logger.info('Done, took %s seconds', time.time() - start_time)
        
        if attack_handler is not None and self._cleanup:
            await asyncio.gather(attack_handler.close())

        report = FuzzerResult.from_attack_summary(self._attack_id, raw_results)
        return report, raw_results

    # ----------------
    # Convenience helpers for deterministic export & hashing
    # ----------------
    def _canonicalize_result(self, result: dict) -> str:
        """
        Return canonicalized JSON string used to hash/export results. Uses sorted keys
        and no extra whitespace to ensure deterministic output.
        """
        return json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def get_result_hash(self, canonical_json_str: str) -> str:
        """
        Return a sha256 hex string prefixed with 0x for easy on-chain compatibility later.
        """
        digest = hashlib.sha256(canonical_json_str.encode("utf-8")).hexdigest()
        return "0x" + digest

    async def fuzz_and_return_json(self, *args: Any, **kwargs: Any) -> tuple[str, FuzzerResult, list[AttackSummary]]:
        """
        Convenience helper that runs fuzz(...) and returns:
          (canonical_json_str, report, raw_results)

        The canonical_json_str is deterministic and suitable for later hashing or proof generation.
        """
        report, raw_results = await self.fuzz(*args, **kwargs)

        combined = {
            "attack_id": self._attack_id,
            "attack_time": self._attack_time,
            # model_dump() returns python-native structures from Pydantic
            "report": report.model_dump() if hasattr(report, "model_dump") else report,
            "raw_results": [r.model_dump() if hasattr(r, "model_dump") else r for r in raw_results]
        }

        canon = self._canonicalize_result(combined)
        return canon, report, raw_results
