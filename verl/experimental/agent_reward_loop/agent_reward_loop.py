# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig
from pydantic import BaseModel
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.torch_functional import pad_2d_list_to_length
from verl.workers.rollout.async_server import async_server_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: List[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
    ) -> List[int]:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            List[int]: List of generated token ids.
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: List[int]
    response_ids: List[int]
    response_mask: List[int]
    num_turns: int = 0
    metrics: AgentLoopMetrics


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(self, config: DictConfig, server_manager: AsyncLLMServerManager, tokenizer: AutoTokenizer):
        """Initialize agent loop.

        Args:
            config (DictConfig): YAML config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.config = config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()
        self.init_class(config, tokenizer)

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any]) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            messages (List[Dict[str, Any]]): Input messages.
            sampling_params (Dict[str, Any]): LLM sampling params.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


class RewardLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    compute_score: float = 0.0


class RewardLoopOutput(BaseModel):
    """Agent loop output."""

    reward_list: List[float]
    reward_extra_info: Dict[str, Any]
    metrics: RewardLoopMetrics


class RewardLoopBase(ABC):
    _class_initialized = False

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer, compute_score):
        """Initialize agent loop.

        Args:
            config (DictConfig): YAML config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.loop = asyncio.get_running_loop()
        self.init_class(config, tokenizer, compute_score)

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, compute_score):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(
        self, data_source: str, response_ids: List[int], ground_truth: str, extra_info: Dict[str, Any]
    ) -> RewardLoopOutput:
        raise NotImplementedError


@ray.remote
class AgentRewardLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(self, config: DictConfig, server_handles: List[ray.actor.ActorHandle]):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.compute_score = get_custom_reward_fn(config)

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        n = 1 if batch.meta_info.get("validate", False) else config.n
        tasks = []

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        agent_names = batch.non_tensor_batch["agent_name"].repeat(n, axis=0)
        raw_prompts = batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0)
        data_sources = batch.non_tensor_batch["data_source"].repeat(n, axis=0)
        ground_truths = np.array(
            [item.non_tensor_batch["reward_model"]["ground_truth"] for item in batch], dtype=object
        ).repeat(n, axis=0)
        extra_infos = batch.non_tensor_batch["extra_info"].repeat(n, axis=0)
        for agent_name, messages, data_source, ground_truth, extra_info in zip(
            agent_names, raw_prompts, data_sources, ground_truths, extra_infos
        ):
            agent_loop_args = dict(
                agent_name=agent_name,
                messages=messages.tolist(),
                sampling_params=sampling_params,
            )
            reward_name = self.config.reward_model.reward_manager
            reward_loop_args = dict(
                reward_name=reward_name,
                data_source=data_source,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            tasks.append(asyncio.create_task(self._run_agent_reward_loop(agent_loop_args, reward_loop_args)))
        outputs = await asyncio.gather(*tasks)
        agent_loop_outputs, reward_loop_outputs = zip(*outputs)

        output = self._postprocess_agent_reward_loop(agent_loop_outputs, reward_loop_outputs)
        return output

    async def _run_agent_reward_loop(
        self,
        agent_loop_kwargs,
        reward_loop_kwargs,
    ):
        agent_loop_output = await self._run_agent_loop(**agent_loop_kwargs)
        response_ids = agent_loop_output.response_ids
        # response_str = self.tokenizer.decode(agent_loop_output.response_ids, skip_special_tokens=True)
        reward_loop_output = await self._run_reward_loop(response_ids=response_ids, **reward_loop_kwargs)
        return agent_loop_output, reward_loop_output

    async def _run_agent_loop(
        self, agent_name: str, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any]
    ) -> AgentLoopOutput:
        agent_loop_class = self.get_agent_loop_class(agent_name)
        agent_loop = agent_loop_class(self.config, self.server_manager, self.tokenizer)
        output = await agent_loop.run(messages, sampling_params)
        return output

    async def _run_reward_loop(
        self,
        reward_name: str,
        response_ids: List[int],
        data_source: str,
        ground_truth: str,
        extra_info: Dict[str, Any],
    ):
        reward_loop_class = self.get_reward_class(reward_name)
        reward_loop = reward_loop_class(self.config, self.tokenizer, self.compute_score)
        output = await reward_loop.run(data_source, response_ids, ground_truth, extra_info)
        return output

    def get_agent_loop_class(self, agent_name: str) -> Type[AgentLoopBase]:
        # TODO: add tool agent registrary
        from verl.experimental.agent_reward_loop.single_turn_agent_loop import SingleTurnAgentLoop
        from verl.experimental.agent_reward_loop.tool_agent_loop import ToolAgentLoop

        if agent_name == "single_turn_agent":
            return SingleTurnAgentLoop
        elif agent_name == "tool_agent":
            return ToolAgentLoop
        raise ValueError(f"Unknown agent_name: {agent_name}")

    def get_reward_class(self, reward_name: str):
        from verl.experimental.agent_reward_loop.dapo_reward_loop import DAPORewardLoop

        if reward_name == "dapo":
            return DAPORewardLoop
        raise ValueError(f"Unknown reward_name: {reward_name}")

    def _postprocess_agent_reward_loop(
        self, inputs_agent: List[AgentLoopOutput], inputs_reward: List[RewardLoopOutput]
    ) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        # prompts
        self.tokenizer.padding_side = "left"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.prompt_ids} for input in inputs_agent],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_ids, prompt_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # responses
        self.tokenizer.padding_side = "right"
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_ids} for input in inputs_agent],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        response_ids, response_attention_mask = outputs["input_ids"], outputs["attention_mask"]

        # response_mask
        outputs = self.tokenizer.pad(
            [{"input_ids": input.response_mask} for input in inputs_agent],
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )

        response_mask = outputs["input_ids"]
        assert response_ids.shape == response_mask.shape, (
            f"mismatch in response_ids and response_mask shape: {response_ids.shape} vs {response_mask.shape}"
        )
        response_mask = response_mask * response_attention_mask

        # reward_tensor
        rewards_list = [input.reward_list for input in inputs_reward]
        reward_tensor = pad_2d_list_to_length(
            rewards_list, 0, max_length=self.config.actor_rollout_ref.rollout.response_length
        )
        assert response_ids.shape == reward_tensor.shape, (
            f"mismatch in response_ids and reward_tensor shape: {response_ids.shape} vs {reward_tensor.shape}"
        )
        reward_tensor = reward_tensor * response_attention_mask

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "reward_tensor": reward_tensor,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(input_ids),
        )

        num_turns = np.array([input.num_turns for input in inputs_agent], dtype=np.int32)
        # agent_metrics = [input.metrics.model_dump() for input in inputs_agent]
        # reward_metrics = [input.metrics.model_dump() for input in inputs_reward]
        metrics = []
        reward_extra_infos = []
        for input_agent, input_reward in zip(inputs_agent, inputs_reward):
            agent_metric = input_agent.metrics.model_dump()
            reward_metric = input_reward.metrics.model_dump()
            metrics.append({**agent_metric, **reward_metric})
            reward_extra_infos.append(input_reward.reward_extra_info)
        reward_extra_infos = np.array(reward_extra_infos, dtype=object)

        return DataProto(
            batch=batch,
            non_tensor_batch={"__num_turns__": num_turns, "reward_extra_infos": reward_extra_infos},
            meta_info={"metrics": metrics},
        )


class AgentRewardLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        self.sleep()

    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        for i in range(self.config.actor_rollout_ref.rollout.agent.num_workers):
            self.agent_loop_workers.append(
                AgentRewardLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                ).remote(self.config, self.async_llm_servers)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [worker.generate_sequences.remote(chunk) for worker, chunk in zip(self.agent_loop_workers, chunkes)]
        )
        output = DataProto.concat(outputs)
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # calculate performance metrics
        metrics = [output.meta_info["metrics"] for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing}
        return output

    def _performance_metrics(self, metrics: List[List[Dict[str, str]]], output: DataProto) -> Dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric["compute_score"] for chunk in metrics for metric in chunk])
        timing["agent_reward_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_reward_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_reward_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_reward_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_reward_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_reward_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_reward_loop/compute_score/min"] = t_compute_score.min()
        timing["agent_reward_loop/compute_score/max"] = t_compute_score.max()
        timing["agent_reward_loop/compute_score/mean"] = t_compute_score.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls + t_compute_score)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_reward_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_reward_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_reward_loop/slowest/compute_score"] = t_compute_score[slowest]
        timing["agent_reward_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_reward_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])
