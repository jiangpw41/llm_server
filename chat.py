import json
import logging
import os
import random
import sys
import time
from typing import Any, Union

import aiohttp
import requests
from openai import AsyncOpenAI, OpenAI
from tenacity import before_sleep_log, retry, wait_fixed

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

logger = logging.getLogger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class ChatClient:
    model_client_dict = {
        "openai_client": [
            OpenAI(base_url="https://aigc.sankuai.com/v1/openai/native ", api_key="1913043036402708564"),
            OpenAI(base_url="https://aigc.sankuai.com/v1/openai/native ", api_key="1640232516982226983"),
            OpenAI(base_url="https://aigc.sankuai.com/v1/openai/native ", api_key="1889606952393760828"),
            OpenAI(base_url="https://aigc.sankuai.com/v1/openai/native ", api_key="1913043036402708564"),
            OpenAI(base_url="https://aigc.sankuai.com/v1/openai/native ", api_key="1913043036402708564"),
            OpenAI(base_url="https://aigc.sankuai.com/v1/openai/native ", api_key="1913043036402708564"),
            ],
        }
    async_client_dict = {
        "openai_client": [
            AsyncOpenAI(base_url="https://aigc.sankuai.com/v1/openai/native ", api_key="1913043036402708564"),
            AsyncOpenAI(base_url="https://aigc.sankuai.com/v1/openai/native ", api_key="1640232516982226983"),
            AsyncOpenAI(base_url="https://aigc.sankuai.com/v1/openai/native ", api_key="1889606952393760828"),
            AsyncOpenAI(base_url="https://aigc.sankuai.com/v1/openai/native ", api_key="1913043036402708564"),
            ],
        }

    def add_model(self, model_name: str, address: str = "local_host", port = "8000", api_key="token-abc123"):
        port = str(port)
        if model_name not in self.model_client_dict:
            self.model_client_dict[model_name] = []
        self.model_client_dict[model_name].append(OpenAI(base_url=f"http://{address}:{port}/v1", api_key=api_key))
        if model_name not in self.async_client_dict:
            self.async_client_dict[model_name] = []
        self.async_client_dict[model_name].append(AsyncOpenAI(base_url=f"http://{address}:{port}/v1", api_key=api_key))

    @retry(wait=wait_fixed(1), before_sleep=before_sleep_log(logger, logging.WARNING))
    def get_response_log(
            self,
            query: Union[str, list[str]],
            instruct: str = None,
            model: str = "gpt-4o-speech",
            **kwargs,
            ) : #  dict[str, Any] | list[dict[str, Any]]
        client_list = self.model_client_dict.get(model, self.model_client_dict["openai_client"])
        client = random.choice(client_list)
        if client_list == self.model_client_dict["openai_client"] and "gpt" not in model:
            import logging
            logging.warning("The model name is not gpt, please check the model name")

        possible_keys = [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "steam_options",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "user",
            "extra_body",
            "timeout",
            ]

        kwargs = {k: v for k, v in kwargs.items() if k in possible_keys}
        print(kwargs)
        if isinstance(query, str):
            query = [query]
        query_ = []
        if instruct is not None:
            query_.append({"role": "system", "content": instruct})

        for idx, item in enumerate(query):
            if idx % 2 == 0:
                query_.append({"role": "user", "content": item})
            else:
                query_.append({"role": "assistant", "content": item})
        completion = client.chat.completions.create(model=model, messages=query_, **kwargs)

        return_dict = []
        for idx, item in enumerate(completion.choices):
            response_content = {"response": item.message.content, "usage": completion.usage}
            if "logprobs" in kwargs:
                response_content["logprobs"] = item.logprobs.content[0].top_logprobs
            return_dict.append(response_content)
            if not kwargs.get("n"):
                break
        # print({"query": query, "answer": return_dict[0]["response"]})
        return return_dict if kwargs.get("n") else return_dict[0]
    
    # @retry(wait=wait_fixed(1), retry=retry_if_exception_type(openai.RateLimitError), before_sleep=before_sleep_log(logger, logging.WARNING))
    # @retry(wait=wait_fixed(1), retry=retry_if_exception_type(openai.RateLimitError))
    async def async_get_response(
            self,
            query: Union[str, list[str]],
            model_name: str,
            instruct: str = None,
            **kwargs,
            ) : # -> dict[str, Any] | list[dict[str, Any]]
        client_list = self.async_client_dict.get(model_name, self.async_client_dict["openai_client"])
        client = random.choice(client_list)
        # if client_list == self.model_client_dict["openai_client"] and "gpt" not in model_name:
        #     import logging
        #     logging.warning("The model name is not gpt, please check the model name")

        possible_keys = [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "steam_options",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "user",
            "extra_body",
            "timeout",
            ]

        kwargs = {k: v for k, v in kwargs.items() if k in possible_keys}

        if isinstance(query, str):
            query = [query]
        query_ = []
        if instruct is not None:
            query_.append({"role": "system", "content": instruct})

        for idx, item in enumerate(query):
            if idx % 2 == 0:
                query_.append({"role": "user", "content": item})
            else:
                query_.append({"role": "assistant", "content": item})
        completion = await client.chat.completions.create(model=model_name, messages=query_, **kwargs)

        return_dict = []
        for idx, item in enumerate(completion.choices):
            response_content = {"response": item.message.content, "usage": completion.usage, }
            if hasattr(item.message, "reasoning_content"):
                response_content["reasoning_content"] = item.message.reasoning_content
            if "logprobs" in kwargs:
                response_content["logprobs"] = item.logprobs.content[0].top_logprobs
            return_dict.append(response_content)
            if not kwargs.get("n"):
                break
        # print({"query": query, "answer": return_dict[0]["response"]})
        return return_dict if kwargs.get("n") else return_dict[0]

    async def async_get_response_batch(self, queries, instructs, model_name, **kwargs):
        """
        批量获取模型响应
        :param queries: 输入查询列表
        :param instructs: 指令列表
        :param model_name: 模型名称
        :param temperature: 生成温度
        :param max_tokens: 最大生成长度
        :return: 批量响应结果
        """
        responses = []
        for query, instruct in zip(queries, instructs):
            try:
                response = await self.async_get_response(
                    query=query,
                    instruct=instruct,
                    model_name=model_name,
                    **kwargs
                )
                responses.append(response)
            except Exception as e:
                print(f"[Batch Request Error] Query: {query}, Error: {e}")
                responses.append({"response": None})  # 返回空响应以保持结果对齐
        return responses

    # @retry(wait=wait_fixed(0.1))
    def get_response(
            self,
            query: Union[str, list[str]],
            model_name: str,
            instruct: str = None,
            **kwargs,
            ) : # -> dict[str, Any] | list[dict[str, Any]]
        client_list = self.model_client_dict.get(model_name, self.model_client_dict["openai_client"])
        client = random.choice(client_list)
        if client_list == self.model_client_dict["openai_client"] and "gpt" not in model_name:
            import logging
            logging.warning("The model name is not gpt, please check the model name")
        # print(kwargs)
        possible_keys = [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "steam_options",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "user",
            "extra_body",
            "timeout",
            ]

        kwargs = {k: v for k, v in kwargs.items() if k in possible_keys}
        if isinstance(query, str):
            query = [query]
        query_ = []
        if instruct is not None:
            query_.append({"role": "system", "content": instruct})

        for idx, item in enumerate(query):
            if idx % 2 == 0:
                query_.append({"role": "user", "content": item})
            else:
                query_.append({"role": "assistant", "content": item})
        completion = client.chat.completions.create(model=model_name, messages=query_, **kwargs)
        return_dict = []
        for idx, item in enumerate(completion.choices):
            response_content = {"response": item.message.content, "usage": completion.usage}
            if hasattr(item.message, "reasoning_content"):
                response_content["reasoning_content"] = item.message.reasoning_content
            if "logprobs" in kwargs:
                response_content["logprobs"] = item.logprobs.content[0].top_logprobs
            return_dict.append(response_content)
            if not kwargs.get("n"):
                break
        # print({"query": query, "answer": return_dict[0]["response"]})
        return return_dict if kwargs.get("n") else return_dict[0]

    @retry(wait=wait_fixed(1))
    def get_embedding(
            self,
            query: Union[str, list[str]],
            instruct: str = None,
            model: str = "gpt-4o-speech",
            ) -> float:
        def post_http_request(prompt: dict, api_url: str) -> dict:
            headers = {"User-Agent": "Test Client"}
            response = requests.post(api_url, headers=headers, json=prompt)
            return response.json()

        client_list = self.model_client_dict.get(model, self.model_client_dict["openai_client"])
        client: OpenAI = random.choice(client_list)
        if client_list == self.model_client_dict["openai_client"] and "gpt" not in model:
            import logging
            logging.warning("The model name is not gpt, please check the model name")

        if isinstance(query, str):
            query = [query]
        query_ = []
        if instruct is not None:
            query_.append({"role": "system", "content": instruct})

        for idx, item in enumerate(query):
            if idx % 2 == 0:
                query_.append({"role": "user", "content": item})
            else:
                query_.append({"role": "assistant", "content": item})
        completion = post_http_request(prompt={
            "model": model,
            "messages": query_,
            }, api_url=f"http://{client.base_url.host}:{client.base_url.port}/pooling")

        return completion["data"][0]["data"][0]

    def batch_add_model(self, model_name, wait=False, address_port_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/guoshiguang02/code/vllm_deploy"):
        if wait:
            while True:
                file_name = os.path.join(address_port_path, model_name.replace("/", "_") + "_0.txt")
                if os.path.exists(file_name):
                    with open(file_name) as f:
                        address_list = json.loads(f.read())
                    timestamp = address_list["timestamp"]
                    if time.time() - timestamp < 60:
                        break
                time.sleep(60 * 2)

        address_list = []
        machine_rank = 0
        while True:
            file_name = os.path.join(address_port_path, model_name.replace("/", "_") + f"_{machine_rank}.txt")
            if os.path.exists(file_name):
                with open(file_name) as f:
                    address_list += f.readlines()
                machine_rank += 1
            else:
                break

        for item in address_list:
            address = json.loads(item).get("address", None)
            port_list = [str(8000 + i) for i in range(json.loads(item).get("gpus_per_node", 8) // json.loads(item).get("vllm_gpus", 4))]
            if address:
                for port in port_list:
                    self.add_model(model_name=model_name, address=address, port=port)
        return len(self.model_client_dict[model_name])

    def get_instance_num(self, model_name):
        return len(self.model_client_dict[model_name])