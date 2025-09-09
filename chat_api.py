import json
import time
from typing import List, Union
import requests
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
root_dir = os.path.dirname(os.path.abspath(__file__))


class LlmHttpClient:
    def __init__(self, appid: Union[List[str], str]):
        if isinstance(appid, str):
            self.appids = [appid]
        else:
            self.appids = appid
        if "." in self.appids[0]:
            self.is_friday = False
        else:
            self.is_friday = True
        self.set_all_url_headers()
        self.failed_instances = set()
        self.failed_instances_cnt_map = dict()

    def set_all_url_headers(self):
        self.urls_and_headers = list()
        self.model = list()
        if self.is_friday:
            for appid in self.appids:
                url = "https://aigc.sankuai.com/v1/openai/native/chat/completions "
                headers = {"Content-Type": "application/json;charset=UTF-8", "Authorization": "Bearer " + appid}
                self.urls_and_headers.append((url, headers))
        else:
            for ip in self.appids:
                port = 44389
                while True:
                    url = "http://{ip}:{port}/health".format(ip=ip, port=port)
                    try:
                        r = requests.get(url, timeout=1)
                        if r.status_code == 200:
                            url = "http://{ip}:{port}/v1/chat/completions".format(ip=ip, port=port)
                            headers = {"Content-Type": "application/json;charset=UTF-8"}
                            self.urls_and_headers.append((url, headers))
                            self.model.append(str(port))
                            port += 1
                            continue
                        else:
                            break
                    except Exception as e:
                        break
        self.services_num = len(self.urls_and_headers)
        self.services_order = 0

    def chat(self, content='', system_prompt='', messages=None, text="", model: str = None, max_tokens: int = 512,
             temperature: float = 0.95, top_p: float = 0.95, n: int = 1, stop: list = [], **kwargs):
        try:
            assert self.services_num > 0 and len(self.failed_instances) < self.services_num, "没有可用的服务"
            assert model is not None or len(self.model) > 0, "没有设置模型"
        except Exception as e:
            return
        ids = -1
        if "ids" in kwargs:
            ids = kwargs["ids"]
            del kwargs["ids"]
        url, headers = self.urls_and_headers[self.services_order]
        model = model if self.is_friday else self.model[self.services_order]
        services_order_cur = self.services_order
        while True:
            self.services_order = (self.services_order + 1) % self.services_num
            if self.services_order not in self.failed_instances:
                break
        if messages is None:
            if len(system_prompt) == 0:
                messages = [{"role": "user", "content": content.strip()}]
            else:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
        request = {"messages": messages, "model": model, "max_tokens": max_tokens, "temperature": temperature,
                   "top_p": top_p, "n": n, "stop": stop}
        request.update(kwargs)
        try:
            r = requests.post(url, json=request, headers=headers, timeout=kwargs.get("timeout", 30),
                              stream=kwargs.get("stream", False))
        except:
            r = None
        # 单条调用时支持输出stream
        if kwargs.get("stream", False):
            return self.stream_process(r)
        else:
            try:
                message = r.json()['choices'][0]['message']
                #logger.info(f"message:{message}")
                thought = message.get('reasoning_content')
                answer = message.get('content', "")
                if thought is None:
                    response = answer
                else:
                    response = json.dumps({"thought": thought, "answer": answer}, ensure_ascii=False)
                if services_order_cur in self.failed_instances_cnt_map:
                    self.failed_instances_cnt_map[services_order_cur] = 0
            except:
                response = None
                if services_order_cur not in self.failed_instances_cnt_map:
                    self.failed_instances_cnt_map[services_order_cur] = 1
                else:
                    self.failed_instances_cnt_map[services_order_cur] += 1
                if self.failed_instances_cnt_map[services_order_cur] >= 100:
                    self.failed_instances.add(services_order_cur)
            if ids >= 0:
                result = (ids, response)
            else:
                result = response
            return result
        
    def chat_batch(self, messages_list: list, contents: list = [], system_prompt='', model: str = None,
                   max_tokens: int = 2048, temperature: float = 0.95, top_p: float = 0.95, n: int = 1, stop: list = [],
                   rpm=10, timeout: int = 60):
        '''
        注意：message_list是必要参数，可以直接指定为空列表[]
        '''
        assert isinstance(messages_list, list), "messages_list是个列表"
        assert self.services_num > 0, "没有可用的服务"
        assert model is not None or len(self.model) > 0, "没有设置模型"
        if len(messages_list) == 0:
            for content in contents:
                if len(system_prompt) == 0:
                    messages = [{"role": "user", "content": content.strip()}]
                else:
                    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
                messages_list.append(messages)
        else:
            assert isinstance(messages_list, list) and isinstance(messages_list[0], list) and isinstance(
                messages_list[0][0], dict) and set(messages_list[0][0].keys()) == {"role",
                                                                                   "content"}, "messages_list格式错误"
        tasks = list()
        results = list()
        with ThreadPoolExecutor(max_workers=rpm) as executor:
            for ids,messages in tqdm(enumerate(messages_list),total=len(messages_list)):
                tasks.append(executor.submit(self.chat, messages=messages, model=model, max_tokens=max_tokens,
                                             temperature=temperature, top_p=top_p, n=n, stop=stop, ids=ids,
                                             timeout=timeout))
                time.sleep(60 / rpm)
            for task in as_completed(tasks):
                data = task.result()
                results.append(data)
        results = sorted(results, key=lambda x: x[0])
        results = [result[1] for result in results]

        return results

    def stream_process(self, respponse):
        answer_pre = ""
        thought = ""
        for chunk in respponse.iter_lines():
            chunk = chunk.decode('utf-8')
            if chunk.startswith("data:"):
                chunk = chunk[5:].strip()
                if chunk == "[DONE]":
                    break
                chunk = json.loads(chunk)
                answer = chunk["content"].strip()
                finish_reason = chunk["choices"][0]["finish_reason"]
                thought += chunk["choices"][0]["delta"].get("reasoning_content", "").strip()
            if (len(answer) > 0 and answer != answer_pre) or finish_reason is not None or len(thought) > 0:
                answer_pre = answer
                if len(thought) == 0:
                    result = json.dumps({"content": answer, "finish_reason": finish_reason}, ensure_ascii=False)
                else:
                    result = json.dumps({"thought": thought, "content": answer, "finish_reason": finish_reason},
                                        ensure_ascii=False)
                yield result
                if finish_reason is not None:
                    break

# 现有API服务的调用
APP_ID = "1913043036402708564"
ds_r1 = ["Doubao-deepseek-r1", "deepseek-r1-friday", "deepseek-v3-friday", "deepseek-chat"]
Qwq = ["QwQ-32B-Friday"]


def llm_api_response(user_inputs, system_prompt="", app_id="1913043036402708564", model='gpt-4-1106-preview'):
    api_base = 'https://aigc.sankuai.com/v1/openai/native/chat/completions' 
    header = {'Content-type': 'application/json', 'Authorization': 'Bearer {appid}'.format(appid=app_id)}
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_inputs}]
    try:
        data = json.dumps({
            "messages": messages,
            "stream": "false",
            "temperature": 0.0,
            'model': model,
            'max_tokens': 500
        })
        res = requests.post(url=api_base, headers=header, data=data)
        result = json.loads(res.text)
        ret = result['choices'][0]['message']['content']

        # logger.info("[GPT4响应] completion response : {}".format(json.dumps(result, ensure_ascii=False)))
        return ret
    except Exception as e:
        # logger.exception( "[GPT4请求异常]: " + repr(e)
        #                  + "\t\t[GPT4请求参数]" + json.dumps(messages, ensure_ascii=False)
        #                  + "\t\t[GPT4响应结果]")
        print(res.text)
        return None