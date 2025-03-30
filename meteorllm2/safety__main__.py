# Credits to mistral creators! Apache 2.0 license
# Copyright 2024 DeepMind Technologies Limited.
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
# Credits to deepmind for half the code

import torch
import jax
import jax.numpy as jnp
import flax
import orbax.checkpoint
import requests
import os
from collections import defaultdict
from fuzzywuzzy import process
import streamlit as st
from typing import Iterable, Optional
import functools
import concurrent.futures  # <-- For safely running heavy tasks (DeepSeek-like)

# -----------------------------
# Utility Functions for Mixed Precision and Parameter Loading
# -----------------------------

Params = dict[str, any]

_SIGLIP_PARAMS_PATH = 'not supported'

def prepare_mixed_precision(
    params: Iterable[torch.nn.Parameter],
    param_dtype: torch.dtype,
    optim_dtype: torch.dtype,
):
    """Appends a freshly allocated fp32 tensor copy of all params to parameters that can be updated."""
    with torch.no_grad():
        for p in params:
            if p.requires_grad:
                p._mp_param = torch.empty_like(p, dtype=optim_dtype)
                p._mp_param.copy_(p.to(optim_dtype))  # type: ignore

            p.data = p.data.to(param_dtype)


def upcast_mixed_precision(
    params: Iterable[torch.nn.Parameter], optim_dtype: torch.dtype
):
    """Ensure all weights and optimizer states are updated in fp32."""
    with torch.no_grad():
        for p in params:
            if p.requires_grad and p.grad is not None:
                p._temp = p.data  # type: ignore
                p.data = p._mp_param  # type: ignore
                p.grad = p.grad.to(optim_dtype)


def downcast_mixed_precision(
    params: Iterable[torch.nn.Parameter], param_dtype: torch.dtype
):
    """Reverts the mixed precision after optimizer step."""
    with torch.no_grad():
        for p in params:
            if p.requires_grad and p.grad is not None:
                p._temp.copy_(p.data)  # type: ignore
                p.data = p._temp  # type: ignore
                p.grad = p.grad.to(param_dtype)


def load_siglip_params(checkpoint_path: str = _SIGLIP_PARAMS_PATH) -> Params:
    """Loads SigLIP parameters."""
    params = load_params(checkpoint_path)['donated_carry']['params']
    out_params = {}
    for key in params:
        new_key = str(key).replace('SigLiPFromPatches_0/', '')
        if 'MlpBlock' in new_key:
            new_key = new_key.replace('Dense', 'DenseGeneral')
        out_params[new_key] = params[key]
    return nest_params(out_params)


def load_and_format_params(path: str, load_siglip: bool = False) -> Params:
    """Loads parameters and formats them for compatibility."""
    params = load_params(path)
    param_state = jax.tree_util.tree_map(jnp.array, params)
    remapped_params = param_remapper(param_state)
    nested_params = nest_params(remapped_params)
    if load_siglip:
        nested_params['transformer']['vision_encoder'] = load_siglip_params()
    return nested_params


def load_metadata(path: str) -> Optional[any]:
    """Loads metadata from a checkpoint path."""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    metadata = checkpointer.metadata(path)
    return metadata


@functools.cache
def load_params(path: str) -> Params:
    """Loads parameters from a checkpoint path."""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    params = checkpointer.restore(path)
    return params


def format_and_save_params(params: Params, path: str) -> None:
    """Formats and saves a parameter checkpoint to the path."""
    params = flatten_and_remap_params(params)
    save_params(params, path)


def save_params(params: Params, path: str) -> None:
    """Saves the given parameters to the given path."""
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer.save(path, params)


def param_remapper(orig_params: Params) -> Params:
    """Remaps params to new module layout."""
    new_params = {}
    for k, v in orig_params.items():
        if 'mlp/' in k:
            layer_name, param = k.rsplit('/', maxsplit=1)
            if layer_name not in new_params:
                new_params[layer_name] = {}
            if 'w' in v:
                new_params[layer_name][param] = v['w']
        else:
            new_params[k] = v
    return new_params


def nest_params(params: Params) -> Params:
    """Nests params as a dict of dicts."""
    nested_params = {}
    for path, param in params.items():
        *path, leaf = path.split('/')
        subdict = nested_params
        for key in path:
            subdict = subdict.setdefault(key, {})
        subdict[leaf] = param
    return nested_params


def flatten_and_remap_params(params: Params) -> Params:
    """Flattens and remaps params from new to old module layout."""
    params = flax.traverse_util.flatten_dict(params, sep='/')

    def remap_name(n: str):
        if n.endswith('/mlp/linear') or n.endswith('/mlp/gating_einsum'):
            n += '/w'
        left, right = n.rsplit('/', maxsplit=1)
        return left + '&' + right

    params = {remap_name(k): v for k, v in params.items()}
    return flax.traverse_util.unflatten_dict(params, sep='&')


# -----------------------------
# Chatbot and Search Functions
# -----------------------------

DICT_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
response = requests.get(DICT_URL)
if response.status_code == 200:
    english_words = set(response.text.split())
else:
    english_words = set()

def load_instructions():
    if os.path.exists("system.txt"):
        with open("system.txt", "r", encoding="utf-8") as file:
            return file.read().strip()
    return "No instructions provided."

instructions = load_instructions()

def search_duckduckgo(query):
    query = "+".join(query.split())  # Format query
    url = f"https://api.duckduckgo.com/?q={query}&format=json&t=h_"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        try:
            data = response.json()
            if "RelatedTopics" in data:
                results = [topic["Text"] for topic in data["RelatedTopics"] if "Text" in topic]
                return results[:5]  # Limit to top 5 results
        except ValueError:
            return []
    return []


def check_dictionary(user_input):
    tokens = user_input.lower().split()
    valid_words = [word for word in tokens if word in english_words]
    return valid_words if valid_words else None


def detect_language(user_input):
    languages = ['python', 'java', 'javascript', 'c', 'c++', 'ruby', 'go', 'swift', 'typescript', 'php', 'html', 'css']
    for language in languages:
        if language in user_input.lower():
            return language
    return 'python'  # Default to Python if no language is found


# --- DeepSeek Integration ---
def search_deepseek(query: str, timeout: int = 5) -> str:
    """
    Runs DeepSeek's heavy search code in a safe manner,
    ensuring that if the computation takes too long, it is canceled.
    """
    def deepseek_task(q):
        # Here would be the real DeepSeek code.
        # For demonstration, we simulate a heavy computation.
        import time
        time.sleep(2)  # simulate heavy processing delay
        return f"Simulated DeepSeek results for query: '{q}'."
    
    # Run the deepseek task in a separate thread with a timeout.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(deepseek_task, query)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            future.cancel()
            return "DeepSeek search timed out. Try refining your query."
        except Exception as e:
            return f"DeepSeek encountered an error: {str(e)}"


knowledge_base = defaultdict(list)

def add_to_knowledge_base(query, response):
    knowledge_base[query].append(response)

def get_best_match(query):
    if knowledge_base:
        matches = process.extract(query, knowledge_base.keys(), limit=3)  # Get top 3 best matches
        for match in matches:
            if match[1] >= 80:  # Only consider high-confidence matches
                return knowledge_base[match[0]]
    return None


def chatbot_response(user_input, context=None, retries=3):
    """
    Returns a chatbot response.
    If the input is prefixed with 'deepseek:', it will use the DeepSeek search function.
    Otherwise, it falls back to DuckDuckGo or previously stored results.
    """
    user_input = user_input.strip().lower()

    attempt = 0
    while attempt < retries:
        match_response = get_best_match(user_input)
        if match_response:
            return "💡 I found something similar in my knowledge base:\n" + "\n".join(match_response)

        language = detect_language(user_input)

        if user_input in knowledge_base:
            response = "💬 I've got some info on that in my knowledge base:\n" + "\n".join(knowledge_base[user_input])
            return response

        # If the user explicitly asks for DeepSeek results with the prefix "deepseek:"
        if user_input.startswith("deepseek:"):
            query = user_input[len("deepseek:"):].strip()
            ds_results = search_deepseek(query)
            response = "🔍 DeepSeek results:\n" + ds_results
            add_to_knowledge_base(user_input, response)
            return response

        ddg_results = search_duckduckgo(user_input)
        if ddg_results:
            response = "🔍 Here's what I found from the web for you:\n" + "\n".join(ddg_results)
            add_to_knowledge_base(user_input, response)
            return response

        attempt += 1
        if attempt == retries:
            return "Hmm, I’m still not sure about that, but I’ll keep learning. 😕"

    return "I couldn’t find an answer this time, but feel free to ask something else!"


# -----------------------------
# Streamlit App Setup
# -----------------------------

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title('AI Chatbot')
st.write("Got a question? Ask me anything, and I'll search for the best info online!")

st.subheader("Chatbot Instructions")
st.text(instructions)

context = {}

user_input = st.text_input("You: ")

if user_input:
    response = chatbot_response(user_input, context)
    st.write(f"AI: {response}")
    context[user_input] = response  # Store last question as context
