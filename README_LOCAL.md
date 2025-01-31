# Generative Agents

在原项目的基础上，我们使用 Ollama 在本地部署 embedding 模型 mxbai-embed-large，生成模型使用api key的方式来调用，来运行 Generative Agents 项目。

## 环境

```bash
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.5 LTS
Release:        22.04
Codename:       jammy
```

## Ollama的安装和模型的本地部署

1. 在Linux系统安装Ollama

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

2. 使用Ollama 安装embedding模型， 可安装[模型列表](https://ollama.com/search?c=embedding)

```sh
# mxbai-embed-large 
ollama pull mxbai-embed-large
```

## 代码修改

1. 主要修改文件 reverie/backend_server/persona/prompt_template/gpt_structure.py, 可参考文件[gpt_structure.py](https://github.com/web3-liber/generative_agents/blob/main/reverie/backend_server/persona/prompt_template/gpt_structure.py)

2. 在文件开头添加变量

```python 
# Ollama API 配置
OLLAMA_API_URL = "http://localhost:11434"
MODEL = "deepseek/deepseek-chat" # 使用 api key 调用
EMBEDDING_MODEL = "mxbai-embed-large:latest" # 本地部署调用
```

3. 在generative_agents/reverie/backend_server/utils.py文件中添加变量

```python
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
```

4. 修改函数 ChatGPT_single_request(prompt), GPT4_request(prompt), ChatGPT_request(prompt)。

原代码：
```python
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
  )
  return completion["choices"][0]["message"]["content"]
```

替换代码：
```python
    response = requests.post(
      url=OPENROUTER_BASE_URL,
      headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        # "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
      },
      data=json.dumps({
        "model": MODEL, # Optional
        "messages": [
          {
            "role": "user",
            "content": prompt
          }
        ]
        
      })
    )    
    response_json = response.json()
    return response_json["choices"][0]["message"]["content"]  
```

5. 修改函数GPT_request(prompt, gpt_parameter)

原代码：
```python
    response = openai.Completion.create(
                model=gpt_parameter["engine"],
                prompt=prompt,
                temperature=gpt_parameter["temperature"],
                max_tokens=gpt_parameter["max_tokens"],
                top_p=gpt_parameter["top_p"],
                frequency_penalty=gpt_parameter["frequency_penalty"],
                presence_penalty=gpt_parameter["presence_penalty"],
                stream=gpt_parameter["stream"],
                stop=gpt_parameter["stop"],)
    return response.choices[0].text
```

替换代码：
```python
    response = requests.post(
      url=OPENROUTER_BASE_URL,
      headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        # "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
      },
      data=json.dumps({
        "model": MODEL, # Optional
        "messages": [
          {
            "role": "user",
            "content": prompt
          }
        ]
        
      })
    )    
    response_json = response.json()
    return response_json["choices"][0]["message"]["content"]
```

6. 修改embeding 模型

原代码：

```python
def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  return openai.Embedding.create(
          input=[text], model=model)['data'][0]['embedding']
```

替换代码：
```python
def get_embedding(text, model=EMBEDDING_MODEL):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  
  response = ollama.embeddings(model=model, prompt=text)
  embedding = response["embedding"]
  return embedding
```

## 运行

运行部分不需要做任何修改，可参考原[README](https://github.com/joonspk-research/generative_agents/blob/main/README.md)文件, 

## Other

#### 本地部署生成模型

##### 部署操作

1. 使用Ollama 安装生成模型，  可安装[模型列表](https://ollama.com/search) 

```sh
# llama3 参数为 8B
ollama run llama3

# llama3.3 参数为 70B
ollama run llama3.3
```

##### 代码修改

1. 主要修改文件 reverie/backend_server/persona/prompt_template/gpt_structure.py, 可参考文件[gpt_structure.py](https://github.com/web3-liber/generative_agents/blob/main/reverie/backend_server/persona/prompt_template/gpt_structure.py)

2. 在文件开头添加变量

```python 
# Ollama API 配置
OLLAMA_API_URL = "http://localhost:11434"
MODEL = "llama3:latest"  # 如果使用llama3.3, 则为"llama3.3:latest"
EMBEDDING_MODEL = "mxbai-embed-large:latest"
```

3. 修改函数 ChatGPT_single_request(prompt), GPT4_request(prompt), ChatGPT_request(prompt)。

原代码：
```python
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
  )
  return completion["choices"][0]["message"]["content"]
```

替换代码：
```python
  completion = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
  )
  return completion['message']['content']
```

4. 修改函数GPT_request(prompt, gpt_parameter)

原代码：
```python
    response = openai.Completion.create(
                model=gpt_parameter["engine"],
                prompt=prompt,
                temperature=gpt_parameter["temperature"],
                max_tokens=gpt_parameter["max_tokens"],
                top_p=gpt_parameter["top_p"],
                frequency_penalty=gpt_parameter["frequency_penalty"],
                presence_penalty=gpt_parameter["presence_penalty"],
                stream=gpt_parameter["stream"],
                stop=gpt_parameter["stop"],)
    return response.choices[0].text
```

替换代码：
```python
  completion = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
  )
  return completion['message']['content']
```



