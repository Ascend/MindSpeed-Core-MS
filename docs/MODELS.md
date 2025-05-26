## MindSpeed-Core-MS 支持模型全集

<table>
<caption>自然语言模型列表</caption>
  <thead>
    <tr>
      <th>模型</th>
      <th>参数</th>
      <th>序列</th>
      <th>实现</th>
      <th>集群</th>
      <th>NPU性能</th>
      <th>参考性能</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/THUDM">GLM4</a></td>
      <td rowspan="2"><a href="https://huggingface.co/THUDM/glm-4-9b">9B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td> 32K </td>
      <th>Mcore</th>
      <td> 2x8 </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://huggingface.co/meta-llama">LLaMA2</td>
      <td rowspan="3"><a href="https://huggingface.co/daryl149/llama-2-7b-hf/tree/main">7B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td>16K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td>32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/meta-llama">LLaMA3</td>
      <td><a href="https://huggingface.co/unsloth/llama-3-8b/tree/main">8B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://modelscope.cn/organization/LLM-Research">LLaMA3.1</td>
      <td rowspan="2"><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B">8B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td>128K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td><a href="https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-70B">70B</a></td>
      <td>8K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/meta-llama">LLaMA3.2</td>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-1B/tree/main">1B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/unsloth/Llama-3.2-3B/tree/main">3B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td>8x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">Qwen1.5</a></td>
      <td> <a href="https://huggingface.co/Qwen/Qwen1.5-1.8B/tree/main">1.8B</a> </td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/Qwen">Qwen2</a></td>
      <td rowspan="1"><a href="https://huggingface.co/Qwen/Qwen2-7B/tree/main">7B</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td rowspan="5"><a href="https://huggingface.co/Qwen">Qwen2.5</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main">1.5B</a></td>
      <td> 32K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td> </td>
    </tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-3B/tree/main">3B</a></td>
      <td> 32K </td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-7B/tree/main">7B</a></td>
      <td>32K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-14B/tree/main">14B</a></td>
      <td>32K</td>
      <th>Mcore</th>
      <td>2x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-32B/tree/main">32B</a></td>
      <td>32K</td>
      <th>Mcore</th>
      <td>4x8</td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://github.com/OpenBMB/MiniCPM">MiniCPM</a></td>
      <td> <a href="https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16/tree/main">2B</a> </td>
      <td> 4K </td>
      <th> Mcore </th>
      <td> 1x8 </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite">DeepSeek-V2-Lite</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/main">16B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 1x8 </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2">DeepSeek-V2</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2/tree/main">236B</a></td>
      <td> 8K </td>
      <th>Mcore</th>
      <td> 20x8 </td>
      <td> </td>
      <td> </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2">DeepSeek-V3</a></td>
      <td><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2/tree/main">671B</a></td>
      <td> 4K </td>
      <th>Mcore</th>
      <td> 64x8 </td>
      <td> </td>
      <td> </td>
    </tr>
  </tbody>
</table>

<table>
  <a id="jump1"></a>
  <caption>多模态模型列表</caption>
  <thead>
    <tr>
      <th>模型任务</th>
      <th>模型</th>
      <th>参数量</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
      <th>NPU性能</th>
      <th>参考性能</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"> 多模态生成 </td>
      <td rowspan="2"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/cogvideox">CogVideoX-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td>  </td>
    </tr>
    <tr>
    <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 亲和场景 </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> </td>
      <td>  </td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/cogvideox">CogVideoX-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td>  </td>
    </tr>
    <tr>
    <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 亲和场景 </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td>  </td>
    </tr>
    <tr>
      <td rowspan="7"> 多模态理解 </td>
      <td><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/llava1.5">LLaVA 1.5</a></td>
      <td><a href="https://github.com/haotian-liu/LLaVA">7B</a></td>
      <td>全参微调</td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td>  </td>
    </tr>
   <tr>
      <td rowspan="3"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/internvl2">Intern-VL-2.0</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-2B">2B</a></td>
      <td>微调</td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td>  </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-8B">8B</a></td>
      <td>微调</td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td>  </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B">76B</a></td>
      <td> 全参微调 </td>
      <td> 8x16 </td>
      <td> BF16 </td>
      <td>  </td>
      <td>  </td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://gitee.com/ascend/MindSpeed-MM/tree/master/examples/qwen2vl">Qwen2-VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct">2B</a></td>
      <td>微调</td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td>  </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct">7B</a></td>
      <td>微调</td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td>  </td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct">72B</a></td>
      <td>微调</td>
      <td> 8x16 </td>
      <td> BF16 </td>
      <td>  </td>
      <td>  </td>
    </tr>
    </tbody>
</table>