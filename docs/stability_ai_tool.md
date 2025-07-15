# Stability AI

[Stability AI](https://platform.stability.ai/) builds image, video, 3D, and audio generation models. The Strands Agents SDK implements a Stability AI [tool](https://strandsagents.com/latest/user-guide/concepts/tools/tools_overview/) that can be used by agents to generate images.

## Installation

Stability AI can be added as a tool to Strands Agents. To install the `strands-agents-tools` package, run:

```bash
pip install strands-agents-tools
```

## Configuration
The Stability AI tool supports 3 image generation models:

| Model | `model_id` |
|-------|------------|
| Stable Diffusion 3.5 Large | `stability.sd3-5-large-v1:0` |
| Stable Image Ultra | `stability.stable-image-ultra-v1:1` |
| Stable Image Core | `stability.stable-image-core-v1:1` |


The models are used with API credits. See the [Stability Platform](https://platform.stability.ai/pricing) for pricing details.

You will need to creat an API key on the [Stability Platform](https://platform.stability.ai/).

## Set your Stability AI API key and model_id as environment variables

For example:
```bash
export STABILITY_API_KEY=sk-xxx
export STABILITY_MODEL_ID=stability.stable-image-ultra-v1:1
```
If no `STABILITY_MODEL_ID` is selected, the tool defaults to using `stability.stable-image-core-v1:1`.

If you want to write the images produced by the tool, set the environment variable `STABILITY_OUTPUT_DIR` to a local filepath.

# Use
The Stability AI tool can be given to a Strands agent. This enables the agent to create images. For example, an agent that makes slide decks, marketing content, or social media posts could include images created to match the user's input.

The model that the agent uses must be able to return responses of at least 2MB in order to work with the Stability AI tool.

Use with Strands agent:
```python
import os
from strands import Agent
from strands_tools import generate_image_stability
```

The agent does not need to be passed your API key.

### Create an agent that will use the tool
```python
agent = Agent(tools=[generate_image_stability])
```

### Basic use - the agent only needs to provide the prompt
```python
agent("Generate an image of a futuristic robot in a cyberpunk city")
```

Please see the [Stability AI tool source code](generate_image_stability.py) for details and structure of the object it returns.

For example, you can access the image from the `tool_result`:

```
for message in agent.messages:
    # Look through all messages for tool results
    if 'content' in message:
        for item in message['content']:
            if isinstance(item, dict) and 'toolResult' in item:
                tool_result = item['toolResult']
                if tool_result.get('status') == 'success':
                    for content_item in tool_result['content']:
                        if isinstance(content_item, dict) and 'image' in content_item:
                            image_bytes = content_item['image']['source']['bytes']
```

### Advanced use with custom parameters
```python
agent.tool.generate_image_stability(
    prompt="A serene mountain landscape",
    aspect_ratio="16:9",
    style_preset="photographic",
    cfg_scale=7.0,
    seed=42
)
```


## Tool Parameters

The available parameters that can be passed to the tool are: 


### Required Parameters

- **`prompt`** (string, **required**)
  - The text prompt to generate the image from. Be descriptive for best results.

### Optional Parameters

- **`return_type`** (string, default: `"json"`)
  - The format in which to return the generated image. Use 'image' to return the image data directly, or 'json' to return a JSON object containing the image data as a base64-encoded string.
  - Options: `"json"`, `"image"`

- **`aspect_ratio`** (string, default: `"1:1"`)
  - Controls the aspect ratio of the generated image. This parameter is only valid for text-to-image requests.
  - Options: `"16:9"`, `"1:1"`, `"21:9"`, `"2:3"`, `"3:2"`, `"4:5"`, `"5:4"`, `"9:16"`, `"9:21"`

- **`seed`** (integer, default: `0`)
  - Optional: Seed for random number generation. Use the same seed to reproduce similar results. Omit or use 0 for random generation.
  - Range: 0 to 4294967294

- **`output_format`** (string, default: `"png"`)
  - Output format for the generated image
  - Options: `"jpeg"`, `"png"`, `"webp"`

- **`style_preset`** (string, no default)
  - Applies a predefined style to the output
  - Options: `"3d-model"`, `"analog-film"`, `"anime"`, `"cinematic"`, `"comic-book"`, `"digital-art"`, `"enhance"`, `"fantasy-art"`, `"isometric"`, `"line-art"`, `"low-poly"`, `"modeling-compound"`, `"neon-punk"`, `"origami"`, `"photographic"`, `"pixel-art"`, `"tile-texture"`

- **`cfg_scale`** (number, default: `4.0`)
  - Controls how closely the image follows the prompt (only used with SD3.5 model). Higher values mean stricter adherence to the prompt.
  - Range: 1.0 to 10.0

- **`negative_prompt`** (string, no default)
  - Text describing what you do not want to see in the generated image. Helps exclude unwanted elements or styles.
  - Max length: 10,000 characters

- **`mode`** (string, default: `"text-to-image"`)
  - Mode of operation
  - Options: `"text-to-image"`, `"image-to-image"`

- **`image`** (string, no default)
  - Input image for image-to-image generation (`mode=image-to-image`). Should be base64-encoded image data in jpeg, png or webp format.

- **`strength`** (number, default: `0.5`)
  - For image-to-image mode: controls how much the input image influences the result. 0 = identical to input, 1 = completely new image based on prompt.
  - Range: 0.0 to 1.0

## References

- [Strands Agents Tools](https://strandsagents.com/latest/user-guide/concepts/tools/tools_overview/)
- [Stability AI Stable Image Generation](https://platform.stability.ai/docs/api-reference#tag/Generate)