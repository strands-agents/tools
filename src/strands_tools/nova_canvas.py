"""
Image generation tool for Strands Agent using Nova Canvas on Amazon Bedrock.

This module provides functionality to generate high-quality images using Amazon Bedrock's
Nova Canvas model based on text prompts. It handles the entire image generation
process including API integration, parameter management, response processing, and
local storage of results.

Key Features:

1. Image Generation:
   • Text-to-image generation using Amazon Nova Canvas
   • Customizable generation parameters (height, width, quality, cfg_scale,
     seed, style, negative_prompt)
   • Support for pre-defined visual styles:
        "3D_ANIMATED_FAMILY_FILM" - A style that alludes to 3D animated
        films. Featuring realistic rendering and characters with cartoonish
        or exaggerated physical features.
        
        "DESIGN_SKETCH" - A style featuring hand-drawn line-art without a
        lot of wash or fill that is not too refined. This style is used to
        convey concepts and ideas.
        
        "FLAT_VECTOR_ILLUSTRATION" - A flat-color illustration style that
        is popular in business communications.
        
        "GRAPHIC_NOVEL_ILLUSTRATION" - A vivid ink illustration style.
        Characters do not have exaggerated features, as with some other more
        cartoon-ish styles.
        
        "MAXIMALISM" - Bright, elaborate, bold, and complex with strong
        shapes, and rich details.
        
        "MIDCENTURY_RETRO" - Alludes to graphic design trends from the
        1940s through 1960s.
        
        "PHOTOREALISM" - Realistic photography style, including different
        repertoires such as stock photography, editorial photography,
        journalistic photography, and more.
        
        "SOFT_DIGITAL_PAINTING" - This style has more finish and refinement
        than a sketch. It includes shading, three dimensionality, and texture
        that might be lacking in other styles.

2. Virtual try-on:
   • Virtual try-on is an image-guided use case of inpainting in which the
     contents of a reference image are superimposed into a source image based
     on the guidance of a mask image.
   • Use case examples for Virtual try-on are:
        1. Adding a logo or text to an image
        2. Use a human and garment image to generate an image with that same
           person wearing it
        3. Place a couch in a living room

3. Background removal:
   • Automatically remove the background of any image, replacing the
     background with transparent pixels.
   • Useful when you want to later composite the image with other elements
     in an image editing app, presentation, or website. 

4. Output Management:
   • Automatic local saving with intelligent filename generation
   • Base64 encoding/decoding for transmission
   • Duplicate filename detection and resolution
   • Organized output directory structure

5. Response Format:
   • Rich response with both text and image data
   • Status tracking and error handling
   • Direct base64 image data for immediate display
   • File path reference for local access

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import nova_canvas
#TODO 
1. add one example for standard generation
2. another for generation with extra parameters
3. virutal try on garmen example
agent = Agent(tools=[generate_image])

# Basic usage with default parameters
agent.tool.generate_image(prompt="A steampunk robot playing chess")

# Advanced usage with Stable Diffusion
agent.tool.generate_image(
    prompt="A futuristic city with flying cars",
    model_id="stability.sd3-5-large-v1:0",
    aspect_ratio="5:4",
    output_format="jpeg",
    negative_prompt="bad lighting, harsh lighting, abstract, surreal, twisted, multiple levels",
)

# Using another Stable Diffusion model
agent.tool.generate_image(
    prompt="A photograph of a cup of coffee from the side",
    model_id="stability.stable-image-ultra-v1:1",
    aspect_ratio="1:1",
    output_format="png",
    negative_prompt="blurry, distorted",
)
```

See the generate_image function docstring for more details on parameters and options.
"""

import base64
import json
import os
import random
import re
import time
from typing import Any

import boto3
from strands.types.tools import ToolResult, ToolUse


TOOL_SPEC = {
    "name": "nova_canvas",
    "description": "Use Amazon Nova Canvas for image generation, virtual try-on, and background removal tasks",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "description": "The task type for Amazon Nova Canvas",
                    "enum": ["TEXT_IMAGE", "VIRTUAL_TRY_ON", "BACKGROUND_REMOVAL"],
                    "default": "TEXT_IMAGE"
                },
                # TEXT_IMAGE parameters
                "text": {
                    "type": "string",
                    "description": "Text prompt for image generation (required for TEXT_IMAGE)"
                },
                "negative_text": {
                    "type": "string",
                    "description": "Negative text prompt (TEXT_IMAGE only)"
                },
                "style": {
                    "type": "string",
                    "description": "Style for image generation (TEXT_IMAGE only)",
                    "enum": ["3D_ANIMATED_FAMILY_FILM", "DESIGN_SKETCH", "FLAT_VECTOR_ILLUSTRATION", "GRAPHIC_NOVEL_ILLUSTRATION", "MAXIMALISM", "MIDCENTURY_RETRO", "PHOTOREALISM", "SOFT_DIGITAL_PAINTING"]
                },
                "width": {
                    "type": "integer",
                    "description": "Image width in pixels (TEXT_IMAGE only)"
                },
                "height": {
                    "type": "integer",
                    "description": "Image height in pixels (TEXT_IMAGE only)"
                },
                # VIRTUAL_TRY_ON parameters
                "image_path": {
                    "type": "string",
                    "description": "Path to source image file to modify (required for VIRTUAL_TRY_ON and BACKGROUND_REMOVAL)"
                },
                "reference_image_path": {
                    "type": "string",
                    "description": "Path to reference image file containing the object to superimpose (required for VIRTUAL_TRY_ON)"
                },
                "mask_type": {
                    "type": "string",
                    "description": "Specifies whether the mask is provided as an image, prompt, or garment mask (required for VIRTUAL_TRY_ON)",
                    "enum": ["IMAGE", "GARMENT", "PROMPT"]
                },
                "mask_image_path": {
                    "type": "string",
                    "description": "Path to mask image file defining areas to edit (black) and ignore (white). Required when mask_type is IMAGE"
                },
                "mask_shape": {
                    "type": "string",
                    "description": "Defines the shape of the mask bounding box, affecting how reference image is transferred",
                    "enum": ["CONTOUR", "BOUNDING_BOX", "DEFAULT"]
                },
                "garment_class": {
                    "type": "string",
                    "description": "Defines the article of clothing being transferred. Required when mask_type is GARMENT",
                    "enum": ["UPPER_BODY", "LOWER_BODY", "FULL_BODY", "FOOTWEAR", "LONG_SLEEVE_SHIRT", "SHORT_SLEEVE_SHIRT", "NO_SLEEVE_SHIRT", "OTHER_UPPER_BODY", "LONG_PANTS", "SHORT_PANTS", "OTHER_LOWER_BODY", "LONG_DRESS", "SHORT_DRESS", "FULL_BODY_OUTFIT", "OTHER_FULL_BODY", "SHOES", "BOOTS", "OTHER_FOOTWEAR"]
                },
                "long_sleeve_style": {
                    "type": "string",
                    "description": "Styling for long sleeve garments (applies only to upper body garments)",
                    "enum": ["SLEEVE_DOWN", "SLEEVE_UP"]
                },
                "tucking_style": {
                    "type": "string",
                    "description": "Tucking style option (applies only to upper body garments)",
                    "enum": ["UNTUCKED", "TUCKED"]
                },
                "outer_layer_style": {
                    "type": "string",
                    "description": "Styling for outer layer garments (applies only to outer layer, upper body garments)",
                    "enum": ["CLOSED", "OPEN"]
                },
                "mask_prompt": {
                    "type": "string",
                    "description": "Natural language text prompt describing regions to edit. Required when mask_type is PROMPT"
                },
                "preserve_body_pose": {
                    "type": "string",
                    "description": "Whether to preserve the body pose in the output image when a person is detected",
                    "enum": ["ON", "OFF", "DEFAULT"]
                },
                "preserve_hands": {
                    "type": "string",
                    "description": "Whether to preserve hands in the output image when a person is detected",
                    "enum": ["ON", "OFF", "DEFAULT"]
                },
                "preserve_face": {
                    "type": "string",
                    "description": "Whether to preserve the face in the output image when a person is detected",
                    "enum": ["OFF", "ON", "DEFAULT"]
                },
                "merge_style": {
                    "type": "string",
                    "description": "Determines how source and reference images are stitched together",
                    "enum": ["BALANCED", "SEAMLESS", "DETAILED"],
                    "default": "BALANCED"
                },
                "return_mask": {
                    "type": "boolean",
                    "description": "Whether to return the mask image with the output image",
                    "default": false
                },
                # BACKGROUND_REMOVAL parameters
                # (uses image_path parameter defined above)
                
                # Common parameters
                "quality": {
                    "type": "string",
                    "description": "Image quality",
                    "enum": ["standard", "premium"],
                    "default": "standard"
                },
                "cfg_scale": {
                    "type": "number",
                    "description": "How strictly to adhere to the prompt. Range: 1.1-10",
                    "minimum": 1.1,
                    "maximum": 10,
                    "default": 6.5
                },
                "seed": {
                    "type": "integer",
                    "description": "Seed for reproducible results"
                },
                "model_id": {
                    "type": "string",
                    "description": "Model ID",
                    "default": "amazon.nova-canvas-v1:0"
                },
                "region": {
                    "type": "string",
                    "description": "AWS region",
                    "default": "us-east-1"
                }
            },
            "required": []
        }
    },
}


def create_filename(prompt: str) -> str:
    """Generate a filename from the prompt text."""
    words = re.findall(r"\w+", prompt.lower())[:5]
    filename = "_".join(words)
    filename = re.sub(r"[^\w\-_\.]", "_", filename)
    return filename[:100]  # Limit filename length


def nova_canvas(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """
    Use Amazon Nova Canvas for image generation, virtual try-on, and background removal.
    
    This function supports three main task types:
    
    1. TEXT_IMAGE - Generate images from text prompts with optional style parameters
    2. VIRTUAL_TRY_ON - Superimpose objects from a reference image onto a source image
    3. BACKGROUND_REMOVAL - Remove the background from an image
    """
    try:
        tool_use_id = tool["toolUseId"]
        tool_input = tool["input"]
        
        task_type = tool_input.get("task_type", "TEXT_IMAGE")
        model_id = tool_input.get("model_id", "amazon.nova-canvas-v1:0")
        region = tool_input.get("region", "us-east-1")
        
        client = boto3.client("bedrock-runtime", region_name=region)
        
        # Build request based on task type
        if task_type == "TEXT_IMAGE":
            request_body = {
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": tool_input.get("text", "A beautiful landscape")
                },
                "imageGenerationConfig": {
                    "quality": tool_input.get("quality", "standard")
                }
            }
            
            # Add optional TEXT_IMAGE parameters
            if "negative_text" in tool_input:
                request_body["textToImageParams"]["negativeText"] = tool_input["negative_text"]
            if "style" in tool_input:
                request_body["textToImageParams"]["style"] = tool_input["style"]
            if "width" in tool_input:
                request_body["imageGenerationConfig"]["width"] = tool_input["width"]
            if "height" in tool_input:
                request_body["imageGenerationConfig"]["height"] = tool_input["height"]
            if "cfg_scale" in tool_input:
                request_body["imageGenerationConfig"]["cfgScale"] = tool_input["cfg_scale"]
            if "seed" in tool_input:
                request_body["imageGenerationConfig"]["seed"] = tool_input["seed"]
                
        elif task_type == "VIRTUAL_TRY_ON":
            # Validate required parameters
            if "source_image" not in tool_input:
                raise ValueError("source_image is required for VIRTUAL_TRY_ON")
            if "reference_image" not in tool_input:
                raise ValueError("reference_image is required for VIRTUAL_TRY_ON")
            if "mask_type" not in tool_input:
                raise ValueError("mask_type is required for VIRTUAL_TRY_ON")
                
            # Initialize request structure
            request_body = {
                "taskType": "VIRTUAL_TRY_ON",
                "virtualTryOnParams": {
                    "sourceImage": tool_input["source_image"],
                    "referenceImage": tool_input["reference_image"],
                    "maskType": tool_input["mask_type"]
                },
                "imageGenerationConfig": {
                    "quality": tool_input.get("quality", "standard")
                }
            }
            
            # Handle mask type specific parameters
            mask_type = tool_input["mask_type"]
            
            if mask_type == "IMAGE":
                if "mask_image" not in tool_input:
                    raise ValueError("mask_image is required when mask_type is IMAGE")
                request_body["virtualTryOnParams"]["imageBasedMask"] = {
                    "maskImage": tool_input["mask_image"]
                }
                
            elif mask_type == "GARMENT":
                if "garment_class" not in tool_input:
                    raise ValueError("garment_class is required when mask_type is GARMENT")
                    
                garment_mask = {
                    "garmentClass": tool_input["garment_class"]
                }
                
                if "mask_shape" in tool_input:
                    garment_mask["maskShape"] = tool_input["mask_shape"]
                
                # Add garment styling if any styling options are provided
                styling_params = ["long_sleeve_style", "tucking_style", "outer_layer_style"]
                if any(param in tool_input for param in styling_params):
                    garment_mask["garmentStyling"] = {}
                    
                    if "long_sleeve_style" in tool_input:
                        garment_mask["garmentStyling"]["longSleeveStyle"] = tool_input["long_sleeve_style"]
                    if "tucking_style" in tool_input:
                        garment_mask["garmentStyling"]["tuckingStyle"] = tool_input["tucking_style"]
                    if "outer_layer_style" in tool_input:
                        garment_mask["garmentStyling"]["outerLayerStyle"] = tool_input["outer_layer_style"]
                
                request_body["virtualTryOnParams"]["garmentBasedMask"] = garment_mask
                
            elif mask_type == "PROMPT":
                if "mask_prompt" not in tool_input:
                    raise ValueError("mask_prompt is required when mask_type is PROMPT")
                    
                prompt_mask = {
                    "maskPrompt": tool_input["mask_prompt"]
                }
                
                if "mask_shape" in tool_input:
                    prompt_mask["maskShape"] = tool_input["mask_shape"]
                    
                request_body["virtualTryOnParams"]["promptBasedMask"] = prompt_mask
            
            # Add mask exclusions if any are provided
            exclusion_params = ["preserve_body_pose", "preserve_hands", "preserve_face"]
            if any(param in tool_input for param in exclusion_params):
                request_body["virtualTryOnParams"]["maskExclusions"] = {}
                
                if "preserve_body_pose" in tool_input:
                    request_body["virtualTryOnParams"]["maskExclusions"]["preserveBodyPose"] = tool_input["preserve_body_pose"]
                if "preserve_hands" in tool_input:
                    request_body["virtualTryOnParams"]["maskExclusions"]["preserveHands"] = tool_input["preserve_hands"]
                if "preserve_face" in tool_input:
                    request_body["virtualTryOnParams"]["maskExclusions"]["preserveFace"] = tool_input["preserve_face"]
            
            # Add merge style and return mask options
            if "merge_style" in tool_input:
                request_body["virtualTryOnParams"]["mergeStyle"] = tool_input["merge_style"]
            if "return_mask" in tool_input:
                request_body["virtualTryOnParams"]["returnMask"] = tool_input["return_mask"]
                
            # Add common generation config parameters
            if "cfg_scale" in tool_input:
                request_body["imageGenerationConfig"]["cfgScale"] = tool_input["cfg_scale"]
            if "seed" in tool_input:
                request_body["imageGenerationConfig"]["seed"] = tool_input["seed"]
                
        elif task_type == "BACKGROUND_REMOVAL":
            if "image" not in tool_input:
                raise ValueError("image is required for BACKGROUND_REMOVAL")
                
            request_body = {
                "taskType": "BACKGROUND_REMOVAL",
                "backgroundRemovalParams": {
                    "image": tool_input["image"]
                }
            }
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Invoke the model
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        # Process response
        model_response = json.loads(response["body"].read().decode("utf-8"))
        
        # Extract image data
        if "images" in model_response and len(model_response["images"]) > 0:
            base64_image_data = model_response["images"][0]
            
            # Create filename based on task type
            if task_type == "TEXT_IMAGE":
                filename = create_filename(tool_input.get("text", "generated_image"))
            elif task_type == "VIRTUAL_TRY_ON":
                filename = f"virtual_try_on_{int(time.time())}"
            else:  # BACKGROUND_REMOVAL
                filename = f"background_removal_{int(time.time())}"
            
            # Save image
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            i = 1
            base_image_path = os.path.join(output_dir, f"{filename}.png")
            image_path = base_image_path
            while os.path.exists(image_path):
                image_path = os.path.join(output_dir, f"{filename}_{i}.png")
                i += 1
            
            with open(image_path, "wb") as file:
                file.write(base64.b64decode(base64_image_data))
            
            # Handle mask if returned
            mask_message = ""
            if task_type == "VIRTUAL_TRY_ON" and tool_input.get("return_mask") and "maskImage" in model_response:
                mask_data = model_response["maskImage"]
                mask_path = os.path.join(output_dir, f"{filename}_mask.png")
                with open(mask_path, "wb") as file:
                    file.write(base64.b64decode(mask_data))
                mask_message = f" Mask saved to {mask_path}."
            
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [
                    {"text": f"{task_type} task completed successfully. Image saved to {image_path}.{mask_message}"},
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": base64.b64decode(base64_image_data)},
                        }
                    },
                ],
            }
        else:
            raise ValueError("No image data found in the response")
            
    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error: {str(e)}"}],
        }style"]
            if "width" in tool_input:
                request_body["imageGenerationConfig"]["width"] = tool_input["width"]
            if "height" in tool_input:
                request_body["imageGenerationConfig"]["height"] = tool_input["height"]
            if "cfg_scale" in tool_input:
                request_body["imageGenerationConfig"]["cfgScale"] = tool_input["cfg_scale"]
            if "seed" in tool_input:
                request_body["imageGenerationConfig"]["seed"] = tool_input["seed"]
                
        elif task_type == "VIRTUAL_TRY_ON":
            request_body = {
                "taskType": "VIRTUAL_TRY_ON",
                "virtualTryOnParams": {
                    "sourceImage": tool_input["source_image"],
                    "referenceImage": tool_input["reference_image"],
                    "maskType": tool_input.get("mask_type", "GARMENT")
                },
                "imageGenerationConfig": {
                    "quality": tool_input.get("quality", "standard")
                }
            }
            
            # Add mask-specific parameters
            if tool_input.get("mask_type") == "IMAGE" and "mask_image" in tool_input:
                request_body["virtualTryOnParams"]["imageBasedMask"] = {
                    "maskImage": tool_input["mask_image"]
                }
            elif tool_input.get("mask_type") == "GARMENT":
                garment_mask = {}
                if "mask_shape" in tool_input:
                    garment_mask["maskShape"] = tool_input["mask_shape"]
                if "garment_class" in tool_input:
                    garment_mask["garmentClass"] = tool_input["garment_class"]
                if any(k in tool_input for k in ["long_sleeve_style", "tucking_style", "outer_layer_style"]):
                    garment_mask["garmentStyling"] = {}
                    if "long_sleeve_style" in tool_input:
                        garment_mask["garmentStyling"]["longSleeveStyle"] = tool_input["long_sleeve_style"]
                    if "tucking_style" in tool_input:
                        garment_mask["garmentStyling"]["tuckingStyle"] = tool_input["tucking_style"]
                    if "outer_layer_style" in tool_input:
                        garment_mask["garmentStyling"]["outerLayerStyle"] = tool_input["outer_layer_style"]
                request_body["virtualTryOnParams"]["garmentBasedMask"] = garment_mask
            elif tool_input.get("mask_type") == "PROMPT" and "mask_prompt" in tool_input:
                prompt_mask = {"maskPrompt": tool_input["mask_prompt"]}
                if "mask_shape" in tool_input:
                    prompt_mask["maskShape"] = tool_input["mask_shape"]
                request_body["virtualTryOnParams"]["promptBasedMask"] = prompt_mask
            
            # Add mask exclusions
            if any(k in tool_input for k in ["preserve_body_pose", "preserve_hands", "preserve_face"]):
                request_body["virtualTryOnParams"]["maskExclusions"] = {}
                if "preserve_body_pose" in tool_input:
                    request_body["virtualTryOnParams"]["maskExclusions"]["preserveBodyPose"] = tool_input["preserve_body_pose"]
                if "preserve_hands" in tool_input:
                    request_body["virtualTryOnParams"]["maskExclusions"]["preserveHands"] = tool_input["preserve_hands"]
                if "preserve_face" in tool_input:
                    request_body["virtualTryOnParams"]["maskExclusions"]["preserveFace"] = tool_input["preserve_face"]
            
            # Add other virtual try-on parameters
            if "merge_style" in tool_input:
                request_body["virtualTryOnParams"]["mergeStyle"] = tool_input["merge_style"]
            if "return_mask" in tool_input:
                request_body["virtualTryOnParams"]["returnMask"] = tool_input["return_mask"]
            if "cfg_scale" in tool_input:
                request_body["imageGenerationConfig"]["cfgScale"] = tool_input["cfg_scale"]
            if "seed" in tool_input:
                request_body["imageGenerationConfig"]["seed"] = tool_input["seed"]
                
        elif task_type == "BACKGROUND_REMOVAL":
            request_body = {
                "taskType": "BACKGROUND_REMOVAL",
                "backgroundRemovalParams": {
                    "image": tool_input["image"]
                }
            }
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Invoke the model
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        # Process response
        model_response = json.loads(response["body"].read().decode("utf-8"))
        base64_image_data = model_response["images"][0]
        
        # Save image
        filename = create_filename(tool_input.get("text", task_type.lower()))
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        i = 1
        base_image_path = os.path.join(output_dir, f"{filename}.png")
        image_path = base_image_path
        while os.path.exists(image_path):
            image_path = os.path.join(output_dir, f"{filename}_{i}.png")
            i += 1
        
        with open(image_path, "wb") as file:
            file.write(base64.b64decode(base64_image_data))
        
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [
                {"text": f"Task {task_type} completed. Image saved to {image_path}"},
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": base64.b64decode(base64_image_data)},
                    }
                },
            ],
        }
        
    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error: {str(e)}"}],
        }
