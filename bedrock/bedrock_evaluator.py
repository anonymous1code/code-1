#!/usr/bin/env python3
"""
Bedrock Claude 3.5 Image Evaluator
Evaluates ASIN images using AWS Bedrock Claude 3.5 Sonnet model
"""

import json
import os
import base64
import boto3
import csv
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union
import time
import io
from PIL import Image
from enum import Enum
from typing import Optional, Tuple

class OutputLabel(Enum):
    """Valid output labels - single source of truth."""
    ACCEPTABLE = ("acceptable", "The generated sofa is consistent.")
    GEOMETRY_ARTIFACTS = ("geometry_artifacts", "Missing seats, wrong shape, incorrect dimensions. Artificats introduced in the side of the sofa that are not present in the original sofa.")
    TEXTURE_LIGHTING_COLOR_ISSUES = ("texture_lighting_color_issues", "Wrong material, color mismatch, lighting problems.")
    WRONG_ORIENTATION = ("wrong_orientation", "Object facing wrong direction.")
    BACKGROUND_ISSUES = ("background_issues", "Non-white background, environmental elements present.")
    IRRELEVANT_OBJECTS = ("irrelevant_objects", "Forbidden objects present (people, pets, multiple sofas, etc.)")
    NON_RELEVANT_MAIN_IMAGE = ("non_relevant_main_image", "The additional source image is non relevant.")

    @property
    def label(self) -> str:
        return self.value[0]
    
    @property
    def description(self) -> str:
        return self.value[1]
    
    @classmethod
    def from_string(cls, s: str) -> Optional["OutputLabel"]:
        """Match string to enum, returns None if not found."""
        s = s.strip().lower().replace("'", "").replace('"', "")
        for item in cls:
            if item.label == s:
                return item
        return None
    
    @classmethod
    def get_prompt_description(cls) -> str:
        """Generate the label list for the prompt."""
        lines = ["**Output Labels List** (use the exact label, and only one):"]
        for item in cls:
            lines.append(f"- '{item.label}': {item.description}")
        return "\n".join(lines)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bedrock_evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class BedrockImageEvaluator:
    def __init__(self, region_name="us-east-1", model_name = None):
        """Initialize the Bedrock client"""

        self.model_id = f"arn:aws:bedrock:us-east-1:211125421161:inference-profile/{model_name}"
        self.model_name = model_name
        # Initialize metadata tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.region_name = region_name
        self._bedrock_runtime = None
        self.PROMPT = (
            "\n**Task**: Determine if the generated object is CONSISTENT with the object in the source image.\n\n"

            "**Context**:\n"
            "- The SOURCE IMAGE shows the object in a lifestyle setting (e.g., home environment, room context, in-use scenario).\n"
            "- The ADDITIONAL SOURCE IMAGE shows the same object in a different setting.\n"
            "- The GENERATED image shows the same object isolated on a white background.\n\n"

            "Consistency is ensured when the material, texture, geometry, number of element of the generated object are respected, i.e., the generated object is exactly the same as the one in the source image. \n\n"

            "**Consistency Criteria** - The following attributes MUST match:\n"
            "1. Core structural elements: Shape, form, and key geometric features\n"
            "2. Material and texture: Fabric type, wood, metal, leather, etc.\n"
            "3. Color: Primary colors and color distribution.\n"
            "4. Orientation: The generated object must be oriented to the left. If the sofa is a corner sofa, the original orientation should be preserved. \n"
            "5. Quantity of essential structural components: Number of seats and cushions (for sofas).\n\n"
            
            "These objects are always allowed to be present in the generated image:\n"
            "- Decorative throw pillows on sofas/couches (unless they appear to be fixed/attached cushions).\n"
            "- Decorative accessories (blankets, throws).\n"
            "- Plaids.\n"
            "- Items inside sofa pockets or compartments (e.g., magazines, papers, electronics).\n"
            "- Items inside cupholders (e.g., glasses, bottles).\n\n"

            "These objects are CONDITIONALLY allowed (ONLY if present in the original image):\n"
            "- Ottomans.\n\n"

            "The following must never appear in the generated image:\n"
            "- People, pets, or body parts.\n"
            "- Multiple main objects (more than one sofa).\n"
            "- Lifestyle or cluttered environments.\n"
            "- Infographics, text, logos, labels, or watermarks.\n"
            "- Electronics (phones, tablets, remotes, etc.), unless placed inside sofa pockets or compartments.\n"
            "- Objects unrelated to the sofa that were not present in the original image.\n\n"

            f"{OutputLabel.get_prompt_description()}"

            "\n\n**Output Format**:\n"
            "```json\n"
            "{\n"
            '  "verdict": "[Concise summary explaining the decision and identifying any specific violations]",\n'
            '  "recommendation":  "[One of the output label]"\n'
            # '  "describe":  "Describe each image in order of appeareance."\n'
            "}\n"
            "```"

        )

    @property
    def bedrock_runtime(self):
        """Only create the client when first accessed"""
        if self._bedrock_runtime is None:
            # Create client NOW, in whatever process is running
            try:
                self._bedrock_runtime = boto3.client(
                    "bedrock-runtime", region_name=self.region_name
                )   
            except Exception as e:
                logger.error(f"Failed to initialize Bedrock client: {e}")
                raise
        return self._bedrock_runtime
    
    def encode_image_to_base64(self, image_input: Union[str, Image.Image]) -> str:
        """Encode image to base64 string"""
        try:
            if isinstance(image_input, str):
                # Handle file path
                with open(image_input, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                return encoded_string
            elif isinstance(image_input, Image.Image):
                # Handle PIL Image
                buffer = io.BytesIO()
                # Convert to RGB if needed (for JPEG compatibility)
                if image_input.mode in ('RGBA', 'LA', 'P'):
                    image_input = image_input.convert('RGB')
                # Save as JPEG to buffer
                image_input.save(buffer, format='JPEG', quality=95)
                buffer.seek(0)
                encoded_string = base64.b64encode(buffer.read()).decode("utf-8")
                return encoded_string
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
        except Exception as e:
            logger.error(f"Failed to encode image {image_input}: {e}")
            raise

    def get_image_media_type(self, image_input: Union[str, Image.Image]) -> str:
        """Get the media type based on file extension or PIL Image format"""
        if isinstance(image_input, str):
            extension = Path(image_input).suffix.lower()
            media_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
                ".webp": "image/webp",
            }
            return media_types.get(extension, "image/jpeg")
        elif isinstance(image_input, Image.Image):
            # For PIL Images, we always convert to JPEG in encode_image_to_base64
            return "image/jpeg"
        else:
            return "image/jpeg"

    def process_single_sample(self, sample_data: Dict[str, Union[str, Image.Image]]) -> Optional[Dict]:
        """Evaluate multiple images using Bedrock Claude 3.5"""

        try:
            # Expected image keys
            expected_keys = ['source_image_main', 'source_image_additional', 'generated_image']
            
            # Prepare content array with images and text
            content = []
            
            # Add each image to the content
            for key in expected_keys:
                if key in sample_data and sample_data[key] is not None:
                    image_base64 = self.encode_image_to_base64(sample_data[key])
                    media_type = self.get_image_media_type(sample_data[key])
                    
                    # Add a text label before each image
                    content.append({
                        "type": "text", 
                        "text": f"[{key.replace('_', ' ').title()}]"
                    })

                    # append the image
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    })
                    
                else:
                    raise ValueError(f"Missing required image: {key}")

            # Add the evaluation prompt
            content.append({"type": "text", "text": self.PROMPT,
                            #  "cache_control": {"type": "ephemeral"}
                        })
            # prompt is too short to do cache write? maybe with images ICL we get to the tokens numbers
            # Prepare the payload
            # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
            }

            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id, body=json.dumps(payload)
            )
            
            # Parse the response
            response_body = json.loads(response["body"].read())
            content = response_body["content"][0]["text"]
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")

            # Try to parse the JSON response
            try:
                evaluation_result = json.loads(content)
                return evaluation_result
            
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {content}")
                return {
                    "error": "JSON parse error",
                    "raw_response": content,
                    "evaluation": {},
                    "recommendation": "ERROR",
                    "final_verdict": f"Failed to parse response: {str(e)}",
                }

        except Exception as e:
           
            logger.error(f"Failed to evaluate sample: {e}")
            return {
                "error": str(e),
                "evaluation": {},
                "recommendation": "ERROR",
                "final_verdict": f"Evaluation failed: {str(e)}",
            }
