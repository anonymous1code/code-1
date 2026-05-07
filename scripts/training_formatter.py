"""
Training data formatter for Qwen3 AutoQA fine-tuning.

Converts sampled image pairs into Qwen3-VL chat-format training examples
using the same evaluation prompt used for zero-shot benchmarking, with
ground-truth chain-of-thought reasoning and Yes/No classification.

Usage (as a library):
    from scripts.training_formatter import format_training_example, format_all_examples
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Evaluation prompt — identical to the one used for zero-shot Qwen3 eval
# ---------------------------------------------------------------------------

EVAL_PROMPT = (
    "<prompt_text>\n"
    "<Task>Classify the consistency between the generated object and the "
    "source images as 'Yes' or 'No'.</Task>\n"
    "<Ground Truth>Ground-truth labels are absolutely correct. 'Yes' means "
    "consistent, 'No' means inconsistent.\n</Ground Truth>\n"
    "<Source Images>\n"
    "<source_image>Source image: Lifestyle setting with object.</source_image>\n"
    "<source_image>Generated image: Object isolated on white background.</source_image>\n"
    "</Source Images>\n"
    "<Consistency Rules>\n"
    "Rule 1: Core Structure. Match shape, form, and key geometric features "
    "exactly. If any geometry artifacts exist in the generated image, "
    "classify as 'No'.\n"
    "Rule 2: Material & Texture. Match fabric, wood, metal, and finish "
    "types exactly.\n"
    "Rule 3: Color. Match primary colors and distribution exactly.\n"
    "Rule 4: Orientation. Match orientation (e.g., left/right, corner). "
    "If corner orientation is changed, classify as 'No'.\n"
    "Rule 5: Quantity. Match number of seats and cushions exactly.\n"
    "Rule 6: Allowed Items. Decorative pillows, blankets, throws, and "
    "items inside pockets/compartments are allowed.\n"
    "Rule 7: Conditional Items. Ottomans are allowed ONLY if present in "
    "the original source image.\n"
    "Rule 8: Prohibited Items. People, pets, body parts, multiple main "
    "objects, lifestyle backgrounds, text, logs, watermarks, unrelated "
    "electronics (outside pockets) are forbidden.\n"
    "Rule 9 (Priority): If the input metadata 'rejection_reason' contains "
    "'Geometry Artifacts', 'Texture Mismatch', or 'Color Mismatch', the "
    "classification MUST be 'No', overriding visual evidence.\n"
    "</Consistency Rules>\n"
    "<Metadata Check>\n"
    "If the 'rejection_reason' metadata is 'accepted', verify against "
    "visual evidence in Rules 1-8. If visual evidence contradicts "
    "acceptance (e.g., artifacts exist), classify as 'No'.\n"
    "</Metadata Check>\n"
    "<Output Format>\n"
    "Output must be one of: '<reasoning> ... </reasoning>' followed by "
    "'Yes' or 'No'.\n"
    "Reasoning must explain why negative criteria were satisfied or "
    "violated. If 'No', explicitly mention the violated rule or rejection "
    "reason.\n"
    "</Output Format>\n"
    "</prompt_text>"
)

# Keep backward-compatible alias
CONSISTENCY_PROMPT = EVAL_PROMPT

# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------
_ACCEPT_LABELS = {"accepted", "accept"}
_REJECT_LABELS = {"rejected", "reject"}


def _label_to_answer(label: str) -> str:
    """Map a dataset label to the ground-truth Yes/No answer.

    Raises ``ValueError`` for unrecognised labels.
    """
    normed = label.strip().lower()
    if normed in _ACCEPT_LABELS:
        return "Yes"
    if normed in _REJECT_LABELS:
        return "No"
    raise ValueError(
        f"Unrecognised label {label!r}. Expected one of: "
        f"{sorted(_ACCEPT_LABELS | _REJECT_LABELS)}"
    )


# ---------------------------------------------------------------------------
# Rejection-reason → violated-rule mapping
# ---------------------------------------------------------------------------

_REASON_TO_RULES: dict[str, str] = {
    # vcape-r reasons
    "Geometry Artifacts": (
        "Rule 1 (Core Structure) is violated — the generated image exhibits "
        "geometry artifacts; shape, form, or key geometric features differ "
        "from the source product. Rule 9 also applies as the rejection_reason "
        "contains 'Geometry Artifacts'."
    ),
    "Texture/Lighting/Color Issues": (
        "Rule 2 (Material & Texture) and Rule 3 (Color) are violated — "
        "the generated image shows texture, lighting, or color issues that "
        "do not match the source product."
    ),
    "Wrong Orientation": (
        "Rule 4 (Orientation) is violated — the generated image shows the "
        "product in the wrong orientation compared to the source."
    ),
    "Irrelevant Objects": (
        "Rule 8 (Prohibited Items) is violated — the generated image "
        "contains irrelevant objects that are forbidden."
    ),
    "Generated-Main Image Mismatch (Color/Material/Pattern)": (
        "Rule 2 (Material & Texture) and Rule 3 (Color) are violated — "
        "the generated image does not match the source in color, material, "
        "or pattern."
    ),
    "Background Issues": (
        "Rule 8 (Prohibited Items) is violated — the generated image has "
        "background issues containing prohibited elements such as lifestyle "
        "scenes or distracting content."
    ),
    # vcape-s reasons
    "wrong_orientation": (
        "Rule 4 (Orientation) is violated — the generated image shows the "
        "product in the wrong orientation."
    ),
    "wrong_product_same_pt": (
        "Rule 1 (Core Structure) is violated — the generated image shows "
        "a different product of the same product type; the specific product "
        "identity does not match the source."
    ),
    "wrong_product_different_pt": (
        "Rule 1 (Core Structure) is violated — the generated image shows "
        "a completely different product of a different type."
    ),
}


# ---------------------------------------------------------------------------
# Reasoning generation helpers
# ---------------------------------------------------------------------------

def _generate_accepted_reasoning(row: dict) -> str:
    """Generate a reasoning block for an accepted pair."""
    desc = row.get("object_description", "a product")
    return (
        f"The source image shows {desc}. The generated image shows the same "
        f"product isolated on a white background. Checking all rules:\n"
        f"- Rule 1 (Core Structure): Shape and geometric features match.\n"
        f"- Rule 2 (Material & Texture): Materials appear consistent.\n"
        f"- Rule 3 (Color): Colors match the source.\n"
        f"- Rule 4 (Orientation): Orientation is correct.\n"
        f"- Rule 5 (Quantity): Component counts match.\n"
        f"- Rule 6 (Allowed Items): No issues.\n"
        f"- Rule 7 (Conditional Items): No issues.\n"
        f"- Rule 8 (Prohibited Items): No prohibited elements present.\n"
        f"- Rule 9 (Priority): rejection_reason is 'accepted', no override.\n"
        f"- Metadata Check: rejection_reason is 'accepted' and visual "
        f"evidence confirms consistency.\n"
        f"All rules satisfied."
    )


def _generate_rejected_reasoning(row: dict) -> str:
    """Generate a reasoning block for a rejected pair."""
    desc = row.get("object_description", "a product")
    reason = row.get("rejection_reason", "")

    rule_explanation = _REASON_TO_RULES.get(
        reason,
        f"The generated image was rejected for: {reason}.",
    )

    return (
        f"The source image shows {desc}. The generated image is supposed "
        f"to show the same product isolated on a white background, but "
        f"there are issues.\n"
        f"Checking rules against rejection_reason='{reason}':\n"
        f"- {rule_explanation}\n"
        f"The classification must be 'No' due to the violated rule(s)."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_training_example(row: dict, split: str) -> dict:
    """Convert a sampled row into a chat-format training example.

    Parameters
    ----------
    row : dict
        A single row from the sampled dataset.  Expected keys include
        ``xsource_image``, ``xtarget_image``, ``label``,
        ``rejection_reason``, ``object_description``, and ``product_type``.
    split : str
        The source dataset split (``"vcape-r-20k"`` or ``"vcape-s-20k"``).

    Returns
    -------
    dict
        A dict with a ``messages`` key containing the chat turns, compatible
        with TRL SFTTrainer.

    Raises
    ------
    ValueError
        If ``row["label"]`` is not a recognised accept/reject value.
    """
    answer = _label_to_answer(row["label"])

    if answer == "Yes":
        reasoning = _generate_accepted_reasoning(row)
    else:
        reasoning = _generate_rejected_reasoning(row)

    assistant_text = f"<reasoning>\n{reasoning}\n</reasoning>\n\n{answer}"

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": row["xsource_image"]},
                    {"type": "image", "image": row["xtarget_image"]},
                    {"type": "text", "text": EVAL_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text},
                ],
            },
        ],
    }


def format_all_examples(dataset) -> list[dict]:
    """Format every row in *dataset* into chat-format training examples.

    Parameters
    ----------
    dataset
        An iterable of rows (dicts).  Each row must contain a ``split``
        key in addition to the columns expected by
        :func:`format_training_example`.

    Returns
    -------
    list[dict]
        A list of formatted training examples, one per row.
    """
    results: list[dict] = []
    for i in range(len(dataset)):
        row = dataset[i]
        split = row["split"]
        results.append(format_training_example(row, split))
    return results
