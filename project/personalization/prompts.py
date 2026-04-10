from __future__ import annotations

from dataclasses import asdict, dataclass


DEFAULT_IDENTITY_TEMPLATES = [
    "a photo of {concept}",
    "a portrait of {concept}",
    "a close-up photo of {concept}",
    "a detailed photo of {concept}",
]

DEFAULT_CLASS_TEMPLATES = [
    "a photo of {class_name}",
    "a portrait of {class_name}",
    "a close-up photo of {class_name}",
]

DEFAULT_KEEP_TEMPLATES = [
    "a natural photo of {concept}",
    "a clean, realistic photo of {concept}",
    "a faithful photo of {concept}",
]

DEFAULT_TARGET_TEMPLATES = [
    "a photo of {concept}",
    "a cinematic photo of {concept}",
    "a detailed portrait of {concept}",
]

DEFAULT_NUISANCE_AXES = [
    {
        "name": "background_indoor_outdoor",
        "positive_templates": [
            "a photo of {concept} indoors",
            "a studio photo of {concept}",
        ],
        "negative_templates": [
            "a photo of {concept} outdoors",
            "a photo of {concept} in nature",
        ],
    },
    {
        "name": "scale_closeup_fullbody",
        "positive_templates": [
            "a close-up photo of {concept}",
            "a tight portrait of {concept}",
        ],
        "negative_templates": [
            "a full-body photo of {concept}",
            "a wide shot of {concept}",
        ],
    },
    {
        "name": "style_realistic_stylized",
        "positive_templates": [
            "a realistic photo of {concept}",
            "a naturalistic photo of {concept}",
        ],
        "negative_templates": [
            "an illustrated picture of {concept}",
            "a stylized rendering of {concept}",
        ],
    },
    {
        "name": "lighting_day_night",
        "positive_templates": [
            "a daylight photo of {concept}",
            "a brightly lit photo of {concept}",
        ],
        "negative_templates": [
            "a night photo of {concept}",
            "a low-light photo of {concept}",
        ],
    },
]


@dataclass
class NuisanceAxis:
    name: str
    positive_prompts: list[str]
    negative_prompts: list[str]

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class PersonalizationPromptBank:
    concept_token: str
    class_name: str
    identity_prompts: list[str]
    class_prompts: list[str]
    keep_prompts: list[str]
    target_prompts: list[str]
    nuisance_axes: list[NuisanceAxis]

    @property
    def instance_prompt(self) -> str:
        return self.identity_prompts[0]

    @property
    def class_prompt(self) -> str:
        return self.class_prompts[0]

    def to_json(self) -> dict:
        return {
            "concept_token": self.concept_token,
            "class_name": self.class_name,
            "identity_prompts": self.identity_prompts,
            "class_prompts": self.class_prompts,
            "keep_prompts": self.keep_prompts,
            "target_prompts": self.target_prompts,
            "nuisance_axes": [axis.to_json() for axis in self.nuisance_axes],
        }


def _render_many(templates: list[str], concept: str, class_name: str) -> list[str]:
    return [template.format(concept=concept, class_name=class_name) for template in templates]


def build_prompt_bank(
    concept_token: str,
    class_name: str,
    identity_templates: list[str] | None = None,
    class_templates: list[str] | None = None,
    keep_templates: list[str] | None = None,
    target_templates: list[str] | None = None,
    nuisance_axes: list[dict] | None = None,
) -> PersonalizationPromptBank:
    identity_templates = identity_templates or DEFAULT_IDENTITY_TEMPLATES
    class_templates = class_templates or DEFAULT_CLASS_TEMPLATES
    keep_templates = keep_templates or DEFAULT_KEEP_TEMPLATES
    target_templates = target_templates or DEFAULT_TARGET_TEMPLATES
    nuisance_axes = nuisance_axes or DEFAULT_NUISANCE_AXES

    concept_phrase = concept_token.strip()
    class_phrase = class_name.strip()

    rendered_axes = []
    for axis in nuisance_axes:
        rendered_axes.append(
            NuisanceAxis(
                name=axis["name"],
                positive_prompts=_render_many(axis["positive_templates"], concept_phrase, class_phrase),
                negative_prompts=_render_many(axis["negative_templates"], concept_phrase, class_phrase),
            )
        )

    return PersonalizationPromptBank(
        concept_token=concept_phrase,
        class_name=class_phrase,
        identity_prompts=_render_many(identity_templates, concept_phrase, class_phrase),
        class_prompts=_render_many(class_templates, concept_phrase, class_phrase),
        keep_prompts=_render_many(keep_templates, concept_phrase, class_phrase),
        target_prompts=_render_many(target_templates, concept_phrase, class_phrase),
        nuisance_axes=rendered_axes,
    )
