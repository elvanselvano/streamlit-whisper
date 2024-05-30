import os

from haystack import Pipeline
from haystack.components.others import Multiplexer
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator


def profile_extractor(pipeline_kwargs: dict = {}):
    template = ""
    with open("agents/templates/profile_extractor_template.txt") as f:
        template = f.read()
    extractor = Pipeline(**pipeline_kwargs)
    extractor.add_component("prompt", Multiplexer(str))
    extractor.add_component("builder", PromptBuilder(template))
    extractor.add_component(
        "generator",
        OpenAIGenerator(model=os.getenv("LLM_MODEL", "gpt-3.5-turbo")),
    )
    extractor.connect("prompt.value", "builder.story")
    extractor.connect("builder", "generator")
    return extractor
