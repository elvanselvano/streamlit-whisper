import os

from haystack import Pipeline
from haystack.components.others import Multiplexer
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator


def profile_extractor(pipeline_kwargs={}):
    template = ""
    with open("agents/templates/profile_extractor_template.txt") as f:
        template = f.read()
    extractor = Pipeline(**pipeline_kwargs)
    extractor.add_component("prompt", Multiplexer(str))
    extractor.add_component("builder", PromptBuilder(template))
    extractor.add_component(
        "generator",
        OllamaGenerator(
            model=os.environ["LLM_MODEL"],
            url=str(os.environ["LLM_URL"]),
            timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", 10 * 60)),
        ),
    )
    extractor.connect("prompt.value", "builder.story")
    extractor.connect("builder", "generator")
    return extractor
