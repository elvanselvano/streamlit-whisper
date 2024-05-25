import os

from haystack import Pipeline
from haystack.components.others import Multiplexer
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator


def create_prompt_generator(pipeline, template_path, agent_name):
    with open(template_path, "r") as f:
        risk_template = f.read()
        pipeline.add_component(
            f"{agent_name}_template_builder", PromptBuilder(risk_template)
        )

    pipeline.add_component(
        agent_name,
        OllamaGenerator(
            model=os.getenv("LLM_MODEL", "gemma"),
            url=str(os.environ["LLM_URL"]),
            timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", 10 * 60)),
        ),
    )
    pipeline.connect(f"{agent_name}_template_builder", agent_name)
    return pipeline


def financial_planner(pipeline_kwargs={}):
    # code is designed this way due to approaching deadline
    # TODO: refactor codes to make it more reusable
    planner = Pipeline(**pipeline_kwargs)
    planner.add_component("profile", Multiplexer(str))
    planner.add_component("chat", Multiplexer(list[dict[str]]))

    agents = ["risk_analyst", "investment_consultant", "financial_planner"]
    for agent in agents:
        create_prompt_generator(planner, f"agents/templates/{agent}.txt", agent)
        planner.connect("chat.value", f"{agent}_template_builder.chat_log")
        planner.connect("profile.value", f"{agent}_template_builder.profile")

    planner.connect("risk_analyst.replies", "financial_planner_template_builder.risks")
    planner.connect(
        "investment_consultant.replies",
        "financial_planner_template_builder.investments",
    )
    return planner
