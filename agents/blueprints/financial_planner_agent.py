import os

from haystack import Pipeline
from haystack.components.others import Multiplexer
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator


def create_prompt_generator(pipeline: Pipeline, template_path: str, agent_name: str):
    with open(template_path, "r") as f:
        risk_template = f.read()
        pipeline.add_component(
            f"{agent_name}_template_builder", PromptBuilder(risk_template)
        )

    pipeline.add_component(
        agent_name,
        OpenAIGenerator(model=os.getenv("LLM_MODEL", "gpt-3.5-turbo")),
    )
    pipeline.connect(f"{agent_name}_template_builder", agent_name)
    return pipeline


def financial_planner(pipeline_kwargs: dict = {}):
    # code is designed this way due to approaching deadline
    # TODO: refactor codes to make it more reusable
    planner = Pipeline(**pipeline_kwargs)
    planner.add_component("chat", Multiplexer(list[dict[str]]))

    agents = ["risk_analyst", "investment_consultant", "financial_planner"]
    for agent in agents:
        create_prompt_generator(planner, f"agents/templates/{agent}.txt", agent)
        planner.connect("chat.value", f"{agent}_template_builder.chat_log")

    planner.connect("risk_analyst.replies", "financial_planner_template_builder.risks")
    planner.connect(
        "investment_consultant.replies",
        "financial_planner_template_builder.investments",
    )
    return planner
