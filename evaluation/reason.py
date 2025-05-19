from deepeval.benchmarks import HellaSwag,ARC,Winogrande,BoolQ
from deepeval.benchmarks.tasks import HellaSwagTask
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
import torch
from typing import List
from loguru import logger
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
from mixq.utils.modelutils import move


class DeepEvalBasellm(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.tokenizer = tokenizer
        self.model, self.device = move(model)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model_inputs = self.tokenizer([prompt], return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=3,
            do_sample=False
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=5,
            do_sample=True
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def get_model_name(self):
        return self.model.config._name_or_path


def reason_test(model, tokenizer,TASK_List: List[str] = ['BoolQ','ARC-E','ARC-C','HellaSwag','WinoGrande'],n_shots: int = 0) -> List[float]:
    scores = []
    # Load the model and tokenizer
    model = DeepEvalBasellm(model=model, tokenizer=tokenizer)
    logger.info(f"Start test reasoning tasks")
    
    if 'BoolQ' in TASK_List:
        logger.info(f"Start test BoolQ")
        # Define benchmark with specific tasks and shots
        boolq = BoolQ(
            n_shots=n_shots,
            local_dataset = "/root/autodl-tmp/datasets/boolq",

            # enforced model generation
            # instructions= "Please answer the question with 'Yes' or 'No' based on passage.",
            # instructions= "",
            # confinement_instructions = "The final answer is",
        )
        # Evaluate the model
        boolq.evaluate(model=model)
        scores.append(boolq.overall_score)
        logger.info(f"BoolQ test finished. Score: {boolq.overall_score}")

    if 'ARC-E' in TASK_List:
        logger.info(f"Start test ARC-Easy")
        arc = ARC(
            n_shots=n_shots,
            local_dataset = "/root/autodl-tmp/datasets/ARC-E",
            confinement_instructions = "The correct option is:",
        )
        arc.evaluate(model = model)
        scores.append(arc.overall_score)
        logger.info(f"ARC-E test finished. Score: {arc.overall_score}")

    if 'ARC-C' in TASK_List:
        logger.info(f"Start test ARC-Challenge")
        arc = ARC(
            n_shots=n_shots,
            local_dataset = "/root/autodl-tmp/datasets/ARC-C",
            mode = 'hard',
            confinement_instructions = "The correct option is",
        )
        arc.evaluate(model = model)
        scores.append(arc.overall_score)
        logger.info(f"ARC-C test finished. Score: {arc.overall_score}")

    if 'HellaSwag' in TASK_List:
        logger.info(f"Start test HellaSwag")
        hs = HellaSwag(
            tasks=[HellaSwagTask.HEALTH, HellaSwagTask.CUTTING_THE_GRASS,HellaSwagTask.FINANCE_AND_BUSINESS,HellaSwagTask.RUNNING_A_MARATHON],
            n_shots=n_shots,
            local_dataset = "/root/autodl-tmp/datasets/hellaswag",
            confinement_instructions = "The correct answer is",
        )
        # Evaluate the model
        hs.evaluate(model=model)
        scores.append(hs.overall_score)
        logger.info(f"HellaSwag test finished. Score: {hs.overall_score}")

    if 'WinoGrande' in TASK_List:
        logger.info(f"Start test WinoGrande")
        # Define benchmark with specific tasks and shots
        wg = Winogrande(
            n_shots=n_shots,
            local_dataset = "/root/autodl-tmp/datasets/winogrande",
            confinement_instructions = "The correct answer is:",
        )
        # Evaluate the model
        wg.evaluate(model=model)
        scores.append(wg.overall_score)
        logger.info(f"WinoGrande test finished. Score: {wg.overall_score}")
    
    return scores

if __name__ == "__main__":
    model_path = "/root/autodl-tmp/models/llama2-13b-chat"
    n_shots = 0
    TASK_List = ['ARC-E','ARC-C','HellaSwag','WinoGrande']
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    scores = reason_test(model, tokenizer,TASK_List=TASK_List,n_shots = n_shots)
    # 记录平均值
    print(f"average score: {np.mean(scores)}")



