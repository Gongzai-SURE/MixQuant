from deepeval.benchmarks import HellaSwag,ARC,Winogrande,BoolQ
from deepeval.benchmarks.tasks import HellaSwagTask
from transformers import AutoModelForCausalLM, AutoTokenizer
from lmformatenforcer import RegexParser
from deepeval.models.base_model import DeepEvalBaseLLM
import torch
from typing import List
from loguru import logger


class DeepEvalBasellm(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)


    def load_model(self):
        return self.model
    
    def generate(self, prompt: str) -> str:
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
       
        generated_ids = self.model.generate(**model_inputs, 
                                            max_new_tokens=5,
                                            do_sample=False
                                            )

        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    # This is optional.
    def batch_generate(self, prompts: List[str]) -> List[str]:

        model_inputs = self.tokenizer(prompts, return_tensors="pt").to(self.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)

    def get_model_name(self):
        return model.config._name_or_path


def reason_test(model, tokenizer,TASK_List: List[str] = ['BoolQ','ARC-E','ARC-C','HellaSwag','WinoGrande']):
    scores = []
    # Load the model and tokenizer
    model = DeepEvalBasellm(model=model, tokenizer=tokenizer)
    logger.info(f"Start test reasoning tasks")
    
    if 'BoolQ' in TASK_List:
        logger.info(f"Start test BoolQ")
        # Define benchmark with specific tasks and shots
        boolq = BoolQ(
            n_shots=0,
            local_dataset = "/root/autodl-tmp/datasets/boolq",
            confinement_instructions = "The final answer is",
        )
        # Evaluate the model
        boolq.evaluate(model=model)
        scores.append(boolq.overall_score)
        logger.info(f"BoolQ test finished. Score: {boolq.overall_score}")

    if 'ARC-E' in TASK_List:
        logger.info(f"Start test ARC-Easy")
        arc = ARC(
            n_shots=0,
            local_dataset = "/root/autodl-tmp/datasets/ARC-E",
            confinement_instructions = "The correct answer is",
        )
        arc.evaluate(model = model)
        scores.append(arc.overall_score)
        logger.info(f"ARC-E test finished. Score: {arc.overall_score}")

    if 'ARC-C' in TASK_List:
        logger.info(f"Start test ARC-Challenge")
        arc = ARC(
            n_shots=0,
            local_dataset = "/root/autodl-tmp/datasets/ARC-C",
            mode = 'hard',
            confinement_instructions = "The correct answer is",
        )
        arc.evaluate(model = model)
        scores.append(arc.overall_score)
        logger.info(f"ARC-C test finished. Score: {arc.overall_score}")

    if 'HellaSwag' in TASK_List:
        logger.info(f"Start test HellaSwag")
        hs = HellaSwag(
            tasks=[HellaSwagTask.HEALTH, HellaSwagTask.CUTTING_THE_GRASS,HellaSwagTask.FINANCE_AND_BUSINESS,HellaSwagTask.RUNNING_A_MARATHON],
            n_shots=0,
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
            n_shots=0,
            local_dataset = "/root/autodl-tmp/datasets/winogrande",
            confinement_instructions = "The correct answer is:",
        )
        # Evaluate the model
        wg.evaluate(model=model)
        scores.append(wg.overall_score)
        logger.info(f"WinoGrande test finished. Score: {wg.overall_score}")
    
    return scores

if __name__ == "__main__":
    model_path = "/root/autodl-tmp/models/llama2-7b"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    scores = reason_test(model, tokenizer)
    print(scores.mean())



