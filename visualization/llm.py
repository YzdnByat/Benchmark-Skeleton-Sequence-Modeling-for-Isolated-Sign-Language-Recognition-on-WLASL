import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class ContextualRescorer:
    def __init__(self):
        print("Loading GPT-2 Language Model for Context Rescoring...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()

    def rescore_candidates(self, context_sentence, candidate_predictions):
        """
        Takes a context sentence with '___' placeholder and a list of tuples:
        [('letter', 25.25), ('hat', 3.10), ...]
        Returns sorted rankings based on GPT-2 contextual loss.
        """
        rescored_results = []

        for word, vision_conf in candidate_predictions:
            full_sentence = context_sentence.replace("___", word.lower())
            inputs = self.tokenizer(full_sentence, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()

            rescored_results.append({
                "word": word,
                "vision_conf": vision_conf,
                "context_loss": loss
            })

        # Sort by lowest contextual loss (highest natural sentence probability)
        rescored_results = sorted(rescored_results, key=lambda x: x["context_loss"])
        return rescored_results