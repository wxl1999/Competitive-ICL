import os
import sys
import copy
import random
import string
import pdb

from typing import List

ICL_EXPLORE_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ICL_EXPLORE_ROOT_PATH)

from src.format.input_example import InputExample
from src.format.input_format import INPUT_FORMATS
from src.format.output_format import OUTPUT_FORMATS
from src.utils.utils import read_jsonl

DATASETS = {}

def register_dataset(dataset_class):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(
    dataset_name: str,
    seed: int,
    few_shot_numbers: int,
    prompt_format: str,
    evaluation_type: str,
    sep_symbol: str,
    max_input_length: int = 1024,
    max_output_length: int = 256,
):
    file_name = dataset_name
    return DATASETS[dataset_name](
        seed=seed,
        file_name=file_name,
        few_shot_numbers=few_shot_numbers,
        prompt_format=prompt_format,
        evaluation_type=evaluation_type,
        sep_symbol=sep_symbol,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
    )


class BaseDataset:

    def __init__(
        self,
        seed: int,
        file_name: str,
        few_shot_numbers: int,
        prompt_format: str,
        evaluation_type: str,
        sep_symbol: str,
        max_input_length: int = 1024,
        max_output_length: int = 256,
    ) -> None:
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.seed = seed
        self.file_name = file_name
        self.few_shot_numbers = few_shot_numbers
        self.prompt_format = prompt_format
        self.evaluation_type = evaluation_type
        self.sep_symbol = sep_symbol
        self.input_format = "plain"
        self.default_input_format = "plain"
        self.default_output_format = None

    def load_train_examples(self):
        train_examples_path = f"data/data-{self.few_shot_numbers}/{self.file_name}/{self.file_name}_{self.few_shot_numbers}_{self.seed}_train.jsonl"
        examples_list = read_jsonl(train_examples_path)
        examples_list = [
            InputExample(
                task=e["task"],
                input=e["input"],
                gt_label_name=e["output"],
                options=e["options"],
            )
            for e in examples_list
        ]
        return examples_list
    
    def load_dev_examples(self):
        dev_samples_path = f"data/data-{self.few_shot_numbers}/{self.file_name}/{self.file_name}_{self.few_shot_numbers}_{self.seed}_dev.jsonl"
        dev_samples_list = read_jsonl(dev_samples_path)
        dev_samples_list = [
            InputExample(
                task=e["task"],
                input=e["input"],
                gt_label_name=e["output"],
                options=e["options"],
            )
            for e in dev_samples_list
        ]
        return dev_samples_list
        

    def load_test_examples(self):
        test_samples_path = f"data/data-{self.few_shot_numbers}/{self.file_name}/{self.file_name}_{self.few_shot_numbers}_{self.seed}_test.jsonl"
        test_samples_list = read_jsonl(test_samples_path)
        test_samples_list = [
            InputExample(
                task=e["task"],
                input=e["input"],
                gt_label_name=e["output"],
                options=e["options"],
            )
            for e in test_samples_list
        ]
        return test_samples_list

    def parse_answer(self, response):
        self.input_format_class = INPUT_FORMATS[
            (
                self.input_format
                if self.input_format is not None
                else self.default_input_format
            )
        ]()
        parsed_answer = self.input_format_class.parse_answer(response)
        return parsed_answer

    def format_input_examples(self, example: InputExample):
        self.input_format_class = INPUT_FORMATS[
            (
                self.input_format
                if self.input_format is not None
                else self.default_input_format
            )
        ]()
        input_str = self.input_format_class.format_input(example)
        return input_str

    def format_output_examples(self, example: InputExample):
        self.output_format_class = OUTPUT_FORMATS[
            (
                self.output_format
                if self.output_format is not None
                else self.default_output_format
            )
        ]()
        output_str = self.output_format_class.format_output(
            example, self.evaluation_type
        )
        return output_str

    def map_example_labels(self, examples: List[InputExample], random_seed):

        random.seed(random_seed)

        mapping_examples = copy.deepcopy(examples)
        options = mapping_examples[0].options
        n_labels = len(options)
        mapping_dict = {}

        if self.evaluation_type == "abstract_numbers":
            span_labels = [str(i) for i in range(n_labels)]

        elif self.evaluation_type == "abstract_letters":
            span_labels = random.sample(string.ascii_uppercase, n_labels)

        else:
            assert self.evaluation_type == "abstract_symbols"
            span_labels = list("@#$%*^{}")

        random.shuffle(span_labels)
        for option, span_label in zip(options, span_labels):
            mapping_dict[option] = span_label

        for idx, mapping_example in enumerate(mapping_examples):
            original_label = mapping_example.gt_label_name
            new_label = mapping_dict[original_label]
            mapping_example.map_gt_label_name = new_label

        return mapping_dict, mapping_examples

    def randomize_example(self, examples: List[InputExample], random_seed):

        random.seed(random_seed)

        random_examples = copy.deepcopy(examples)
        options = random_examples[0].options

        for idx, random_example in enumerate(random_examples):
            new_label = random.choice(options)
            random_example.random_gt_label_name = new_label

        return random_examples

    def get_icl_examples(self, examples: List[InputExample]):
        examples_str_list = []
        if self.prompt_format == "natural_language":
            for example in examples:
                input_str = self.format_input_examples(example)
                output_str = self.format_output_examples(example)
                examples_str_list.append(input_str + output_str)
        else:
            assert self.prompt_format == "minimal"
            for example in examples:
                input_str = example.input
                output_str = self.format_output_examples(example)
                examples_str_list.append(input_str + "\n" + output_str)

        examples_str = self.sep_symbol.join(examples_str_list)
        return examples_str
    
    def get_options(self, examples: List[InputExample]):
        options_list = []
        for example in examples:
            options = example.options
            options_list.append(options)
        return options_list


class ClassificationDataset(BaseDataset):

    def __init__(
        self,
        seed: int,
        file_name: str,
        few_shot_numbers: int,
        prompt_format: str,
        evaluation_type: str,
        sep_symbol: str,
        max_input_length: int,
        max_output_length: int,
    ) -> None:
        super().__init__(
            seed,
            file_name,
            few_shot_numbers,
            prompt_format,
            evaluation_type,
            sep_symbol,
            max_input_length,
            max_output_length,
        )
        self.output_format = "classification"
        self.is_classification_dataset = True
        self.to_tf = {
            "entailment": "True",
            "not_entailment": "False",
            "equivalent": "True",
            "not_equivalent": "False",
            "contradiction": "False",
            "neutral": "Unknown",
            "Unknown": "Unknown",
            "Yes": "True",
            "No": "False",
        }


class NLIDataset(ClassificationDataset):

    def __init__(
        self,
        seed: int,
        file_name: str,
        few_shot_numbers: int,
        prompt_format: str,
        evaluation_type: str,
        sep_symbol: str,
        max_input_length: int,
        max_output_length: int,
    ) -> None:
        super().__init__(
            seed,
            file_name,
            few_shot_numbers,
            prompt_format,
            evaluation_type,
            sep_symbol,
            max_input_length,
            max_output_length,
        )

    def split_sent(self, input: str):
        sentences = input.split("[SEP]")
        sentence_1 = sentences[0].replace("sentence 1:", "").strip()
        sentence_2 = sentences[1].replace("sentence 2:", "").strip()
        return (sentence_1, sentence_2)
    
    def load_train_examples(self):
        train_samples_path = f"data/data-{self.few_shot_numbers}/{self.file_name}/{self.file_name}_{self.few_shot_numbers}_{self.seed}_train.jsonl"
        train_samples_list = read_jsonl(train_samples_path)
        train_samples_list = [
            InputExample(
                task=e["task"],
                input=e["input"],
                sentence1=self.split_sent(e["input"])[0],
                sentence2=self.split_sent(e["input"])[1],
                gt_label_name=self.to_tf[e["output"]],
                options=[self.to_tf[option] for option in e["options"]],
            )
            for e in train_samples_list
        ]
        return train_samples_list
    
    def load_dev_examples(self):
        dev_samples_path = f"data/data-{self.few_shot_numbers}/{self.file_name}/{self.file_name}_{self.few_shot_numbers}_{self.seed}_dev.jsonl"
        dev_samples_list = read_jsonl(dev_samples_path)
        dev_samples_list = [
            InputExample(
                task=e["task"],
                input=e["input"],
                sentence1=self.split_sent(e["input"])[0],
                sentence2=self.split_sent(e["input"])[1],
                gt_label_name=self.to_tf[e["output"]],
                options=[self.to_tf[option] for option in e["options"]],
            )
            for e in dev_samples_list
        ]
        return dev_samples_list
    
    def load_test_examples(self):
        test_samples_path = f"data/data-{self.few_shot_numbers}/{self.file_name}/{self.file_name}_{self.few_shot_numbers}_{self.seed}_test.jsonl"
        test_samples_list = read_jsonl(test_samples_path)
        test_samples_list = [
            InputExample(
                task=e["task"],
                input=e["input"],
                sentence1=self.split_sent(e["input"])[0],
                sentence2=self.split_sent(e["input"])[1],
                gt_label_name=self.to_tf[e["output"]],
                options=[self.to_tf[option] for option in e["options"]],
            )
            for e in test_samples_list
        ]
        return test_samples_list


@register_dataset
class SST2Dataset(ClassificationDataset):
    name = "glue-sst2"
    input_format = "sentiment1"
    task = "sentiment"

@register_dataset
class FinancialPhraseBank(ClassificationDataset):
    name = "financial_phrasebank"
    input_format = "sentiment1"
    task = "sentiment"


@register_dataset
class TweetEvalEmotionDataset(ClassificationDataset):
    name = "tweet_eval-emotion"
    input_format = "sentiment1"
    task = "sentiment"


@register_dataset
class PoemSentimentDataset(ClassificationDataset):
    name = "poem_sentiment"
    input_format = "sentiment1"
    task = "sentiment"


@register_dataset
class TRECDataset(ClassificationDataset):
    name = "trec"
    input_format = "topic1"
    task = "topic"


@register_dataset
class TweetEvalAtheismDataset(ClassificationDataset):
    name = "tweet_eval-stance_atheism"
    input_format = "tweet_atheism1"
    task = "tweet_atheism"


@register_dataset
class TweetEvalFeministDataset(ClassificationDataset):
    name = "tweet_eval-stance_feminist"
    input_format = "tweet_feminist1"
    task = "tweet_feminist"
    

@register_dataset
class TweetEvalHateDataset(ClassificationDataset):
    name = "tweet_eval-hate"
    input_format = "hate_speech1"
    task = "hate_speech"


@register_dataset    
class EthosRaceDataset(ClassificationDataset):
    name = "ethos-race"
    input_format = "hate_speech_race1"
    task = "hate_speech_race"
    

@register_dataset    
class EthosGenderDataset(ClassificationDataset):
    name = "ethos-gender"
    input_format = "hate_speech_gender1"
    task = "hate_speech_gender"
    

@register_dataset    
class EthosReligionDataset(ClassificationDataset):
    name = "ethos-religion"
    input_format = "hate_speech_religion1"
    task = "hate_speech_religion"

@register_dataset    
class EthosNationalOriginDataset(ClassificationDataset):
    name = "ethos-national_origin"
    input_format = "hate_speech_national1"
    task = "hate_speech_national"

@register_dataset
class SICKDataset(NLIDataset):
    name = "sick"
    input_format = "nli1"
    task = "nli"


@register_dataset
class WNLIDataset(NLIDataset):
    name = "glue-wnli"
    input_format = "nli_entailment1"
    task = "nli_entailment"
    

@register_dataset
class MRPCDataset(NLIDataset):
    name = "glue-mrpc"
    input_format = "paraphrase1"
    task = "paraphrase"
    
    
@register_dataset
class SNLIDataset(NLIDataset):
    name = "snli"
    input_format = "nli1"
    task = "nli"

@register_dataset
class ADECorpusClassification(ClassificationDataset):
    name = "ade_corpus_v2-classification"

@register_dataset
class AGNews(ClassificationDataset):
    name = "ag_news"

@register_dataset
class Anli(NLIDataset):
    name = "anli"

@register_dataset
class Art(ClassificationDataset):
    name = "art"

@register_dataset
class Blimp_ana(ClassificationDataset):
    name = "blimp-anaphor_number_agreement"

@register_dataset
class Blimp_eli(ClassificationDataset):
    name = "blimp-ellipsis_n_bar_2"

@register_dataset
class Blimp_lp(ClassificationDataset):
    name = "blimp-sentential_negation_npi_licensor_present"

@register_dataset
class Blimp_s(ClassificationDataset):
    name = "blimp-sentential_negation_npi_scope"
    
@register_dataset
class Boolq(ClassificationDataset):
    name = "boolq"
    
@register_dataset
class circa(ClassificationDataset):
    name = "circa"
    
@register_dataset
class Climate_fever(ClassificationDataset):
    name = "climate_fever"
    
@register_dataset
class Crows_pairs(ClassificationDataset):
    name = "crows_pairs"
    
@register_dataset
class Ethos_directed_generalized(ClassificationDataset):
    name = "ethos-directed_vs_generalized"
    
@register_dataset
class Ethos_disability(ClassificationDataset):
    name = "ethos-disability"
    
@register_dataset
class Ethos_sexual_orientation(ClassificationDataset):
    name = "ethos-sexual_orientation"
    
@register_dataset
class Glue_cola(ClassificationDataset):
    name = "glue-cola"
    
@register_dataset
class Glue_mnli(NLIDataset):
    name = "glue-mnli"
    
@register_dataset
class Glue_qnli(NLIDataset):
    name = "glue-qnli"
    
@register_dataset
class Glue_qqp(ClassificationDataset):
    name = "glue-qqp"
    
@register_dataset
class Glue_rte(NLIDataset):
    name = "glue-rte"
    
@register_dataset
class Google_wellformed_query(ClassificationDataset):
    name = "google_wellformed_query"
    
@register_dataset
class Hate_speech_offensive(ClassificationDataset):
    name = "hate_speech_offensive"
    
@register_dataset
class Hate_speech18(ClassificationDataset):
    name = "hate_speech18"
    
@register_dataset
class Hatexplain(ClassificationDataset):
    name = "hatexplain"
    
@register_dataset
class Health_fact(ClassificationDataset):
    name = "health_fact"
        
@register_dataset
class Imdb(ClassificationDataset):
    name = "imdb"
    
@register_dataset
class Liar(ClassificationDataset):
    name = "liar"
    
@register_dataset
class Mc_taco(ClassificationDataset):
    name = "mc_taco"
    
@register_dataset
class Medical_questions_pairs(ClassificationDataset):
    name = "medical_questions_pairs"
    
@register_dataset
class Numer_sense(ClassificationDataset):
    name = "numer_sense"
    
@register_dataset
class Paws(ClassificationDataset):
    name = "paws"
    
@register_dataset
class Piqa(ClassificationDataset):
    name = "piqa"
        
@register_dataset
class Rotten_tomatoes(ClassificationDataset):
    name = "rotten_tomatoes"
    
@register_dataset
class Scitail(NLIDataset):
    name = "scitail"
    
@register_dataset
class Sms_spam(ClassificationDataset):
    name = "sms_spam"
    
@register_dataset
class Superglue_cb(NLIDataset):
    name = "superglue-cb"
    
@register_dataset
class Superglue_rte(NLIDataset):
    name = "superglue-rte"
    
@register_dataset
class Superglue_wic(ClassificationDataset):
    name = "superglue-wic"
    
@register_dataset
class Superglue_wsc(ClassificationDataset):
    name = "superglue-wsc"

@register_dataset
class Tweet_eval_irony(ClassificationDataset):
    name = "tweet_eval-irony"
    
@register_dataset
class Tweet_eval_offensive(ClassificationDataset):
    name = "tweet_eval-offensive"
    
@register_dataset
class Tweet_eval_sentiment(ClassificationDataset):
    name = "tweet_eval-sentiment"
    
@register_dataset
class Tweet_eval_stance_abortion(ClassificationDataset):
    name = "tweet_eval-stance_abortion"
    
@register_dataset
class Tweet_eval_stance_climate(ClassificationDataset):
    name = "tweet_eval-stance_climate"
    
@register_dataset
class Tweet_eval_stance_hillary(ClassificationDataset):
    name = "tweet_eval-stance_hillary"
    
@register_dataset
class Wiki_qa(ClassificationDataset):
    name = "wiki_qa"
    
@register_dataset
class Wiqa(ClassificationDataset):
    name = "wiqa"
    
@register_dataset
class Yelp_polarity(ClassificationDataset):
    name = "yelp_polarity"