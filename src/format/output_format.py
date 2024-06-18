from abc import ABC, abstractmethod
from src.format.input_example import InputExample

OUTPUT_FORMATS = {}

def register_output_format(format_class):
    OUTPUT_FORMATS[format_class.name] = format_class
    return format_class

class BaseOutputFormat(ABC):
    name = None
    link_str = "\nOutput: "
    seed = None

    @abstractmethod
    def format_output(self, example: InputExample, evaluation_type: str) -> str:
        """
        Format output in augmented natural language.
        """
        raise NotImplementedError

@register_output_format
class ClassificationFormat(BaseOutputFormat):
    """
    Output format for Classification datasets
    """
    name = 'classification'

    def format_output(self, example: InputExample, evaluation_type: str) -> str:
        """
        Get output in augmented natural language, for example:
        [belief] hotel price range cheap , hotel type hotel , duration two [belief]
        """
        if evaluation_type == "golden":
            return example.gt_label_name
        elif evaluation_type == "random":
            return example.random_gt_label_name
        else:
            evaluation_type.startswith("abstract")
            return example.map_gt_label_name
