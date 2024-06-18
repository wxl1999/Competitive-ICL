class InputExample:
    """
    A single training/test example.
    """
    def __init__(
        self, task: str=None, input: str=None, options: str=None, sentence1: str=None, sentence2: str=None, gt_label_name: str=None, random_gt_label_name: str=None, map_gt_label_name: str=None
    ):
        self.task = task
        self.input = input
        self.options = options
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.gt_label_name = gt_label_name
        self.random_gt_label_name = random_gt_label_name
        self.map_gt_label_name = map_gt_label_name